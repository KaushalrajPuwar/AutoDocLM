"""
Cross-File Call Extractor (Step 5.4).

Builds a map of function calls that cross file boundaries, using:
  Tier 1 — Python: ast CallVisitor
  Tier 2 — JS/TS:  tree-sitter call_expression walker

Key design decisions:
- Module path conversion uses package_roots.file_to_module_path() so paths like
  src/flask/globals.py are correctly mapped to flask.globals (not src.flask.globals).
- The symbol map uses only EXACT full-path keys. The "dotted suffix cascade" that
  registered every trailing suffix has been removed. This prevents namespace
  flattening where commonly-named symbols like 'config' or 'utils' resolve to the
  wrong file.
- Unresolvable calls (external libraries) are silently skipped at DEBUG level.
"""
import ast
import json
import logging
from pathlib import Path

from src.analysis.package_roots import file_to_module_path

logger = logging.getLogger(__name__)

_JS_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTENSIONS = {".ts", ".tsx"}


# ---------------------------------------------------------------------------
# Tier 1: Python AST call extraction
# ---------------------------------------------------------------------------

class _PythonCallVisitor(ast.NodeVisitor):
    """
    Walks a Python AST, collecting all call expressions inside function bodies.
    Resolves aliases from the file's import alias map.
    """

    def __init__(self, file_alias_map: dict):
        self.calls: dict[str, list[str]] = {}
        self.current_function: str | None = None
        self.file_alias_map = file_alias_map

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev = self.current_function
        self.current_function = node.name
        if node.name not in self.calls:
            self.calls[node.name] = []
        self.generic_visit(node)
        self.current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        if not self.current_function:
            self.generic_visit(node)
            return

        call_name: str | None = None

        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            temp = node.func
            while isinstance(temp, ast.Attribute):
                parts.insert(0, temp.attr)
                temp = temp.value
            if isinstance(temp, ast.Name):
                parts.insert(0, temp.id)
            call_name = ".".join(parts)

        if call_name:
            # Resolve the leading name through the import alias map
            parts = call_name.split(".")
            if parts[0] in self.file_alias_map:
                resolved_base = self.file_alias_map[parts[0]]
                call_name = ".".join([resolved_base] + parts[1:])
            self.calls[self.current_function].append(call_name)

        self.generic_visit(node)


def _extract_python_calls(
    file_str: str,
    file_path: Path,
    file_alias_map: dict,
    symbol_map: dict[str, str],
) -> dict[str, dict]:
    """Return cross-file call edges for one Python file."""
    result: dict[str, dict] = {}
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=file_str)
        visitor = _PythonCallVisitor(file_alias_map)
        visitor.visit(tree)

        for calling_fn, called_symbols in visitor.calls.items():
            for symbol in called_symbols:
                target_file = symbol_map.get(symbol)
                if target_file and target_file != file_str:
                    if calling_fn not in result:
                        result[calling_fn] = {"calls": []}
                    call_info = {
                        "file": target_file,
                        "function": symbol.split(".")[-1],
                    }
                    if call_info not in result[calling_fn]["calls"]:
                        result[calling_fn]["calls"].append(call_info)

    except Exception as e:
        logger.warning(f"Could not extract Python calls from {file_str}: {e}")

    return result


# ---------------------------------------------------------------------------
# Tier 2: JS/TS tree-sitter call extraction
# ---------------------------------------------------------------------------

def _get_js_ts_language(suffix: str):
    """Lazy-load the appropriate tree-sitter Language object."""
    try:
        if suffix in _TS_EXTENSIONS:
            import tree_sitter_typescript as tsts
            from tree_sitter import Language
            return Language(tsts.language_typescript())
        else:
            import tree_sitter_javascript as tsjs
            from tree_sitter import Language
            return Language(tsjs.language())
    except Exception as e:
        logger.warning(f"tree-sitter unavailable for {suffix}: {e}")
        return None


def _extract_js_ts_calls(
    file_str: str,
    file_path: Path,
    file_alias_map: dict,
    symbol_map: dict[str, str],
) -> dict[str, dict]:
    """
    Return cross-file call edges for one JS/TS file using tree-sitter.
    Finds call_expression nodes and resolves them against the alias map.
    """
    result: dict[str, dict] = {}
    suffix = Path(file_str).suffix.lower()
    language = _get_js_ts_language(suffix)
    if language is None:
        return result

    try:
        from tree_sitter import Parser
        parser = Parser(language)
        content = file_path.read_bytes()
        tree = parser.parse(content)
    except Exception as e:
        logger.warning(f"tree-sitter parse failed for {file_str}: {e}")
        return result

    current_function: list[str | None] = [None]

    def walk(node):
        if node.type in (
            "function_declaration", "function_expression",
            "arrow_function", "method_definition",
        ):
            name_node = next(
                (c for c in node.children if c.type == "identifier"), None
            )
            fn_name = name_node.text.decode("utf-8") if name_node else "<anonymous>"
            prev = current_function[0]
            current_function[0] = fn_name
            for child in node.children:
                walk(child)
            current_function[0] = prev
            return

        if node.type == "call_expression" and current_function[0]:
            fn_node = node.children[0] if node.children else None
            call_name: str | None = None
            if fn_node:
                if fn_node.type == "identifier":
                    call_name = fn_node.text.decode("utf-8")
                elif fn_node.type == "member_expression":
                    call_name = fn_node.text.decode("utf-8")

            if call_name:
                parts = call_name.split(".")
                if parts[0] in file_alias_map:
                    resolved_base = file_alias_map[parts[0]]
                    call_name = ".".join([resolved_base] + parts[1:])

                target_file = symbol_map.get(call_name)
                if target_file and target_file != file_str:
                    fn = current_function[0]
                    if fn not in result:
                        result[fn] = {"calls": []}
                    call_info = {
                        "file": target_file,
                        "function": call_name.split(".")[-1],
                    }
                    if call_info not in result[fn]["calls"]:
                        result[fn]["calls"].append(call_info)

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_cross_file_calls(
    repo_path: Path,
    chunks_path: Path,
    import_graph_path: Path,
    output_path: Path,
):
    """
    Build a map of function calls that cross file boundaries.

    The symbol map uses EXACT full-path module keys only. The "dotted suffix
    cascade" approach (which registered every trailing suffix of a symbol to
    handle Flask's src/ layout) has been removed. Module paths are now derived
    correctly via file_to_module_path(), which strips the package root prefix so
    that src/flask/globals.py produces key "flask.globals" directly — matching how
    user code actually imports it.

    Args:
        repo_path:          Absolute path to the raw repository root.
        chunks_path:        Path to chunks.jsonl.
        import_graph_path:  Path to import_graph.json (must include alias_map
                            and package_roots produced by build_import_graph).
        output_path:        Path to write cross_file_calls.json.
    """
    logger.info("Starting cross-file call extraction...")

    try:
        with open(import_graph_path, "r", encoding="utf-8") as f:
            import_graph_data = json.load(f)
        alias_map: dict[str, dict] = import_graph_data.get("alias_map", {})
        package_roots: list[str] = import_graph_data.get("package_roots", ["."])
    except FileNotFoundError:
        logger.error(f"Import graph not found at {import_graph_path}")
        return

    # Build symbol → file map using EXACT keys only
    # Key format: "<module.path>.<SymbolName>" e.g. "flask.globals.g"
    symbol_map: dict[str, str] = {}

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                lang = chunk.get("language", "").lower()
                file_rel = chunk.get("file", "")
                symbol = chunk.get("symbol")
                if not symbol or not file_rel:
                    continue

                if lang == "python":
                    # Correctly derive module path respecting the package root
                    module_path = file_to_module_path(file_rel, package_roots)
                    full_key = f"{module_path}.{symbol}"
                    symbol_map[full_key] = file_rel
                    # Also register the bare module path → file for module-level calls
                    if module_path not in symbol_map:
                        symbol_map[module_path] = file_rel

                elif lang in ("javascript", "typescript"):
                    module_key = file_rel
                    symbol_map[f"{module_key}.{symbol}"] = file_rel
                    if symbol not in symbol_map:
                        symbol_map[symbol] = file_rel

    except FileNotFoundError:
        logger.error(f"Chunks file not found at {chunks_path}")
        return

    # Iterate over every aliased file in the import graph
    cross_calls: dict[str, dict] = {}

    for file_str, file_aliases in alias_map.items():
        file_path = repo_path / file_str
        if not file_path.is_file():
            continue

        suffix = Path(file_str).suffix.lower()

        if suffix == ".py":
            calls = _extract_python_calls(
                file_str, file_path, file_aliases, symbol_map
            )
        elif suffix in _JS_EXTENSIONS | _TS_EXTENSIONS:
            calls = _extract_js_ts_calls(
                file_str, file_path, file_aliases, symbol_map
            )
        else:
            continue

        if calls:
            cross_calls[file_str] = calls

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cross_calls, f, indent=2)

    total_edges = sum(
        len(fn_data["calls"])
        for file_data in cross_calls.values()
        for fn_data in file_data.values()
    )
    logger.info(
        f"Cross-file call extraction complete. "
        f"{len(cross_calls)} files with cross-calls, {total_edges} total edges → {output_path}"
    )
