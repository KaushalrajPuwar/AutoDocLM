import ast
import json
import logging
import os
from pathlib import Path
import networkx as nx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Tier 1: Python AST import extraction
# ─────────────────────────────────────────────

class _PythonImportVisitor(ast.NodeVisitor):
    """AST visitor that extracts all imports and aliases from a Python file."""

    def __init__(self, file_path: str, repo_root: Path):
        self.imports: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.file_path = file_path
        self.repo_root = repo_root
        self.current_file_dir = Path(file_path).parent

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.asname:
                self.aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module
        if module is None:  # e.g., `from . import something`
            # Still record the individual names as imports from this package
            base_path = self.current_file_dir
            for _ in range(node.level - 1):
                base_path = base_path.parent
            for alias in node.names:
                try:
                    resolved = (base_path / alias.name).resolve().relative_to(self.repo_root)
                    module_name = str(resolved).replace(os.sep, ".")
                except ValueError:
                    module_name = alias.name
                self.imports.add(module_name)
                target = alias.asname or alias.name
                self.aliases[target] = module_name
            return

        # Handle relative imports
        if node.level > 0:
            base_path = self.current_file_dir
            for _ in range(node.level - 1):
                base_path = base_path.parent
            full_module_path = (base_path / module.replace(".", os.sep)).resolve()
            try:
                module_as_path = full_module_path.relative_to(self.repo_root)
                module_name = str(module_as_path).replace(os.sep, ".")
            except ValueError:
                module_name = module
        else:
            module_name = module

        self.imports.add(module_name)
        for alias in node.names:
            full_symbol_name = f"{module_name}.{alias.name}"
            target = alias.asname or alias.name
            self.aliases[target] = full_symbol_name
        self.generic_visit(node)


def _resolve_import_to_file(imp: str, all_files: list[str]) -> str | None:
    """
    Tries to resolve a dotted import string to a repository file path.

    Strategy (in order):
    1. Direct: `src.utils` → `src/utils.py`
    2. Package: `src.utils` → `src/utils/__init__.py`
    3. Suffix scan: look for any known file whose normalised path ends with
       the candidate path, e.g., `flask.helpers` → `src/flask/helpers.py`.
    """
    path_candidate = imp.replace(".", os.sep)

    # 1. Direct file
    potential_file = path_candidate + ".py"
    if potential_file in all_files:
        return potential_file

    # 2. Package __init__
    potential_init = os.path.join(path_candidate, "__init__.py")
    if potential_init in all_files:
        return potential_init

    # 3. Suffix scan for nested packages (e.g. flask.helpers → src/flask/helpers.py)
    file_suffix = "/" + potential_file.replace(os.sep, "/")
    init_suffix = "/" + potential_init.replace(os.sep, "/")
    for f in all_files:
        norm = "/" + f.replace(os.sep, "/")
        if norm.endswith(file_suffix) or norm.endswith(init_suffix):
            return f

    return None



def _extract_python_imports(
    file_str: str,
    file_path: Path,
    repo_root: Path,
    all_repo_py_files: list[str],
    graph: nx.DiGraph,
    alias_map: dict,
):
    """Parse one Python file and add import edges to the graph."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=file_str)
        visitor = _PythonImportVisitor(file_str, repo_root)
        visitor.visit(tree)
        alias_map[file_str] = visitor.aliases

        for imp in visitor.imports:
            resolved = _resolve_import_to_file(imp, all_repo_py_files)
            if resolved and resolved != file_str:
                graph.add_node(resolved, type="file")
                graph.add_edge(file_str, resolved)

    except Exception as e:
        logger.warning(f"Could not parse {file_str} for Python imports: {e}")


# ─────────────────────────────────────────────
# Tier 2: JS/TS tree-sitter import extraction
# ─────────────────────────────────────────────

_JS_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTENSIONS = {".ts", ".tsx"}

def _get_js_ts_language(suffix: str):
    """Lazy-load the appropriate tree-sitter Language object."""
    try:
        if suffix in _TS_EXTENSIONS:
            import tree_sitter_typescript as tstsx
            from tree_sitter import Language
            return Language(tstsx.language_typescript())
        else:
            import tree_sitter_javascript as tsjs
            from tree_sitter import Language
            return Language(tsjs.language())
    except Exception as e:
        logger.warning(f"tree-sitter language unavailable for {suffix}: {e}")
        return None


def _resolve_js_import_to_file(raw_spec: str, caller_path: str, all_js_files: list[str]) -> str | None:
    """
    Resolve a JS/TS import specifier to a repo-relative file path.
    Handles relative paths (./foo, ../bar) and bare specifiers.
    """
    if not raw_spec.startswith("."):
        return None  # External package — skip

    caller_dir = Path(caller_path).parent
    candidate = (caller_dir / raw_spec).as_posix()

    # Try with known extensions
    for ext in [".js", ".ts", ".jsx", ".tsx", ".mjs", "/index.js", "/index.ts"]:
        check = candidate + ext if not candidate.endswith(ext) else candidate
        if check in all_js_files:
            return check
        # Also try without the caller extension repeated
        check2 = candidate + ext
        if check2 in all_js_files:
            return check2

    return None


def _extract_js_ts_imports(
    file_str: str,
    file_path: Path,
    repo_root: Path,
    all_js_files: list[str],
    graph: nx.DiGraph,
    alias_map: dict,
):
    """Parse one JS/TS file with tree-sitter and add import edges."""
    suffix = Path(file_str).suffix.lower()
    language = _get_js_ts_language(suffix)
    if language is None:
        return

    try:
        from tree_sitter import Parser
        parser = Parser(language)
        content = file_path.read_bytes()
        tree = parser.parse(content)
    except Exception as e:
        logger.warning(f"tree-sitter parse failed for {file_str}: {e}")
        return

    file_aliases: dict[str, str] = {}

    def walk(node):
        # ES6: import foo from './foo'
        # ES6: import { bar } from './bar'
        if node.type in ("import_statement", "import_declaration"):
            source_node = next(
                (c for c in node.children if c.type == "string"), None
            )
            if source_node:
                raw_spec = source_node.text.decode("utf-8").strip("'\"")
                resolved = _resolve_js_import_to_file(raw_spec, file_str, all_js_files)
                if resolved and resolved != file_str:
                    graph.add_node(resolved, type="file")
                    graph.add_edge(file_str, resolved)

                # Record aliases from named imports: import { x as y } ...
                for child in node.children:
                    if child.type in ("import_clause", "named_imports"):
                        for spec in child.children:
                            if spec.type == "import_specifier":
                                names = [c.text.decode("utf-8") for c in spec.children if c.type == "identifier"]
                                if len(names) == 2:  # import { orig as alias }
                                    file_aliases[names[1]] = f"{raw_spec}.{names[0]}"
                                elif len(names) == 1:
                                    file_aliases[names[0]] = f"{raw_spec}.{names[0]}"

        # CommonJS: require('./foo')
        if node.type == "call_expression":
            fn = node.children[0] if node.children else None
            if fn and fn.text == b"require":
                args = next((c for c in node.children if c.type == "arguments"), None)
                if args:
                    str_node = next((c for c in args.children if c.type == "string"), None)
                    if str_node:
                        raw_spec = str_node.text.decode("utf-8").strip("'\"")
                        resolved = _resolve_js_import_to_file(raw_spec, file_str, all_js_files)
                        if resolved and resolved != file_str:
                            graph.add_node(resolved, type="file")
                            graph.add_edge(file_str, resolved)

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    alias_map[file_str] = file_aliases


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def build_import_graph(
    repo_path: Path,
    classified_files_path: Path,
    output_path: Path,
    include_tests: bool = False,
):
    """
    Builds a directed graph of all internal module dependencies in the repository.

    Tier 1 — Python files: ast-based
    Tier 2 — JS/TS files: tree-sitter-based

    classified_files.json schema:
        { "relative/path/to/file": { "category": "SOURCE_CODE", "is_test": false } }
    """
    logger.info("Starting import graph extraction...")

    try:
        with open(classified_files_path, "r", encoding="utf-8") as f:
            classified_files = json.load(f)
    except FileNotFoundError:
        logger.error(f"Classified files manifest not found at {classified_files_path}")
        return

    graph = nx.DiGraph()
    alias_map: dict[str, dict] = {}

    # Collect all repo files for resolution
    all_py_files = [str(p.relative_to(repo_path)) for p in repo_path.rglob("*.py")]
    all_js_files = [
        str(p.relative_to(repo_path))
        for p in repo_path.rglob("*")
        if p.suffix.lower() in _JS_EXTENSIONS | _TS_EXTENSIONS
    ]

    # classified_files is { filepath: { "category": ..., "is_test": bool } }
    for file_str, metadata in classified_files.items():
        if metadata.get("category") != "SOURCE_CODE":
            continue
        if not include_tests and metadata.get("is_test", False):
            continue

        suffix = Path(file_str).suffix.lower()
        file_path = repo_path / file_str
        if not file_path.is_file():
            continue

        graph.add_node(file_str, type="file")

        if suffix == ".py":
            _extract_python_imports(file_str, file_path, repo_path, all_py_files, graph, alias_map)
        elif suffix in _JS_EXTENSIONS | _TS_EXTENSIONS:
            _extract_js_ts_imports(file_str, file_path, repo_path, all_js_files, graph, alias_map)

    graph_data = nx.node_link_data(graph)
    graph_data["alias_map"] = alias_map

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    logger.info(
        f"Import graph extraction complete. "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges → {output_path}"
    )
