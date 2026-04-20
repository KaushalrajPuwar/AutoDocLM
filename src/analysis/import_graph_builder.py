"""
Import Graph Builder (Step 5.3).

Builds a directed graph of all internal module dependencies in the repository.

  Tier 1 — Python files: ast-based import extraction
  Tier 2 — JS/TS files: tree-sitter-based import extraction

Key design decisions:
- The file list is derived from the classified_files.json manifest, NOT from
  rglob, so .venv/, build/, __pycache__/ etc. are never considered.
- Python module paths are derived via package_roots.detect_package_roots() which
  honours explicit pyproject.toml/setup.cfg declarations, flat layouts (most common),
  and src/ layouts. No fuzzy suffix scanning is used.
- All import edges that cannot be resolved to a known repo file are silently ignored
  (they are external library imports).
"""
import ast
import json
import logging
import os
from pathlib import Path

import networkx as nx

from src.analysis.package_roots import (
    detect_package_roots,
    resolve_import_to_file,
    file_to_module_path,
)

logger = logging.getLogger(__name__)

_JS_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTENSIONS = {".ts", ".tsx"}


# ---------------------------------------------------------------------------
# Tier 1: Python AST import extraction
# ---------------------------------------------------------------------------

class _PythonImportVisitor(ast.NodeVisitor):
    """AST visitor that extracts all imports and aliases from a Python file."""

    def __init__(self, file_path: str, repo_root: Path, package_roots: list[str]):
        self.imports: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.file_path = file_path
        self.repo_root = repo_root
        self.package_roots = package_roots
        self.current_file_dir = Path(file_path).parent

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.asname:
                self.aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module

        if module is None:
            # e.g. `from . import something` — resolve relative to current dir
            base_path = self.current_file_dir
            for _ in range(node.level - 1):
                base_path = base_path.parent
            for alias in node.names:
                try:
                    resolved_path_str = os.path.normpath(str(base_path / alias.name))
                    module_name = file_to_module_path(resolved_path_str, self.package_roots)
                except Exception:
                    module_name = alias.name
                self.imports.add(module_name)
                target = alias.asname or alias.name
                self.aliases[target] = module_name
            return

        # Handle relative imports (e.g., from .utils import x)
        if node.level > 0:
            base_path = self.current_file_dir
            for _ in range(node.level - 1):
                base_path = base_path.parent
            try:
                resolved_path_str = os.path.normpath(str(base_path / module.replace(".", os.sep)))
                module_name = file_to_module_path(resolved_path_str, self.package_roots)
            except Exception:
                module_name = module
        else:
            module_name = module

        self.imports.add(module_name)
        for alias in node.names:
            full_symbol_name = f"{module_name}.{alias.name}"
            target = alias.asname or alias.name
            self.aliases[target] = full_symbol_name
        self.generic_visit(node)


def _extract_python_imports(
    file_str: str,
    file_path: Path,
    repo_root: Path,
    package_roots: list[str],
    all_repo_py_files: list[str],
    graph: nx.DiGraph,
    alias_map: dict,
):
    """Parse one Python file and add import edges to the graph."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=file_str)
        visitor = _PythonImportVisitor(file_str, repo_root, package_roots)
        visitor.visit(tree)
        alias_map[file_str] = visitor.aliases

        for imp in visitor.imports:
            resolved = resolve_import_to_file(imp, package_roots, all_repo_py_files)
            if resolved and resolved != file_str:
                graph.add_node(resolved, type="file")
                graph.add_edge(file_str, resolved)

    except Exception as e:
        logger.warning(f"Could not parse {file_str} for Python imports: {e}")


# ---------------------------------------------------------------------------
# Tier 2: JS/TS tree-sitter import extraction
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
        logger.warning(f"tree-sitter language unavailable for {suffix}: {e}")
        return None


def _resolve_js_import_to_file(
    raw_spec: str, caller_path: str, all_js_files: list[str]
) -> str | None:
    """
    Resolve a JS/TS import specifier to a repo-relative file path.
    Only handles relative paths (./foo, ../bar). External packages are skipped.
    """
    if not raw_spec.startswith("."):
        return None  # External package — skip

    caller_dir = Path(caller_path).parent
    candidate = (caller_dir / raw_spec).as_posix()
    file_set = set(all_js_files)

    for ext in [".js", ".ts", ".jsx", ".tsx", ".mjs", "/index.js", "/index.ts"]:
        check = candidate + ext if not candidate.endswith(ext) else candidate
        if check in file_set:
            return check
        check2 = candidate + ext
        if check2 in file_set:
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

                for child in node.children:
                    if child.type in ("import_clause", "named_imports"):
                        for spec in child.children:
                            if spec.type == "import_specifier":
                                names = [
                                    c.text.decode("utf-8")
                                    for c in spec.children
                                    if c.type == "identifier"
                                ]
                                if len(names) == 2:
                                    file_aliases[names[1]] = f"{raw_spec}.{names[0]}"
                                elif len(names) == 1:
                                    file_aliases[names[0]] = f"{raw_spec}.{names[0]}"

        if node.type == "call_expression":
            fn = node.children[0] if node.children else None
            if fn and fn.text == b"require":
                args = next((c for c in node.children if c.type == "arguments"), None)
                if args:
                    str_node = next(
                        (c for c in args.children if c.type == "string"), None
                    )
                    if str_node:
                        raw_spec = str_node.text.decode("utf-8").strip("'\"")
                        resolved = _resolve_js_import_to_file(
                            raw_spec, file_str, all_js_files
                        )
                        if resolved and resolved != file_str:
                            graph.add_node(resolved, type="file")
                            graph.add_edge(file_str, resolved)

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    alias_map[file_str] = file_aliases


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_import_graph(
    repo_path: Path,
    classified_files_path: Path,
    output_path: Path,
    include_tests: bool = False,
):
    """
    Build a directed graph of all internal module dependencies in the repository.

    The file list is derived EXCLUSIVELY from classified_files.json — no rglob
    is used. This prevents .venv/, build/, and __pycache__/ from polluting the graph.

    Module path resolution uses package_roots.detect_package_roots() which handles
    flat layouts, src/ layouts, and explicit declarations without fuzzy matching.

    Args:
        repo_path:              Absolute path to the raw repository root.
        classified_files_path:  Path to classified_files.json.
        output_path:            Path to write import_graph.json.
        include_tests:          If True, include TEST-classified files in the graph.
    """
    logger.info("Starting import graph extraction...")

    try:
        with open(classified_files_path, "r", encoding="utf-8") as f:
            classified_files: dict = json.load(f)
    except FileNotFoundError:
        logger.error(f"Classified files manifest not found at {classified_files_path}")
        return

    # Detect package roots once — used for all Python file resolution
    package_roots = detect_package_roots(repo_path)
    logger.info(f"Detected package roots: {package_roots}")

    graph = nx.DiGraph()
    alias_map: dict[str, dict] = {}

    # Build file lists from the manifest (not rglob) so we only see user files
    all_py_files = [
        f for f, meta in classified_files.items()
        if f.endswith(".py")
    ]
    all_js_files = [
        f for f, meta in classified_files.items()
        if Path(f).suffix.lower() in _JS_EXTENSIONS | _TS_EXTENSIONS
    ]

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
            _extract_python_imports(
                file_str, file_path, repo_path, package_roots,
                all_py_files, graph, alias_map
            )
        elif suffix in _JS_EXTENSIONS | _TS_EXTENSIONS:
            _extract_js_ts_imports(
                file_str, file_path, repo_path, all_js_files, graph, alias_map
            )

    graph_data = nx.node_link_data(graph)
    graph_data["alias_map"] = alias_map
    graph_data["package_roots"] = package_roots  # Persist for downstream steps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    logger.info(
        f"Import graph extraction complete. "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges → {output_path}"
    )
