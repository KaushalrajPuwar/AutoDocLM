"""
Tree-sitter-based chunker for JavaScript and TypeScript (Tier 2).
Uses compiled grammars to extract classes, methods, and functions with
decorator capture and qualified symbol names.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import tree_sitter_javascript as tsj
import tree_sitter_typescript as tst
from tree_sitter import Language, Node, Parser

from src.chunking.models import Chunk

logger = logging.getLogger(__name__)

# Build pre-compiled Language instances (warm once, reuse across calls)
_JS_LANG = Language(tsj.language())
_TS_LANG = Language(tst.language_typescript())
_TSX_LANG = Language(tst.language_tsx())

_PARSERS = {
    ".js":  Parser(_JS_LANG),
    ".jsx": Parser(_JS_LANG),
    ".ts":  Parser(_TS_LANG),
    ".tsx": Parser(_TSX_LANG),
}

# Node types that define a named callable or class scope
_CLASS_TYPES = {"class_declaration", "class_expression"}
_FUNCTION_TYPES = {
    "function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
    "generator_function_declaration",
}

MIN_FUNCTION_LINES = 3


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_name(node: Node, source: bytes) -> Optional[str]:
    """
    Extracts the name of a function or class node.
    - JS/JSX: first `identifier` or `property_identifier` child
    - TS/TSX: first `type_identifier` child (TypeScript uses this for class names)
    """
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return _node_text(child, source)
    return None


def _get_decorators(node: Node, source: bytes) -> List[str]:
    """
    In tree-sitter-javascript, decorators appear as `decorator` nodes that are
    siblings immediately before the class/method they decorate. We scan the
    parent's children list backwards from our node to collect them.
    """
    decorators: List[str] = []
    if node.parent is None:
        return decorators
    siblings = list(node.parent.children)
    idx = next((i for i, c in enumerate(siblings) if c.id == node.id), -1)
    if idx == -1:
        return decorators
    # Walk backwards from the node collecting decorator siblings
    for i in range(idx - 1, -1, -1):
        sib = siblings[i]
        if sib.type == "decorator":
            decorators.insert(0, _node_text(sib, source))
        elif sib.is_named:
            break
    return decorators


def _line_of(node: Node) -> Tuple[int, int]:
    """Return 1-based (start_line, end_line)."""
    return node.start_point[0] + 1, node.end_point[0] + 1


def _walk_tree(
    node: Node,
    source: bytes,
    rel_path: str,
    language: str,
    chunks: List[Chunk],
    parent_class: Optional[str] = None,
):
    """Recursively walk tree-sitter AST and emit chunks for classes and functions."""
    if node.type in _CLASS_TYPES:
        class_name = _node_name(node, source) or "AnonymousClass"
        decorators = _get_decorators(node, source)
        line_start, line_end = _line_of(node)
        class_text = _node_text(node, source)

        chunks.append(Chunk(
            chunk_id=f"{rel_path}::{class_name}",
            file=rel_path,
            language=language,
            symbol=class_name,
            chunk_type="class",
            parent_class=None,
            decorators=decorators,
            line_start=line_start,
            line_end=line_end,
            chunk_text=class_text,
        ))

        # Recurse into the class body, passing the class name down
        for child in node.children:
            _walk_tree(child, source, rel_path, language, chunks, parent_class=class_name)

    elif node.type in _FUNCTION_TYPES:
        func_name = _node_name(node, source) or "anonymous"
        line_start, line_end = _line_of(node)
        num_lines = line_end - line_start + 1

        # Skip trivially small standalone functions
        if num_lines < MIN_FUNCTION_LINES and parent_class is None:
            return

        decorators = _get_decorators(node, source)
        chunk_type = "method" if parent_class else "function"
        qualified_name = f"{parent_class}.{func_name}" if parent_class else func_name
        chunk_text = _node_text(node, source)
        if parent_class:
            chunk_text = f"// class {parent_class}:\n{chunk_text}"

        chunks.append(Chunk(
            chunk_id=f"{rel_path}::{qualified_name}",
            file=rel_path,
            language=language,
            symbol=qualified_name,
            chunk_type=chunk_type,
            parent_class=parent_class,
            decorators=decorators,
            line_start=line_start,
            line_end=line_end,
            chunk_text=chunk_text,
        ))
        # Recurse into function body — nested functions / classes are rare but possible
        for child in node.children:
            _walk_tree(child, source, rel_path, language, chunks, parent_class=None)

    else:
        # For any other node, recursively descend unless it's a class body we already handled
        for child in node.children:
            _walk_tree(child, source, rel_path, language, chunks, parent_class=parent_class)


def parse_js_ts_file(file_path: Path, rel_path: str) -> List[Chunk]:
    """
    Parse a JS/TS/JSX/TSX file via tree-sitter and extract semantically meaningful chunks.

    Args:
        file_path: Absolute path to the source file.
        rel_path: Relative path stored in chunk metadata.

    Returns:
        List of Chunk objects extracted from the file.
    """
    chunks: List[Chunk] = []
    ext = file_path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        logger.warning(f"No parser for extension {ext} in {rel_path}")
        return chunks

    language = "typescript" if ext in (".ts", ".tsx") else "javascript"

    try:
        source = file_path.read_bytes()
    except Exception as e:
        logger.warning(f"Cannot read {rel_path}: {e}")
        return chunks

    tree = parser.parse(source)
    if tree.root_node.has_error:
        logger.warning(f"tree-sitter parse errors in {rel_path} (will still extract valid nodes)")

    _walk_tree(tree.root_node, source, rel_path, language, chunks)
    return chunks
