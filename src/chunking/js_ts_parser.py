"""
Tree-sitter-based chunker for JavaScript and TypeScript (Tier 2).
Uses the Subtractive Isolation strategy — same guarantee as the Python parser:
no line of code appears in more than one chunk.

Two-pass algorithm:
  Pass 1 (collect): Walk the tree-sitter AST and record every node that will
                    be promoted to its own chunk, with absolute byte offsets.
  Pass 2 (isolate): For each collected class node, subtract the byte ranges
                    of its direct promoted children via isolation.py.
"""
import logging
from pathlib import Path
from typing import List, Optional

import tree_sitter_javascript as tsj
import tree_sitter_typescript as tst
from tree_sitter import Language, Node, Parser

from src.chunking.models import Chunk
from src.chunking.isolation import subtract_children, make_relative_ranges

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_name(node: Node, source: bytes) -> Optional[str]:
    """
    Extracts the name of a function or class node.
    - JS/JSX: first `identifier` or `property_identifier` child
    - TS/TSX: first `type_identifier` child
    """
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return _node_text(child, source)
    return None


def _get_decorators(node: Node, source: bytes) -> List[str]:
    """
    In tree-sitter-javascript, decorators appear as `decorator` nodes that
    are siblings immediately before the class/method they decorate.
    """
    decorators: List[str] = []
    if node.parent is None:
        return decorators
    siblings = list(node.parent.children)
    idx = next((i for i, c in enumerate(siblings) if c.id == node.id), -1)
    if idx == -1:
        return decorators
    for i in range(idx - 1, -1, -1):
        sib = siblings[i]
        if sib.type == "decorator":
            decorators.insert(0, _node_text(sib, source))
        elif sib.is_named:
            break
    return decorators


def _line_of(node: Node) -> tuple[int, int]:
    """Return 1-based (start_line, end_line)."""
    return node.start_point[0] + 1, node.end_point[0] + 1


# ---------------------------------------------------------------------------
# Pass 1: Collect promoted nodes
# ---------------------------------------------------------------------------

class _PromotedNode:
    """Holds a discovered tree-sitter node with metadata for chunk production."""
    __slots__ = (
        "node", "rel_path", "language", "symbol", "chunk_type",
        "parent_class", "decorators", "start_byte", "end_byte",
        "line_start", "line_end",
    )

    def __init__(
        self,
        node: Node,
        rel_path: str,
        language: str,
        symbol: str,
        chunk_type: str,
        parent_class: Optional[str],
        decorators: List[str],
    ):
        self.node = node
        self.rel_path = rel_path
        self.language = language
        self.symbol = symbol
        self.chunk_type = chunk_type
        self.parent_class = parent_class
        self.decorators = decorators
        self.start_byte = node.start_byte
        self.end_byte = node.end_byte
        self.line_start, self.line_end = _line_of(node)


def _collect(
    node: Node,
    source: bytes,
    rel_path: str,
    language: str,
    parent_class: Optional[str] = None,
) -> list[_PromotedNode]:
    """
    Recursively walk the tree-sitter AST and collect all nodes that will
    become their own chunks.

    Rules:
    - Class declarations/expressions → always promoted.
    - Functions/methods → promoted if they are methods, or if they are
      >= MIN_FUNCTION_LINES as a standalone top-level function.
    - Nested closures (functions inside functions) → NOT promoted.
    """
    promoted: list[_PromotedNode] = []

    if node.type in _CLASS_TYPES:
        class_name = _node_name(node, source) or "AnonymousClass"
        decorators = _get_decorators(node, source)
        line_start, line_end = _line_of(node)

        promoted.append(_PromotedNode(
            node=node,
            rel_path=rel_path,
            language=language,
            symbol=class_name,
            chunk_type="class",
            parent_class=None,
            decorators=decorators,
        ))
        # Recurse into the class body to find methods / nested classes
        for child in node.children:
            promoted.extend(
                _collect(child, source, rel_path, language, parent_class=class_name)
            )

    elif node.type in _FUNCTION_TYPES:
        func_name = _node_name(node, source) or "anonymous"
        line_start, line_end = _line_of(node)
        num_lines = line_end - line_start + 1

        # Skip trivially small top-level functions
        if num_lines < MIN_FUNCTION_LINES and parent_class is None:
            return promoted

        decorators = _get_decorators(node, source)
        chunk_type = "method" if parent_class else "function"
        qualified_name = f"{parent_class}.{func_name}" if parent_class else func_name

        promoted.append(_PromotedNode(
            node=node,
            rel_path=rel_path,
            language=language,
            symbol=qualified_name,
            chunk_type=chunk_type,
            parent_class=parent_class,
            decorators=decorators,
        ))
        # We do NOT recurse into function bodies — nested closures stay in
        # the parent function's chunk by design.

    else:
        # For any non-class, non-function node, pass through the current
        # parent_class context (e.g., class_body → its methods)
        for child in node.children:
            promoted.extend(
                _collect(child, source, rel_path, language, parent_class=parent_class)
            )

    return promoted


# ---------------------------------------------------------------------------
# Pass 2: Build isolated chunks
# ---------------------------------------------------------------------------

def _build_chunks(
    promoted: list[_PromotedNode],
    source: bytes,
    comment_prefix: str,
) -> list[Chunk]:
    """
    Build isolated Chunk objects from the collected promoted nodes.
    Class chunks have their children's bodies subtracted.
    """
    # Group by parent_class for fast child lookup
    by_parent: dict[Optional[str], list[_PromotedNode]] = {}
    for p in promoted:
        by_parent.setdefault(p.parent_class, []).append(p)

    chunks: list[Chunk] = []
    for p in promoted:
        parent_bytes = source[p.start_byte:p.end_byte]

        if p.chunk_type == "class":
            # Subtract all direct children
            direct_children = by_parent.get(p.symbol, [])
            children_abs = [
                (c.start_byte, c.end_byte, f"{c.chunk_type}: {c.node.type}")
                for c in direct_children
            ]
            # Use node name from symbol for the label
            children_abs_labeled = [
                (c.start_byte, c.end_byte, f"{c.chunk_type}: {c.symbol.split('.')[-1]}")
                for c in direct_children
            ]
            child_ranges = make_relative_ranges(p.start_byte, children_abs_labeled)
            chunk_text = subtract_children(parent_bytes, child_ranges, comment_prefix=comment_prefix)

        else:
            chunk_text = parent_bytes.decode("utf-8", errors="replace")
            if p.parent_class:
                prefix = f"{comment_prefix} class {p.parent_class}:"
                chunk_text = f"{prefix}\n{chunk_text}"

        chunks.append(Chunk(
            chunk_id=f"{p.rel_path}::{p.symbol}",
            file=p.rel_path,
            language=p.language,
            symbol=p.symbol,
            chunk_type=p.chunk_type,
            parent_class=p.parent_class,
            decorators=p.decorators,
            line_start=p.line_start,
            line_end=p.line_end,
            chunk_text=chunk_text,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_js_ts_file(file_path: Path, rel_path: str) -> List[Chunk]:
    """
    Parse a JS/TS/JSX/TSX file via tree-sitter and extract semantically
    meaningful chunks using the Subtractive Isolation strategy.

    Guarantees:
    - No line of code appears in more than one chunk's chunk_text.
    - Class chunks contain only signatures and placeholder comments.
    - Method/function chunks contain their full implementation.

    Args:
        file_path: Absolute path to the source file.
        rel_path:  Relative path stored in chunk metadata.

    Returns:
        List of Chunk objects extracted from the file.
    """
    chunks: list[Chunk] = []
    ext = file_path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        logger.warning(f"No parser for extension {ext} in {rel_path}")
        return chunks

    language = "typescript" if ext in (".ts", ".tsx") else "javascript"
    comment_prefix = "//"

    try:
        source = file_path.read_bytes()
    except Exception as e:
        logger.warning(f"Cannot read {rel_path}: {e}")
        return chunks

    tree = parser.parse(source)
    if tree.root_node.has_error:
        logger.warning(
            f"tree-sitter parse errors in {rel_path} (will still extract valid nodes)"
        )

    # Pass 1: collect
    promoted = _collect(tree.root_node, source, rel_path, language)

    # Pass 2: isolate and build
    return _build_chunks(promoted, source, comment_prefix)
