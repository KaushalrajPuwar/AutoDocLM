"""
Python AST-based chunker (Tier 1) — Subtractive Isolation Strategy.

Extracts functions, methods, and classes with full decorator capture and
qualified symbol names (ClassName.method_name).

The "Subtractive Isolation" strategy ensures that:
  - Class chunks contain ONLY the class signature, docstring, class-level
    fields, and placeholder comments for each promoted child.
  - Method/function chunks contain the FULL implementation body.
  - No line of code is ever duplicated across two chunks.

Two-pass algorithm:
  Pass 1 (collect): Walk the AST and record every node that will be promoted
                    to its own chunk, along with its absolute byte offsets.
  Pass 2 (isolate): For each collected node, subtract the byte ranges of its
                    direct promoted children before producing chunk_text.
"""
import ast
import logging
from pathlib import Path
from typing import List, Optional

from src.chunking.models import Chunk
from src.chunking.isolation import subtract_children, make_relative_ranges, ChildRange

logger = logging.getLogger(__name__)

MIN_FUNCTION_LINES = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unparse_decorator(node: ast.expr) -> str:
    """Convert a decorator AST node to its source string representation."""
    try:
        return "@" + ast.unparse(node)
    except Exception:
        return "@<unknown_decorator>"


def _extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract the docstring from a function or class node, if present."""
    if not isinstance(node.body, list) or not node.body:
        return None
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value.value.strip()
    return None


def _byte_offset(source_bytes: bytes, line: int, col: int) -> int:
    """
    Convert a 1-based (line, col) position from the Python AST into an
    absolute byte offset within source_bytes.

    Python's ast module provides end_lineno/end_col_offset which are
    1-based lines and 0-based columns.
    """
    lines = source_bytes.split(b"\n")
    offset = 0
    for i, ln in enumerate(lines):
        if i + 1 == line:
            return offset + col
        offset += len(ln) + 1  # +1 for the newline
    return offset


def _node_byte_range(node: ast.AST, source_bytes: bytes) -> tuple[int, int]:
    """
    Return the absolute byte range [start, end) of an AST node, including
    any leading decorators if present.

    Start is always at the BEGINNING of the line that contains the first
    decorator (or the def/class keyword if no decorators), so that the
    child range cleanly spans the entire visible declaration including
    the @ sign, rather than mid-line.
    """
    if hasattr(node, "decorator_list") and node.decorator_list:
        start_line = node.decorator_list[0].lineno
    else:
        start_line = node.lineno

    # Find the byte offset of the START of this line (column 0)
    # so the placeholder cleanly replaces the whole decorated block.
    lines = source_bytes.split(b"\n")
    start_byte = sum(len(lines[i]) + 1 for i in range(start_line - 1))
    end_byte = _byte_offset(source_bytes, node.end_lineno, node.end_col_offset)
    return start_byte, end_byte


# ---------------------------------------------------------------------------
# Pass 1: Collect promoted nodes
# ---------------------------------------------------------------------------

class _PromotedNode:
    """Holds a discovered AST node and the metadata needed to build its chunk."""
    __slots__ = (
        "node", "rel_path", "symbol", "chunk_type", "parent_class",
        "decorators", "docstring", "start_byte", "end_byte",
        "line_start", "line_end",
    )

    def __init__(
        self,
        node: ast.AST,
        rel_path: str,
        symbol: str,
        chunk_type: str,
        parent_class: Optional[str],
        decorators: List[str],
        docstring: Optional[str],
        start_byte: int,
        end_byte: int,
    ):
        self.node = node
        self.rel_path = rel_path
        self.symbol = symbol
        self.chunk_type = chunk_type
        self.parent_class = parent_class
        self.decorators = decorators
        self.docstring = docstring
        self.start_byte = start_byte
        self.end_byte = end_byte
        # 1-based line numbers from the AST (used in the Chunk model)
        self.line_start = node.decorator_list[0].lineno if (
            hasattr(node, "decorator_list") and node.decorator_list
        ) else node.lineno
        self.line_end = node.end_lineno


def _collect(
    tree_nodes: list,
    source_bytes: bytes,
    rel_path: str,
    parent_class: Optional[str] = None,
) -> list[_PromotedNode]:
    """
    Recursively walk AST child nodes and collect all nodes that should
    become their own chunk.

    Rules:
    - ClassDef        → always promoted (chunk_type = "class")
    - FunctionDef /
      AsyncFunctionDef → promoted if:
        - it is a method (has a parent_class), OR
        - it is >= MIN_FUNCTION_LINES long as a standalone function
    - Nested closures inside functions → NOT promoted (stay in parent body)
    """
    promoted: list[_PromotedNode] = []

    for node in tree_nodes:
        if isinstance(node, ast.ClassDef):
            decorators = [_unparse_decorator(d) for d in node.decorator_list]
            docstring = _extract_docstring(node)
            start_byte, end_byte = _node_byte_range(node, source_bytes)

            promoted.append(_PromotedNode(
                node=node,
                rel_path=rel_path,
                symbol=node.name,
                chunk_type="class",
                parent_class=None,
                decorators=decorators,
                docstring=docstring,
                start_byte=start_byte,
                end_byte=end_byte,
            ))
            # Recurse into class body (methods / nested classes)
            promoted.extend(
                _collect(
                    list(ast.iter_child_nodes(node)),
                    source_bytes,
                    rel_path,
                    parent_class=node.name,
                )
            )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            num_lines = node.end_lineno - node.lineno + 1
            # Skip trivially small top-level functions
            if num_lines < MIN_FUNCTION_LINES and parent_class is None:
                continue

            decorators = [_unparse_decorator(d) for d in node.decorator_list]
            docstring = _extract_docstring(node)
            qualified_name = (
                f"{parent_class}.{node.name}" if parent_class else node.name
            )
            chunk_type = "method" if parent_class else "function"
            start_byte, end_byte = _node_byte_range(node, source_bytes)

            promoted.append(_PromotedNode(
                node=node,
                rel_path=rel_path,
                symbol=qualified_name,
                chunk_type=chunk_type,
                parent_class=parent_class,
                decorators=decorators,
                docstring=docstring,
                start_byte=start_byte,
                end_byte=end_byte,
            ))
            # NOTE: We do NOT recurse into function bodies.
            # Nested closures stay inside their enclosing function's chunk.
            # This is intentional — closures are not independently
            # addressable symbols and their context is their parent function.

    return promoted


# ---------------------------------------------------------------------------
# Pass 2: Build isolated chunks
# ---------------------------------------------------------------------------

def _build_chunks(
    promoted: list[_PromotedNode],
    source_bytes: bytes,
) -> list[Chunk]:
    """
    For each promoted node, produce a Chunk whose chunk_text has been
    processed by subtract_children() to remove bodies of promoted children.
    """
    # Build a lookup: for each node, which other promoted nodes are its
    # DIRECT children (i.e., their parent_class matches this class symbol)?
    # For functions: we don't recurse so there are no promoted children.
    chunks: list[Chunk] = []

    # Group promoted nodes by their parent_class for fast lookup
    by_parent: dict[Optional[str], list[_PromotedNode]] = {}
    for p in promoted:
        key = p.parent_class
        by_parent.setdefault(key, []).append(p)

    for p in promoted:
        parent_bytes = source_bytes[p.start_byte:p.end_byte]

        if p.chunk_type == "class":
            # Children to subtract: all promoted nodes whose parent_class
            # is this class's symbol name.
            direct_children = by_parent.get(p.symbol, [])
            children_abs = [
                (c.start_byte, c.end_byte, _child_label(c))
                for c in direct_children
            ]
            child_ranges = make_relative_ranges(p.start_byte, children_abs)
            chunk_text = subtract_children(parent_bytes, child_ranges, comment_prefix="#")

        elif p.chunk_type in ("function", "method"):
            # Functions/methods have no promoted children — nested closures
            # stay in the body by design.
            chunk_text = parent_bytes.decode("utf-8", errors="replace")
            # For methods, prefix with parent class signature for context
            if p.parent_class:
                chunk_text = f"# class {p.parent_class}:\n{chunk_text}"

        else:
            chunk_text = parent_bytes.decode("utf-8", errors="replace")

        chunks.append(Chunk(
            chunk_id=f"{p.rel_path}::{p.symbol}",
            file=p.rel_path,
            language="python",
            symbol=p.symbol,
            chunk_type=p.chunk_type,
            parent_class=p.parent_class,
            decorators=p.decorators,
            line_start=p.line_start,
            line_end=p.line_end,
            chunk_text=chunk_text,
            docstring=p.docstring,
        ))

    return chunks


def _child_label(p: "_PromotedNode") -> str:
    """Generate a human-readable placeholder label for a child node."""
    return f"{p.chunk_type}: {p.node.name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_python_file(file_path: Path, rel_path: str) -> List[Chunk]:
    """
    Parse a Python source file and extract semantically meaningful chunks
    using the Subtractive Isolation strategy.

    Guarantees:
    - No line of code appears in more than one chunk's chunk_text.
    - Class chunks contain only signatures, docstrings, class-level fields,
      and placeholder comments for methods.
    - Method/function chunks contain their full implementation.

    Args:
        file_path: Absolute path to the .py file.
        rel_path:  Relative path stored in chunk metadata.

    Returns:
        List of Chunk objects extracted from the file.
    """
    try:
        source_bytes = file_path.read_bytes()
    except Exception as e:
        logger.warning(f"Cannot read {rel_path}: {e}")
        return []

    try:
        tree = ast.parse(source_bytes.decode("utf-8", errors="replace"))
    except SyntaxError as e:
        logger.warning(f"SyntaxError in {rel_path}: {e}")
        return []

    # Pass 1: Collect all promoted nodes (top-level only — recursion is
    # handled inside _collect for class members)
    promoted = _collect(
        list(ast.iter_child_nodes(tree)),
        source_bytes,
        rel_path,
        parent_class=None,
    )

    # Pass 2: Build isolated chunks
    return _build_chunks(promoted, source_bytes)
