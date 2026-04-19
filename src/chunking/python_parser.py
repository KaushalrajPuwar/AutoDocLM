"""
Python AST-based chunker (Tier 1).
Extracts functions, methods, and classes with full decorator capture and
qualified symbol names (ClassName.method_name).
"""
import ast
import logging
from pathlib import Path
from typing import List, Optional

from src.chunking.models import Chunk

logger = logging.getLogger(__name__)

MIN_FUNCTION_LINES = 3


def _unparse_decorator(node: ast.expr) -> str:
    """Convert a decorator AST node to its source string representation."""
    try:
        return "@" + ast.unparse(node)
    except Exception:
        return "@<unknown_decorator>"


def _get_source_slice(source_lines: List[str], start_line: int, end_line: int) -> str:
    """Extract source text from 1-based line numbers (inclusive)."""
    return "\n".join(source_lines[start_line - 1 : end_line])


def _extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract the docstring from a function or class node, if present."""
    if not isinstance(node.body, list) or not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first.value.value.strip()
    return None


def parse_python_file(file_path: Path, rel_path: str) -> List[Chunk]:
    """
    Parse a Python source file and extract semantically meaningful chunks.

    Chunking rules (from PIPELINE.md):
    - Top-level functions and async functions → chunk_type: "function"
    - Classes (header + docstring only) → chunk_type: "class"
    - Methods within classes → chunk_type: "method", symbol is "ClassName.method"
    - Decorators are captured and included in chunk_text
    - Functions < MIN_FUNCTION_LINES are skipped (trivial helpers)
    - Import statements are NOT chunked here (handled in static analysis step)

    Args:
        file_path: Absolute path to the .py file.
        rel_path: Relative path stored in chunk metadata.

    Returns:
        List of Chunk objects extracted from the file.
    """
    chunks: List[Chunk] = []

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Cannot read {rel_path}: {e}")
        return chunks

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.warning(f"SyntaxError in {rel_path}: {e}")
        return chunks

    source_lines = source.splitlines()

    def process_node(node: ast.AST, parent_class: Optional[str] = None):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            decorators = [_unparse_decorator(d) for d in node.decorator_list]
            docstring = _extract_docstring(node)

            # Chunk for the class itself (header + docstring only)
            class_chunk_text = _get_source_slice(source_lines, node.lineno, node.end_lineno)
            chunks.append(Chunk(
                chunk_id=f"{rel_path}::{class_name}",
                file=rel_path,
                language="python",
                symbol=class_name,
                chunk_type="class",
                parent_class=None,
                decorators=decorators,
                line_start=node.lineno,
                line_end=node.end_lineno,
                chunk_text=class_chunk_text,
                docstring=docstring,
            ))

            # Process methods inside the class
            for child in ast.iter_child_nodes(node):
                process_node(child, parent_class=class_name)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            num_lines = node.end_lineno - node.lineno + 1
            
            # Skip trivially small functions unless they are methods
            if num_lines < MIN_FUNCTION_LINES and parent_class is None:
                return

            decorators = [_unparse_decorator(d) for d in node.decorator_list]
            docstring = _extract_docstring(node)

            # Qualified symbol name (e.g., "UserView.get" or "standalone_func")
            qualified_name = f"{parent_class}.{func_name}" if parent_class else func_name
            chunk_type = "method" if parent_class else "function"

            # Include decorators in line extraction (decorators start before the def line)
            decorator_start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            chunk_text = _get_source_slice(source_lines, decorator_start, node.end_lineno)

            # For methods, prefix with parent class signature for context
            if parent_class:
                chunk_text = f"# class {parent_class}:\n{chunk_text}"

            chunks.append(Chunk(
                chunk_id=f"{rel_path}::{qualified_name}",
                file=rel_path,
                language="python",
                symbol=qualified_name,
                chunk_type=chunk_type,
                parent_class=parent_class,
                decorators=decorators,
                line_start=decorator_start,
                line_end=node.end_lineno,
                chunk_text=chunk_text,
                docstring=docstring,
            ))

    for node in ast.iter_child_nodes(tree):
        process_node(node)

    return chunks
