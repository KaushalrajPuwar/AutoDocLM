"""
Subtractive Isolation Utility.

Language-agnostic text manipulation utility that removes child code blocks from
a parent source text and replaces them with single-line placeholder comments.

This is the core algorithm for the "Subtractive Isolation" chunking strategy.
It is intentionally free of any parser logic — it operates purely on byte
offsets into raw source text, which are provided by the caller (AST or
tree-sitter).

Usage:
    child_ranges = [
        (start_byte_1, end_byte_1, "method: get"),
        (start_byte_2, end_byte_2, "method: post"),
    ]
    isolated_text = subtract_children(full_source_bytes, child_ranges, comment="#")
"""
from __future__ import annotations

from typing import NamedTuple


class ChildRange(NamedTuple):
    """
    Represents a byte range of a child node to be subtracted from a parent.

    Attributes:
        start: Byte offset of the first character of the child body
               (inclusive). Should point to the start of the child's
               definition/signature, NOT just the body braces.
        end:   Byte offset just PAST the last character of the child body
               (exclusive). Same semantics as Python slice notation.
        label: Human-readable label used in the placeholder comment.
               E.g. "method: get_connection" → "# <method: get_connection> ..."
    """
    start: int
    end: int
    label: str


def subtract_children(
    source: bytes,
    children: list[ChildRange],
    comment_prefix: str = "#",
) -> str:
    """
    Return source text with each child range replaced by a placeholder comment.

    The replacement is a single line:
        # <label> ...
    or for JS/TS:
        // <label> ...

    The original indentation of the child's first line is preserved in the
    placeholder so the visual structure of the parent is maintained.

    Args:
        source:         Raw source bytes for the PARENT node's text.
                        This is the bytes of just the parent (not the whole file),
                        as returned by tree-sitter or sliced via AST line offsets.
        children:       List of ChildRange instances describing byte ranges
                        RELATIVE TO source (i.e., already offset-adjusted by
                        the caller to be 0-based within the parent slice).
        comment_prefix: The comment syntax for the language. Use "#" for
                        Python, "//" for JS/TS.

    Returns:
        The isolated source text as a str, with child bodies replaced.

    Raises:
        ValueError: If any child range is out of bounds for the given source.
    """
    if not children:
        return source.decode("utf-8", errors="replace")

    # Sort children by start byte so we can process left-to-right
    sorted_children = sorted(children, key=lambda c: c.start)

    # Validate ranges
    src_len = len(source)
    for c in sorted_children:
        if c.start < 0 or c.end > src_len or c.start >= c.end:
            raise ValueError(
                f"ChildRange({c.label}) byte range [{c.start}:{c.end}] is "
                f"out of bounds for source of length {src_len}."
            )

    # Build the result by splicing segments together
    result_parts: list[bytes] = []
    cursor = 0

    for child in sorted_children:
        # If this child starts before our cursor, it overlaps with a previous
        # child. This should not happen with a correct two-pass collector, but
        # we guard against it defensively.
        if child.start < cursor:
            continue

        # Keep the text before this child unchanged
        if child.start > cursor:
            result_parts.append(source[cursor:child.start])

        # Determine the indentation of this child's first line so the
        # placeholder lines up correctly.
        indent = _leading_whitespace(source, child.start)

        # Insert placeholder
        placeholder = f"{indent}{comment_prefix} <{child.label}> ...\n"
        result_parts.append(placeholder.encode("utf-8"))

        cursor = child.end
        # Consume any trailing newline that belongs to the child so we don't
        # end up with double blank lines.
        if cursor < src_len and source[cursor:cursor + 1] == b"\n":
            cursor += 1

    # Append the remainder after the last child
    if cursor < src_len:
        result_parts.append(source[cursor:])

    return b"".join(result_parts).decode("utf-8", errors="replace")


def make_relative_ranges(
    parent_start_byte: int,
    children_absolute: list[tuple[int, int, str]],
) -> list[ChildRange]:
    """
    Convert absolute file-level byte offsets of children to offsets that are
    relative to the parent node's start, for use with subtract_children().

    Args:
        parent_start_byte:   The absolute byte offset of the parent node's
                             start within the full source file.
        children_absolute:   List of (abs_start, abs_end, label) tuples for
                             each promoted child node.

    Returns:
        List of ChildRange with offsets relative to the parent.
    """
    return [
        ChildRange(
            start=abs_start - parent_start_byte,
            end=abs_end - parent_start_byte,
            label=label,
        )
        for abs_start, abs_end, label in children_absolute
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _leading_whitespace(source: bytes, byte_pos: int) -> str:
    """
    Return the leading whitespace of the line that contains byte_pos.
    Scans backwards from byte_pos to the nearest newline.
    """
    line_start = byte_pos
    while line_start > 0 and source[line_start - 1:line_start] != b"\n":
        line_start -= 1
    indent_bytes = []
    pos = line_start
    while pos < len(source) and source[pos:pos + 1] in (b" ", b"\t"):
        indent_bytes.append(source[pos:pos + 1])
        pos += 1
    return b"".join(indent_bytes).decode("utf-8")
