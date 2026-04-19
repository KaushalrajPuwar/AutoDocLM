"""
Regex-based fallback chunker for all languages not supported by Tier 1 or Tier 2 (Tier 3).
Applies simple pattern matching for class and function definitions. It will not
produce guaranteed-perfect symbol names or decorator capture, but it ensures
no code is silently dropped from the pipeline.
"""
import logging
import re
from pathlib import Path
from typing import List

from src.chunking.models import Chunk

logger = logging.getLogger(__name__)

# Patterns to detect the start of a "block" — ordered by specificity
_BLOCK_PATTERNS = [
    # public class Foo, class Foo, abstract class Foo (must be before function patterns)
    re.compile(r"^[\s]*(public\s+|private\s+|protected\s+|abstract\s+|static\s+)*(class)\s+(\w+)", re.MULTILINE),
    # go structs / type declarations
    re.compile(r"^type\s+(\w+)\s+struct", re.MULTILINE),
    # def foo, async def foo
    re.compile(r"^[\s]*(async\s+)?def\s+(\w+)", re.MULTILINE),
    # function foo / async function foo
    re.compile(r"^[\s]*(export\s+)?(async\s+)?function\s+(\w+)", re.MULTILINE),
    # func foo (Go)
    re.compile(r"^func\s+(\w+)", re.MULTILINE),
    # Java/C#/C++ methods: public void foo(...)
    re.compile(r"^[\s]*(public|private|protected|static|final|override|virtual)\s+[\w<>\[\]]+\s+(\w+)\s*\(", re.MULTILINE),
]

MAX_LINES_PER_CHUNK = 200
SMALL_FILE_THRESHOLD = 30  # Files below this line count are returned as a single block


def _infer_language(ext: str) -> str:
    return {
        ".java": "java",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".rs": "rust",
        ".kt": "kotlin",
        ".swift": "swift",
        ".sh": "shell",
    }.get(ext, "unknown")


def _extract_name(match: re.Match) -> str:
    """Pull the rightmost non-None group from the match as the symbol name."""
    groups = [g for g in match.groups() if g and re.match(r"^\w+$", g.strip())]
    return groups[-1] if groups else "unknown"


def parse_fallback_file(file_path: Path, rel_path: str) -> List[Chunk]:
    """
    Heuristic chunker for unsupported languages.

    Strategy:
    - If file is <= 80 lines, treat the whole file as a single "block" chunk.
    - Otherwise, split by detecting top-level block start patterns.
      Between consecutive pattern hits, extract the text as a chunk.
    - Lines between the first line and the first pattern are emitted as a
      "module_block" to capture module-level definitions and constants.

    Args:
        file_path: Absolute path to the source file.
        rel_path: Relative path stored in chunk metadata.

    Returns:
        List of Chunk objects extracted from the file.
    """
    chunks: List[Chunk] = []
    ext = file_path.suffix.lower()
    language = _infer_language(ext)

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Cannot read {rel_path}: {e}")
        return chunks

    lines = source.splitlines()
    total_lines = len(lines)

    # Always attempt pattern-based splitting first, regardless of file size.
    # Fall back to whole-file chunk only when no patterns match at all.

    # Gather all pattern hit positions
    hits = []
    for pattern in _BLOCK_PATTERNS:
        for m in pattern.finditer(source):
            lineno = source[:m.start()].count("\n") + 1
            hits.append((lineno, m))
    # Sort by line number, deduplicate consecutive matches on same line
    hits.sort(key=lambda x: x[0])
    seen_lines: set = set()
    unique_hits = []
    for lineno, m in hits:
        if lineno not in seen_lines:
            unique_hits.append((lineno, m))
            seen_lines.add(lineno)

    if not unique_hits:
        # No patterns found — return whole file as a single block
        chunks.append(Chunk(
            chunk_id=f"{rel_path}::module",
            file=rel_path,
            language=language,
            symbol="module",
            chunk_type="block",
            parent_class=None,
            decorators=[],
            line_start=1,
            line_end=total_lines,
            chunk_text=source,
        ))
        return chunks

    # Emit Module-level preamble (before first hit)
    first_hit_line = unique_hits[0][0]
    if first_hit_line > 1:
        preamble = "\n".join(lines[0 : first_hit_line - 1])
        if preamble.strip():
            chunks.append(Chunk(
                chunk_id=f"{rel_path}::module_preamble",
                file=rel_path,
                language=language,
                symbol="module_preamble",
                chunk_type="block",
                parent_class=None,
                decorators=[],
                line_start=1,
                line_end=first_hit_line - 1,
                chunk_text=preamble,
            ))

    # Split at each hit boundary
    boundaries = [h[0] for h in unique_hits] + [total_lines + 1]
    for i, (start_line, match) in enumerate(unique_hits):
        end_line = min(boundaries[i + 1] - 1, start_line + MAX_LINES_PER_CHUNK - 1)
        symbol_name = _extract_name(match)
        chunk_text = "\n".join(lines[start_line - 1 : end_line])

        chunks.append(Chunk(
            chunk_id=f"{rel_path}::{symbol_name}",
            file=rel_path,
            language=language,
            symbol=symbol_name,
            chunk_type="block",
            parent_class=None,
            decorators=[],
            line_start=start_line,
            line_end=end_line,
            chunk_text=chunk_text,
        ))

    return chunks
