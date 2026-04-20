"""
Chunking Orchestrator (Step 4).
Routes each source file to its appropriate parser tier (Python AST → JS/TS tree-sitter → Fallback)
and writes all output chunks to chunks/chunks.jsonl.
"""
import json
import logging
from pathlib import Path
from typing import List

from src.chunking.models import Chunk
from src.chunking.python_parser import parse_python_file
from src.chunking.js_ts_parser import parse_js_ts_file
from src.chunking.fallback_parser import parse_fallback_file
from src.config import RunConfig

logger = logging.getLogger(__name__)

# Isolation integrity markers — class chunks must NOT contain these
_CLASS_LEAK_MARKERS = ("\ndef ", "\nasync def ")

# Extension → parser tier routing
_PYTHON_EXTS = {".py"}
_JS_TS_EXTS = {".js", ".jsx", ".ts", ".tsx"}

# Categories worth chunking (excludes DOCS, CONFIG, BUILD, ASSET)
_CHUNKABLE_CATEGORIES = {"SOURCE_CODE", "TEST"}


def _route_file(file_path: Path, rel_path: str) -> List[Chunk]:
    """Dispatch a file to the correct parser based on extension."""
    ext = file_path.suffix.lower()

    if ext in _PYTHON_EXTS:
        return parse_python_file(file_path, rel_path)
    elif ext in _JS_TS_EXTS:
        return parse_js_ts_file(file_path, rel_path)
    else:
        return parse_fallback_file(file_path, rel_path)


def _validate_isolation(chunks: List[Chunk], rel_path: str) -> None:
    """
    Structural integrity guard — warns if any class chunk still contains
    method body text, which would mean the Subtractive Isolation failed.

    This check is a permanent regression guard. It costs O(n * k) string
    scans per file (negligible) and ensures the failure mode is never silent.
    """
    for chunk in chunks:
        if chunk.chunk_type == "class":
            for marker in _CLASS_LEAK_MARKERS:
                if marker in chunk.chunk_text:
                    logger.warning(
                        f"[ISOLATION FAIL] Class chunk still contains method "
                        f"body code (marker: {marker!r}): {chunk.chunk_id} "
                        f"in {rel_path}"
                    )
                    break


def chunk_repo(config: RunConfig, project_dir: Path) -> Path:
    """
    Step 4 — Chunk all SOURCE_CODE and TEST files in the classified manifest.

    Reads `manifest/classified_files.json`, routes each relevant file to its
    appropriate parser (Python AST, JS/TS tree-sitter, or regex fallback), and
    writes all chunks to `chunks/chunks.jsonl`.

    The max_chunks limit is NOT enforced here as a hard cutoff — all chunks are
    generated to enable correct centrality scoring in later steps (Step 5).
    The limit will be enforced during LLM context assembly (Step 7+).

    Args:
        config: The active RunConfig.
        project_dir: The root output directory for this repo run.

    Returns:
        Path to the generated chunks.jsonl file.
    """
    raw_repo_dir = project_dir / "raw_repo"
    classified_path = project_dir / "manifest" / "classified_files.json"
    chunks_dir = project_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = chunks_dir / "chunks.jsonl"

    # --- Load manifest ---
    with open(classified_path, "r", encoding="utf-8") as f:
        classified: dict = json.load(f)

    total_chunks = 0
    seen_chunk_ids = set()

    with open(chunks_path, "w", encoding="utf-8") as out_f:
        for rel_path, meta in classified.items():
            category = meta.get("category", "UNKNOWN")
            if category not in _CHUNKABLE_CATEGORIES:
                continue

            file_path = raw_repo_dir / rel_path
            if not file_path.exists():
                logger.warning(f"File not found (skipping): {rel_path}")
                continue

            try:
                chunks = _route_file(file_path, rel_path)
            except Exception as e:
                logger.warning(f"Failed to parse {rel_path}: {e}")
                continue

            _validate_isolation(chunks, rel_path)

            for chunk in chunks:
                # Guard against duplicate chunk IDs (e.g. same-named functions)
                original_id = chunk.chunk_id
                suffix = 1
                while chunk.chunk_id in seen_chunk_ids:
                    chunk.chunk_id = f"{original_id}_{suffix}"
                    suffix += 1
                seen_chunk_ids.add(chunk.chunk_id)

                out_f.write(json.dumps(chunk.to_dict()) + "\n")
                total_chunks += 1

    logger.info(f"Step 4 complete. Total chunks written: {total_chunks} → {chunks_path}")
    return chunks_path
