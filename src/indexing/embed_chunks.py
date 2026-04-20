"""
Chunk Embedding Generator

Reads chunks.jsonl, optionally filters out test-file chunks using
classified_files.json, and calls the Ollama embedding model in batches
to produce embedding vectors for every remaining chunk.

Future improvement note (tracked)
----------------------------------
The `is_test` flag is currently determined by cross-referencing
classified_files.json because chunks.jsonl does not carry it directly.
A future cleanup should add `is_test: bool` to the Chunk dataclass and
persist it in chunks.jsonl at Step 4, removing the need for the
two-file lookup here.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    """A chunk record paired with its embedding vector from Ollama."""
    chunk_id: str
    file: str
    symbol: str
    chunk_type: str
    language: str
    line_start: int
    line_end: int
    embed_text: str           # the enriched text that was embedded
    embedding: Optional[list[float]]   # None if embedding failed for this chunk


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_embed_text(chunk: dict) -> str:
    """
    Construct the string sent to the embedding model.

    Enriches raw chunk_text with file path, symbol name, and chunk type
    so vectors for same-named functions in different files are
    discriminating enough for useful similarity search.
    """
    file_line = f"File: {chunk.get('file', '')}"
    symbol_line = f"Symbol: {chunk.get('symbol', '')}"
    type_line = f"Type: {chunk.get('chunk_type', '')}"
    code = chunk.get("chunk_text", "")
    return f"{file_line}\n{symbol_line}\n{type_line}\n{code}"


def _load_test_file_set(classified_files_path: Path) -> set[str]:
    """
    Return a set of relative file paths that are tagged is_test=True.
    Returns an empty set if the classified_files.json cannot be read.

    NOTE: This two-file lookup (chunks.jsonl + classified_files.json) is a
    known friction point. A future Step 4 cleanup should store `is_test`
    directly in chunks.jsonl to remove the need for this cross-reference.
    """
    if not classified_files_path.exists():
        logger.warning(
            f"classified_files.json not found at {classified_files_path}. "
            "Cannot filter test chunks — all chunks will be embedded."
        )
        return set()
    try:
        with open(classified_files_path, "r", encoding="utf-8") as f:
            classified: dict = json.load(f)
        return {
            rel_path
            for rel_path, meta in classified.items()
            if meta.get("is_test", False)
        }
    except Exception as e:
        logger.warning(f"Failed to parse classified_files.json: {e}. Skipping test filter.")
        return set()


def _load_chunks(chunks_path: Path, test_files: set[str], include_tests: bool) -> list[dict]:
    """
    Stream chunks.jsonl and return those eligible for embedding.
    Chunks from test files are excluded unless include_tests=True.
    """
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSONL line: {e}")
                continue

            if not include_tests and chunk.get("file", "") in test_files:
                continue
            chunks.append(chunk)

    logger.info(f"Loaded {len(chunks)} chunks eligible for embedding.")
    return chunks


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks_path: Path,
    classified_files_path: Path,
    existing_ids: set[str],
    ollama_model: str = "qwen3-embedding:0.6b",
    batch_size: int = 16,
    include_tests: bool = False,
) -> list[EmbeddedChunk]:
    """
    Generate embeddings for all chunks not already in the vector store.

    Args:
        chunks_path:            Path to chunks/chunks.jsonl.
        classified_files_path:  Path to manifest/classified_files.json
                                (used to identify test files).
        existing_ids:           Set of chunk_ids already in ChromaDB.
                                Chunks in this set are skipped.
        ollama_model:           Ollama embedding model name.
        batch_size:             Number of chunks to embed per Ollama API call.
        include_tests:          If True, test-file chunks are also embedded.

    Returns:
        List of EmbeddedChunk objects. Chunks where embedding failed have
        embedding=None and are logged as warnings.

    Raises:
        FileNotFoundError: if chunks_path does not exist.
        ConnectionError:   if Ollama is not reachable (re-raised for caller).
    """
    # Lazy import so the module can be imported even on systems without ollama
    # (the ImportError surfaces only when the function is actually called).
    try:
        import ollama as _ollama
    except ImportError:
        raise ImportError(
            "The 'ollama' package is not installed. Run: uv add ollama"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {chunks_path}")

    # --- Load test file set for filtering ---
    test_files = _load_test_file_set(classified_files_path)

    # --- Load and filter chunks ---
    all_chunks = _load_chunks(chunks_path, test_files, include_tests)

    # --- Skip already-indexed chunks ---
    new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]
    skipped = len(all_chunks) - len(new_chunks)
    logger.info(
        f"{len(new_chunks)} new chunks to embed. "
        f"{skipped} already indexed and skipped."
    )

    if not new_chunks:
        return []

    # --- Verify Ollama connectivity + model availability ---
    try:
        _ollama.embeddings(model=ollama_model, prompt="ping")
    except Exception as e:
        # Re-raise as ConnectionError so the orchestrator can catch it cleanly.
        raise ConnectionError(
            f"Cannot reach Ollama or model '{ollama_model}' is not pulled. "
            f"Run: ollama pull {ollama_model}\nUnderlying error: {e}"
        )

    # --- Embed in batches ---
    results: list[EmbeddedChunk] = []
    total = len(new_chunks)

    for batch_start in range(0, total, batch_size):
        batch = new_chunks[batch_start : batch_start + batch_size]
        embed_texts = [_build_embed_text(c) for c in batch]

        logger.info(
            f"Embedding batch {batch_start // batch_size + 1} "
            f"({batch_start + 1}–{min(batch_start + batch_size, total)} / {total}) …"
        )

        # Call Ollama once per batch using the multi-input embed endpoint.
        # Falls back to individual calls if the server doesn't support batching.
        try:
            response = _ollama.embed(model=ollama_model, input=embed_texts)
            embeddings_batch = response.embeddings  # list of list[float]
        except Exception as batch_err:
            logger.warning(
                f"Batch embed failed ({batch_err}). Falling back to individual calls."
            )
            embeddings_batch = []
            for text in embed_texts:
                try:
                    r = _ollama.embeddings(model=ollama_model, prompt=text)
                    embeddings_batch.append(r["embedding"])
                except Exception as single_err:
                    logger.warning(f"Single embed failed: {single_err}")
                    embeddings_batch.append(None)

        for chunk, text, vector in zip(batch, embed_texts, embeddings_batch):
            if vector is None:
                logger.warning(f"No embedding produced for chunk: {chunk['chunk_id']}")
            results.append(
                EmbeddedChunk(
                    chunk_id=chunk["chunk_id"],
                    file=chunk.get("file", ""),
                    symbol=chunk.get("symbol", ""),
                    chunk_type=chunk.get("chunk_type", ""),
                    language=chunk.get("language", ""),
                    line_start=chunk.get("line_start", 0),
                    line_end=chunk.get("line_end", 0),
                    embed_text=text,
                    embedding=vector,
                )
            )

    success_count = sum(1 for r in results if r.embedding is not None)
    fail_count = len(results) - success_count
    logger.info(
        f"Embedding complete. {success_count} succeeded, {fail_count} failed."
    )
    return results
