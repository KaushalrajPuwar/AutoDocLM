"""
ChromaDB Vector Store Manager

Manages the persistent ChromaDB collection that holds all chunk embeddings.
Provides:
  - Collection creation / retrieval
  - Idempotent upsert of EmbeddedChunk objects
  - Metadata JSON file output (lightweight fast-lookup index)
  - Similarity search with optional per-folder file filtering

Collection design
-----------------
Collection name : "chunks"
Distance metric : cosine  (appropriate for text embeddings)
Embedding dim   : 1024   (qwen3-embedding:0.6b; fixed for this project)

The run_indexing() function is the single entry point called by pipeline.py.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb

from src.indexing.embed_chunks import EmbeddedChunk, embed_chunks

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "chunks"


# ---------------------------------------------------------------------------
# Data model for query results
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single chunk returned by a similarity search."""
    chunk_id: str
    file: str
    symbol: str
    chunk_type: str
    language: str
    line_start: int
    line_end: int
    text: str         # The enriched code text (including decorators/context)
    distance: float   # cosine distance; lower = more similar


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_or_create_collection(db_path: Path) -> chromadb.Collection:
    """
    Open (or create) the persistent ChromaDB collection at db_path.

    Args:
        db_path: Directory path for ChromaDB's PersistentClient.
                 Typically: outputs/<repo>/embeddings/chroma.db/

    Returns:
        A chromadb.Collection object ready for upsert / query.
    """
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"ChromaDB collection '{_COLLECTION_NAME}' at {db_path} "
        f"contains {collection.count()} existing chunks."
    )
    return collection


def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    """
    Return the set of chunk_ids already stored in the collection.
    Used by embed_chunks() to skip already-indexed chunks.
    """
    # ChromaDB .get() with no filter returns all IDs.
    # For very large collections this could be slow, but for our hard limit
    # of 2000 chunks it is always fast.
    result = collection.get(include=[])   # include=[] → return only ids
    return set(result["ids"])


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    collection: chromadb.Collection,
    embedded_chunks: list[EmbeddedChunk],
) -> int:
    """
    Insert (or update) EmbeddedChunk records into ChromaDB.

    Chunks with embedding=None are skipped.
    Uses upsert (not add) so re-runs with overlapping IDs never crash.

    Args:
        collection:      Target ChromaDB collection.
        embedded_chunks: List produced by embed_chunks().

    Returns:
        Number of records successfully upserted.
    """
    valid = [ec for ec in embedded_chunks if ec.embedding is not None]
    skipped = len(embedded_chunks) - len(valid)

    if skipped:
        logger.warning(
            f"{skipped} chunks had no embedding and were not inserted."
        )

    if not valid:
        logger.info("No new chunks to upsert into ChromaDB.")
        return 0

    ids = [ec.chunk_id for ec in valid]
    embeddings = [ec.embedding for ec in valid]
    documents = [ec.embed_text for ec in valid]
    metadatas = [
        {
            "file": ec.file,
            "symbol": ec.symbol,
            "chunk_type": ec.chunk_type,
            "language": ec.language,
            "line_start": ec.line_start,
            "line_end": ec.line_end,
        }
        for ec in valid
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Upserted {len(valid)} chunks into ChromaDB collection.")
    return len(valid)


# ---------------------------------------------------------------------------
# Metadata file
# ---------------------------------------------------------------------------

def write_chunk_metadata(
    collection: chromadb.Collection,
    output_path: Path,
) -> None:
    """
    Write a lightweight chunk_metadata.json from all records in the collection.

    This file allows Step 7.3 (and any other step) to enumerate what is
    indexed without opening ChromaDB. It is always regenerated from the
    full collection so it stays in sync even after partial re-runs.

    Output schema:
    {
      "<chunk_id>": {
        "file": "...",
        "symbol": "...",
        "chunk_type": "...",
        "language": "...",
        "line_start": <int>,
        "line_end": <int>
      },
      ...
    }
    """
    result = collection.get(include=["metadatas"])
    ids = result["ids"]
    metadatas = result["metadatas"]

    metadata_map: dict[str, dict] = {}
    for chunk_id, meta in zip(ids, metadatas):
        metadata_map[chunk_id] = meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=2)

    logger.info(
        f"chunk_metadata.json written: {len(metadata_map)} entries → {output_path}"
    )


# ---------------------------------------------------------------------------
# Query / retrieval (used by Step 7.3)
# ---------------------------------------------------------------------------

def search_similar_chunks(
    collection: chromadb.Collection,
    query: str,
    ollama_model: str = "qwen3-embedding:0.6b",
    n_results: int = 5,
    file_filter: Optional[list[str]] = None,
) -> list[SearchResult]:
    """
    Retrieve the most semantically similar chunks to a query string.

    Args:
        collection:   The ChromaDB collection to search.
        query:        A natural language or code string to embed and search.
        ollama_model: Ollama model used to embed the query (must match the
                      model used during indexing).
        n_results:    Maximum number of results to return.
        file_filter:  If provided, restrict results to chunks from these
                      relative file paths. Used by Step 7.3 to search
                      within a specific folder's files only.

    Returns:
        List of SearchResult objects sorted by ascending cosine distance
        (most similar first). Empty list if query embedding fails.
    """
    try:
        import ollama as _ollama
        response = _ollama.embeddings(model=ollama_model, prompt=query)
        query_embedding = response["embedding"]
    except Exception as e:
        logger.warning(f"Failed to embed query for similarity search: {e}")
        return []

    # Build ChromaDB where-filter for file restriction.
    where: Optional[dict] = None
    if file_filter:
        if len(file_filter) == 1:
            where = {"file": {"$eq": file_filter[0]}}
        else:
            where = {"file": {"$in": file_filter}}

    # Clamp n_results to what is actually in the collection (after filtering)
    # to avoid ChromaDB errors when the collection is small.
    available = collection.count()
    clamped_n = min(n_results, max(available, 1))

    try:
        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=clamped_n,
            include=["metadatas", "distances", "documents"],
        )
        if where:
            query_kwargs["where"] = where

        result = collection.query(**query_kwargs)
    except Exception as e:
        logger.warning(f"ChromaDB query failed: {e}")
        return []

    search_results: list[SearchResult] = []
    ids_list = result["ids"][0]
    distances_list = result["distances"][0]
    metadatas_list = result["metadatas"][0]
    documents_list = result["documents"][0]

    for chunk_id, distance, meta, text in zip(ids_list, distances_list, metadatas_list, documents_list):
        search_results.append(
            SearchResult(
                chunk_id=chunk_id,
                file=meta.get("file", ""),
                symbol=meta.get("symbol", ""),
                chunk_type=meta.get("chunk_type", ""),
                language=meta.get("language", ""),
                line_start=meta.get("line_start", 0),
                line_end=meta.get("line_end", 0),
                text=text,
                distance=distance,
            )
        )

    return search_results


# ---------------------------------------------------------------------------
# Main entry point (called by pipeline.py)
# ---------------------------------------------------------------------------

def run_indexing(
    chunks_path: Path,
    classified_files_path: Path,
    embeddings_dir: Path,
    ollama_model: str = "qwen3-embedding:0.6b",
    batch_size: int = 16,
    include_tests: bool = False,
) -> Optional[Path]:
    """
    Full Step 6 orchestration: embed all new chunks and persist to ChromaDB.

    Flow:
      1. Open (or create) the ChromaDB persistent collection.
      2. Fetch existing chunk IDs for idempotency.
      3. Call embed_chunks() to generate embeddings for new chunks only.
      4. Upsert new embeddings into ChromaDB.
      5. Regenerate chunk_metadata.json from the full collection.

    Args:
        chunks_path:            Path to chunks/chunks.jsonl.
        classified_files_path:  Path to manifest/classified_files.json.
        embeddings_dir:         Directory for chroma.db/ and chunk_metadata.json.
                                Typically: outputs/<repo>/embeddings/
        ollama_model:           Ollama embedding model to use.
        batch_size:             Chunks per Ollama call.
        include_tests:          If True, test-file chunks are embedded too.

    Returns:
        Path to chunk_metadata.json on success, or None if the step was
        skipped due to an Ollama connectivity failure.
    """
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    db_path = embeddings_dir / "chroma.db"
    metadata_path = embeddings_dir / "chunk_metadata.json"

    logger.info(f"Step 6: Starting indexing. DB path: {db_path}")

    # 1. Open collection
    collection = get_or_create_collection(db_path)

    # 2. Get existing IDs for deduplication
    existing_ids = get_existing_ids(collection)
    logger.info(f"Found {len(existing_ids)} chunks already in the collection.")

    # 3. Generate embeddings for new chunks
    try:
        embedded = embed_chunks(
            chunks_path=chunks_path,
            classified_files_path=classified_files_path,
            existing_ids=existing_ids,
            ollama_model=ollama_model,
            batch_size=batch_size,
            include_tests=include_tests,
        )
    except ConnectionError as e:
        logger.error(
            f"Step 6 skipped: {e}\n"
            "Pipeline will continue without embeddings. "
            "Step 7.3 will rely on graph evidence only."
        )
        return None
    except FileNotFoundError as e:
        logger.error(f"Step 6 failed: {e}")
        return None

    # 4. Upsert into ChromaDB
    upserted = upsert_chunks(collection, embedded)

    # 5. Regenerate full metadata file from the collection
    write_chunk_metadata(collection, metadata_path)

    logger.info(
        f"Step 6 complete. "
        f"Total in collection: {collection.count()}. "
        f"New this run: {upserted}."
    )
    return metadata_path
