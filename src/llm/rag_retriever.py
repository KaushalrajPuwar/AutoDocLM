"""
RAG Retriever — thin adapter between the LLM layer and the ChromaDB Indexing layer.

Handles:
- Lazy loading of the persistent ChromaDB collection.
- Graceful degradation when embeddings are unavailable (Ollama offline / Step 6 skipped).
- Formatting of SearchResult objects into an evidence text block suitable for prompt injection.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def load_collection(out_dir: str) -> Optional[object]:
    """
    Open the persistent ChromaDB collection for the given repo output directory.

    Returns the collection object on success, or None if the DB path doesn't exist
    (i.e. Step 6 was skipped or Ollama was offline during indexing).
    """
    db_path = os.path.join(out_dir, "embeddings", "chroma.db")
    if not os.path.exists(db_path):
        logger.warning(
            f"ChromaDB not found at {db_path}. "
            "Step 7.3 will run without implementation evidence (graph-only mode)."
        )
        return None

    try:
        import chromadb
        from src.indexing.vector_store_chroma import get_or_create_collection
        from pathlib import Path
        collection = get_or_create_collection(Path(db_path))
        logger.info(f"ChromaDB collection loaded for RAG: {collection.count()} chunks available.")
        return collection
    except Exception as e:
        logger.warning(f"Failed to open ChromaDB collection: {e}. Running without RAG evidence.")
        return None


def retrieve_evidence(
    collection: Optional[object],
    query: str,
    file_filter: Optional[list] = None,
    exclude_filter: Optional[list] = None,
    n: int = 5,
    ollama_model: str = "qwen3-embedding:0.6b",
) -> str:
    """
    Retrieve the top-N semantically similar chunks and format them as a
    human-readable evidence block for prompt injection.

    Args:
        collection:     Open ChromaDB collection, or None (graceful degradation).
        query:          The programmatic search query.
        file_filter:    List of file paths to restrict the search to (inclusive).
        exclude_filter: List of file paths to explicitly exclude from the search.
        n:              Number of chunks to retrieve.
        ollama_model:   Embedding model used during indexing.
    """
    if collection is None:
        return "No implementation evidence available (Step 6 embeddings not built)."

    try:
        from src.indexing.vector_store_chroma import search_similar_chunks
        results = search_similar_chunks(
            collection=collection,
            query=query,
            ollama_model=ollama_model,
            n_results=n,
            file_filter=file_filter,
            exclude_filter=exclude_filter,
        )
    except Exception as e:
        logger.warning(f"RAG search failed: {e}. Falling back to no evidence.")
        return "No implementation evidence available (search error)."

    if not results:
        return "No implementation evidence available (no relevant chunks found)."

    # Format each result as a labelled block so the LLM can distinguish sources
    blocks = []
    for r in results:
        header = f"--- CHUNK: {r.file} :: {r.symbol} ({r.chunk_type}) [distance: {r.distance:.4f}] ---"
        blocks.append(f"{header}\n{r.text}")

    return "\n\n".join(blocks)
