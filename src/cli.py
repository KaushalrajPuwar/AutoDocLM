import argparse
import logging
import sys
from typing import Optional

from src.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INCLUDE_TESTS,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MAX_FILE_SIZE_KB,
    DEFAULT_MAX_FILES,
    DEFAULT_SKIP_LARGE_ASSETS,
    DEFAULT_USE_EMBEDDINGS,
    DEFAULT_INFERENCE_API_KEY,
    DEFAULT_INFERENCE_BASE_URL,
    DEFAULT_INFERENCE_CONCURRENCY,
    DEFAULT_SERVE_HOST,
    DEFAULT_SERVE_PORT,
    RunConfig,
)
from src.pipeline import Orchestrator

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AutoDocLM: Automatic Repository Documentation Generator")
    parser.add_argument("-r", "--repo-url", type=str, default=None, help="Public GitHub repository URL to document.")
    parser.add_argument("-z", "--repo-zip", type=str, default=None, help="Local zip file of the repository to document.")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Maximum number of files to process.")
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS, help="Maximum number of chunks to extract.")
    parser.add_argument("--max-file-size-kb", type=int, default=DEFAULT_MAX_FILE_SIZE_KB, help="Maximum file size in KB to parse.")
    parser.add_argument("--use-embeddings", action="store_true", default=DEFAULT_USE_EMBEDDINGS, help="Enable generating ChromaDB chunk embeddings for retrieval (requires Ollama running locally).")
    parser.add_argument("--include-tests", action="store_true", default=DEFAULT_INCLUDE_TESTS, help="Include test files in graph analysis.")
    parser.add_argument("--skip-large-assets", action="store_true", default=DEFAULT_SKIP_LARGE_ASSETS, help="Skip large binary assets during analysis.")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help=f"Ollama embedding model name (default: {DEFAULT_EMBEDDING_MODEL}).")
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE, help=f"Number of chunks per Ollama embedding call (default: {DEFAULT_EMBEDDING_BATCH_SIZE}).")
    parser.add_argument("--inference-api-key", type=str, default=DEFAULT_INFERENCE_API_KEY, help="API key for inference provider (defaults to API_KEY env var).")
    parser.add_argument("--inference-base-url", type=str, default=DEFAULT_INFERENCE_BASE_URL, help=f"Base URL for inference provider (default: {DEFAULT_INFERENCE_BASE_URL}).")
    parser.add_argument("--inference-concurrency", type=int, default=DEFAULT_INFERENCE_CONCURRENCY, help=f"Max concurrent LLM requests (default: {DEFAULT_INFERENCE_CONCURRENCY}).")
    parser.add_argument("--no-serve-site", action="store_true", default=False, help="Do not host docs locally after Step 10 build.")
    parser.add_argument("--serve-host", type=str, default=DEFAULT_SERVE_HOST, help=f"Host for local docs server (default: {DEFAULT_SERVE_HOST}).")
    parser.add_argument("--serve-port", type=int, default=DEFAULT_SERVE_PORT, help=f"Port for local docs server (default: {DEFAULT_SERVE_PORT}).")
    parser.add_argument("--force-clone", action="store_true", default=False, help="Force deletion and re-clone of existing raw_repo directory.")

    args = parser.parse_args()

    if not args.repo_url and not args.repo_zip:
        parser.error("You must provide either --repo-url or --repo-zip.")
    
    if args.repo_url and args.repo_zip:
        parser.error("Provide only one of --repo-url or --repo-zip. Not both.")

    config = RunConfig(
        repo_url=args.repo_url,
        repo_zip=args.repo_zip,
        max_files=args.max_files,
        max_chunks=args.max_chunks,
        max_file_size_kb=args.max_file_size_kb,
        use_embeddings=args.use_embeddings,
        include_tests=args.include_tests,
        skip_large_assets=args.skip_large_assets,
        force_clone=args.force_clone,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        inference_api_key=args.inference_api_key,
        inference_base_url=args.inference_base_url,
        inference_concurrency=args.inference_concurrency,
        serve_site=not args.no_serve_site,
        serve_host=args.serve_host,
        serve_port=args.serve_port,
    )

    try:
        orchestrator = Orchestrator(config)
        orchestrator.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
