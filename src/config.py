import json
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Core configurations
DEFAULT_MAX_FILES = 300
DEFAULT_MAX_CHUNKS = 2000
DEFAULT_MAX_FILE_SIZE_KB = 300
DEFAULT_INCLUDE_TESTS = False
DEFAULT_SKIP_LARGE_ASSETS = True
DEFAULT_USE_EMBEDDINGS = True

# Step 6: Embedding configuration
# Model is fixed for this project (qwen3-embedding:0.6b, 1024-dim, 32K context).
# If ever swapped, the ChromaDB collection must be deleted and rebuilt.
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_EMBEDDING_BATCH_SIZE = 16

# Step 7: LLM Inference configuration
DEFAULT_INFERENCE_API_KEY = os.environ.get("API_KEY")
DEFAULT_INFERENCE_BASE_URL = os.environ.get("BASE_URL", "https://inference.api.nscale.com/v1")
DEFAULT_INFERENCE_CONCURRENCY = 10
DEFAULT_CHUNK_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ARCH_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

@dataclass
class RunConfig:
    """Configuration for a single pipeline run."""
    repo_url: Optional[str] = None
    repo_zip: Optional[str] = None
    max_files: int = DEFAULT_MAX_FILES
    max_chunks: int = DEFAULT_MAX_CHUNKS
    max_file_size_kb: int = DEFAULT_MAX_FILE_SIZE_KB
    use_embeddings: bool = DEFAULT_USE_EMBEDDINGS
    include_tests: bool = DEFAULT_INCLUDE_TESTS
    skip_large_assets: bool = DEFAULT_SKIP_LARGE_ASSETS
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    inference_api_key: Optional[str] = DEFAULT_INFERENCE_API_KEY
    inference_base_url: str = DEFAULT_INFERENCE_BASE_URL
    inference_concurrency: int = DEFAULT_INFERENCE_CONCURRENCY
    chunk_model: str = DEFAULT_CHUNK_MODEL
    arch_model: str = DEFAULT_ARCH_MODEL
    force_clone: bool = False

    def model_dump(self):
        """Return a dictionary representation of the config."""
        return {
            "repo_url": self.repo_url,
            "repo_zip": self.repo_zip,
            "max_files": self.max_files,
            "max_chunks": self.max_chunks,
            "max_file_size_kb": self.max_file_size_kb,
            "use_embeddings": self.use_embeddings,
            "include_tests": self.include_tests,
            "skip_large_assets": self.skip_large_assets,
            "force_clone": self.force_clone,
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "inference_api_key": "***" if self.inference_api_key else None,
            "inference_base_url": self.inference_base_url,
            "inference_concurrency": self.inference_concurrency,
            "chunk_model": self.chunk_model,
            "arch_model": self.arch_model,
        }
