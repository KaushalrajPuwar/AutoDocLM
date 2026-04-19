import json
from dataclasses import dataclass
from typing import Optional

# Core configurations
DEFAULT_MAX_FILES = 300
DEFAULT_MAX_CHUNKS = 2000
DEFAULT_MAX_FILE_SIZE_KB = 300
DEFAULT_INCLUDE_TESTS = False
DEFAULT_SKIP_LARGE_ASSETS = True
DEFAULT_USE_EMBEDDINGS = False

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
        }
