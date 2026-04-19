import json
import logging
import os
from pathlib import Path
from typing import List

import pathspec

from src.config import RunConfig

logger = logging.getLogger(__name__)

DEFAULT_IGNORES = [
    ".git/",
    "node_modules/",
    "venv/",
    ".venv/",
    "dist/",
    "build/",
    ".idea/",
    ".vscode/",
    "__pycache__/",
    ".pytest_cache/",
    "target/",
    "*.so",
    "*.dll",
    "*.exe",
    "*.o",
    "*.a",
    "*.bin",
    "*.pyc"
]

def load_gitignore_spec(raw_repo_dir: Path) -> pathspec.PathSpec:
    """Loads .gitignore from repo root into a PathSpec."""
    patterns = list(DEFAULT_IGNORES)
    gitignore_path = raw_repo_dir / ".gitignore"
    if gitignore_path.exists():
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                patterns.extend(f.readlines())
        except Exception as e:
            logger.warning(f"Failed to read .gitignore: {e}")
    
    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)

def filter_files(config: RunConfig, raw_repo_dir: Path, output_dir: Path) -> Path:
    """
    Crawls raw_repo, applies ignore rules, checks sizes, and outputs a manifest.
    
    Args:
        config: RunConfig object.
        raw_repo_dir: Path to the cloned repository.
        output_dir: The project outputs directory.
    
    Returns:
        Path to the generated file_manifest.json
    """
    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "file_manifest.json"

    spec = load_gitignore_spec(raw_repo_dir)
    valid_files: List[str] = []
    
    max_size_bytes = config.max_file_size_kb * 1024

    for root, dirs, files in os.walk(raw_repo_dir):
        # We need relative paths for pathspec matching to work accurately
        # and to store clean relative paths in the manifest.
        for item in dirs + files:
            full_path = Path(root) / item
            try:
                rel_path_str = str(full_path.relative_to(raw_repo_dir))
            except ValueError:
                continue

            if full_path.is_dir() and spec.match_file(rel_path_str + "/"):
                 # Modify dirs in-place to prevent os.walk from descending
                 if item in dirs:
                     dirs.remove(item)

        for file_name in files:
            full_path = Path(root) / file_name
            try:
                 rel_path_str = str(full_path.relative_to(raw_repo_dir))
            except ValueError:
                 continue

            if spec.match_file(rel_path_str):
                continue
            
            # Check file size
            try:
                size_bytes = full_path.stat().st_size
                if size_bytes > max_size_bytes:
                    logger.debug(f"Skipping {rel_path_str}: size {size_bytes / 1024:.1f}KB exceeds limit.")
                    continue
            except OSError:
                continue
            
            valid_files.append(rel_path_str)

    # Note: We are deferring the hard config.max_files cutoff to Step 4 (chunking)
    # to allow centrality scoring based on the full repo context first.
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({"files": valid_files}, f, indent=2)
    
    logger.info(f"Filtered repository to {len(valid_files)} valid files.")
    return manifest_path
