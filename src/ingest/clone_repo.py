import json
import logging
import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

from src.config import RunConfig

logger = logging.getLogger(__name__)

def ingest_repo(config: RunConfig, metadata: dict, output_dir: Path) -> Path:
    """
    Ingests the repository either by cloning from a URL or extracting from a zip file.
    
    Args:
        config: The parsed run configuration.
        metadata: The dictionary object to update with run metadata (e.g. commit hash)
        output_dir: The directory where the repo should be cloned/extracted
        
    Returns:
        The path to the raw_repo directory.
    """
    raw_repo_dir = output_dir / "raw_repo"

    # Handle existing directory
    if raw_repo_dir.exists():
        if config.force_clone:
            logger.info(f"Removing existing raw_repo directory due to --force-clone: {raw_repo_dir}")
            shutil.rmtree(raw_repo_dir)
        else:
            logger.info(f"Found existing raw_repo, skipping clone/extract as --force-clone is not set.")
            # Still try to grab commit hash if it's a git repo
            if (raw_repo_dir / ".git").exists():
                 try:
                     commit_hash = subprocess.check_output(
                         ["git", "rev-parse", "HEAD"], cwd=raw_repo_dir, text=True
                     ).strip()
                     metadata["commit_hash"] = commit_hash
                 except subprocess.CalledProcessError:
                     logger.warning("Failed to determine commit hash for existing repository.")
            return raw_repo_dir

    raw_repo_dir.mkdir(parents=True, exist_ok=True)

    if config.repo_url:
        logger.info(f"Cloning repository: {config.repo_url} into {raw_repo_dir}")
        try:
            # We clone directly into the directory if it's empty
            # If not empty, this might fail, hence the rmtree above for force_clone
            subprocess.run(["git", "clone", config.repo_url, str(raw_repo_dir)], check=True, capture_output=True, text=True)
            
            # Extract commit hash
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=raw_repo_dir, text=True
            ).strip()
            metadata["commit_hash"] = commit_hash
            logger.info(f"Successfully cloned repository. Commit hash: {commit_hash}")

        except subprocess.CalledProcessError as e:
             logger.error(f"Git clone failed: {e.stderr}")
             raise RuntimeError(f"Git clone failed: {e.stderr}")

    elif config.repo_zip:
        logger.info(f"Extracting zip repository: {config.repo_zip} into {raw_repo_dir}")
        zip_path = Path(config.repo_zip)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        try:
             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_repo_dir)
             
             metadata["commit_hash"] = "local_zip"
             logger.info(f"Successfully extracted the zip file.")
        except Exception as e:
             logger.error(f"ZIP extraction failed: {e}")
             raise RuntimeError(f"ZIP extraction failed: {e}")
             
    else:
        raise ValueError("Neither repo_url nor repo_zip was provided.")

    return raw_repo_dir
