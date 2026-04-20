import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def get_cache_key(input_data: str, model: str, prompt_version: str) -> str:
    """
    Generate a SHA-256 cache key based on the exact input string, model name, and prompt version.
    """
    content = input_data + model + prompt_version
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_cache_path(out_dir: str, stage: str, cache_key: str) -> str:
    """
    Get the absolute path to the cache file.
    """
    return os.path.join(out_dir, ".cache", "llm", stage, f"{cache_key}.json")

def read_cache(out_dir: str, stage: str, cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Try to read a cached JSON response. Returns None if it doesn't exist.
    """
    path = get_cache_path(out_dir, stage, cache_key)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache at {path}: {e}")
            return None
    return None

def write_cache(out_dir: str, stage: str, cache_key: str, data: Dict[str, Any]) -> None:
    """
    Write a successful JSON response to cache.
    """
    path = get_cache_path(out_dir, stage, cache_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write cache at {path}: {e}")
