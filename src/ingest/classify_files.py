import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.config import RunConfig

logger = logging.getLogger(__name__)

CATEGORY_EXTENTIONS = {
    "SOURCE_CODE": {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php"},
    "DOCS": {".md", ".rst", ".txt"},
    "CONFIG": {".toml", ".yaml", ".yml", ".json", ".ini", ".env", ".cfg", ".xml"},
    "ASSET": {".png", ".jpg", ".jpeg", ".svg", ".css", ".ico", ".gif", ".html"}
}

BUILD_FILES = {"dockerfile", "makefile", "cmakelists.txt", "package.json", "pom.xml"}

def is_test_file(rel_path: str) -> bool:
    """Uses path heuristics to determine if a file is a test file."""
    path_lower = rel_path.lower()
    name_lower = Path(rel_path).name.lower()
    
    parts = Path(path_lower).parts
    if "tests" in parts or "test" in parts or "__tests__" in parts or "spec" in parts:
        return True
    
    if name_lower.startswith("test_") or name_lower.endswith("_test.py") or name_lower.endswith(".test.js") or name_lower.endswith(".spec.js"):
        return True
    
    return False

def classify_files(config: RunConfig, manifest_path: Path, output_dir: Path) -> Path:
    """
    Reads the file manifest and assigns each file to a category and checks if it's a test.
    
    Args:
        config: RunConfig object.
        manifest_path: Path to the generated file_manifest.json
        output_dir: The project outputs directory.
    
    Returns:
        Path to classified_files.json
    """
    classified_path = output_dir / "manifest" / "classified_files.json"
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    
    files = manifest_data.get("files", [])
    classified_data: Dict[str, Any] = {}
    
    for rel_path in files:
        path_obj = Path(rel_path)
        ext = path_obj.suffix.lower()
        name = path_obj.name.lower()
        
        category = "UNKNOWN"
        is_test = is_test_file(rel_path)
        
        # Build files
        if name in BUILD_FILES or "makefile" in name:
            category = "BUILD"
        elif any(ext in exts for c, exts in CATEGORY_EXTENTIONS.items() if c == "SOURCE_CODE"):
            category = "SOURCE_CODE"
        elif any(ext in exts for c, exts in CATEGORY_EXTENTIONS.items() if c == "DOCS"):
            category = "DOCS"
        elif any(ext in exts for c, exts in CATEGORY_EXTENTIONS.items() if c == "CONFIG"):
            category = "CONFIG"
        elif any(ext in exts for c, exts in CATEGORY_EXTENTIONS.items() if c == "ASSET"):
            category = "ASSET"
        
        # If it's a test file and no other specific category, it's typically source code
        # However PIPELINE.md suggests TEST as its own category.
        if is_test and category in ["SOURCE_CODE", "UNKNOWN"]:
            category = "TEST"

        # Apply skip_large_assets toggle for documentation clarity
        if config.skip_large_assets and category == "ASSET":
            # We already skipped very large files by size limit, but we can drop assets entirely 
            # from downstream analysis to reduce noise if requested.
            continue
            
        classified_data[rel_path] = {
            "category": category,
            "is_test": is_test
        }
    
    with open(classified_path, 'w', encoding='utf-8') as f:
        json.dump(classified_data, f, indent=2)
    
    logger.info(f"Classified {len(classified_data)} files into {classified_path.name}")
    return classified_path
