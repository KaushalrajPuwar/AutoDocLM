import json
import logging
import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    # Python < 3.11
    import tomli as tomllib

logger = logging.getLogger(__name__)

def _extract_from_requirements_txt(file_path: Path) -> list[str]:
    """Extracts dependencies from a requirements.txt file."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            valid_lines = []
            for line in f:
                # Strip inline and full-line comments
                clean_line = line.split('#')[0].strip()
                if clean_line:
                    valid_lines.append(clean_line)
            return valid_lines
    except Exception as e:
        logger.error(f"Error reading requirements.txt {file_path}: {e}")
        return []

def _extract_from_pyproject_toml(file_path: Path) -> list[str]:
    """Extracts dependencies from a pyproject.toml file."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
        
        dependencies = []
        
        # Standard dependencies
        if "project" in data and "dependencies" in data["project"]:
            dependencies.extend(data["project"]["dependencies"])
            
        # Optional dependencies
        if "project" in data and "optional-dependencies" in data["project"]:
            for group in data["project"]["optional-dependencies"].values():
                dependencies.extend(group)

        # Poetry dependencies
        if "tool" in data and "poetry" in data["tool"] and "dependencies" in data["tool"]["poetry"]:
            # Poetry lists python version as a dependency, filter it out
            deps = data["tool"]["poetry"]["dependencies"]
            dependencies.extend([f"{k}{v}" for k, v in deps.items() if k.lower() != "python"])

        return list(set(dependencies))
    except Exception as e:
        logger.error(f"Error reading pyproject.toml {file_path}: {e}")
        return []

def _extract_from_package_json(file_path: Path) -> list[str]:
    """Extracts dependencies from a package.json file."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dependencies = []
        if "dependencies" in data:
            dependencies.extend([f"{k}@{v}" for k, v in data["dependencies"].items()])
        if "devDependencies" in data:
            dependencies.extend([f"{k}@{v}" for k, v in data["devDependencies"].items()])
        
        return list(set(dependencies))
    except Exception as e:
        logger.error(f"Error reading package.json {file_path}: {e}")
        return []

def extract_dependencies(repo_path: Path, output_path: Path):
    """
    Extracts dependencies from various dependency files in the repository
    and saves them to a JSON file.
    """
    logger.info("Starting dependency extraction...")
    
    all_dependencies = {
        "python": [],
        "javascript": [],
        "other": []
    }

    dependency_files = {
        "requirements.txt": ("python", _extract_from_requirements_txt),
        "pyproject.toml": ("python", _extract_from_pyproject_toml),
        "package.json": ("javascript", _extract_from_package_json),
    }

    for file_name, (lang, extractor) in dependency_files.items():
        found_file = next(repo_path.rglob(file_name), None)
        if found_file:
            logger.info(f"Found {file_name}, extracting dependencies...")
            deps = extractor(found_file)
            all_dependencies[lang].extend(deps)
            logger.info(f"Found {len(deps)} dependencies in {file_name}.")

    # Remove duplicates
    for lang in all_dependencies:
        all_dependencies[lang] = sorted(list(set(all_dependencies[lang])))

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_dependencies, f, indent=2)
        
    logger.info(f"Dependency extraction complete. Results saved to {output_path}")

if __name__ == '__main__':
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    # Create dummy files for testing
    dummy_repo = Path("./dummy_repo")
    dummy_repo.mkdir(exist_ok=True)
    
    (dummy_repo / "requirements.txt").write_text("flask>=2.0\nrequests\n# a comment")
    (dummy_repo / "pyproject.toml").write_text("""
[project]
name = "my-project"
version = "0.1.0"
dependencies = [
    "numpy",
    "pandas",
]
[project.optional-dependencies]
dev = [
    "pytest",
]
""")
    (dummy_repo / "package.json").write_text("""
{
    "name": "my-app",
    "version": "1.0.0",
    "dependencies": {
        "react": "^17.0.0",
        "react-dom": "^17.0.0"
    },
    "devDependencies": {
        "eslint": "^7.0.0"
    }
}
""")
    
    output_dir = Path("./outputs/dummy_repo/analysis")
    extract_dependencies(dummy_repo, output_dir / "dependencies.json")
    
    # Clean up dummy files
    import shutil
    shutil.rmtree(dummy_repo)
    # shutil.rmtree(Path("./outputs"))
