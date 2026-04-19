import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

def _find_python_main(file_path: Path) -> bool:
    """Checks if a Python file contains a __main__ block."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return 'if __name__ == "__main__":' in content or "if __name__ == '__main__':" in content
    except Exception:
        return False

def _find_package_json_scripts(file_path: Path) -> list[str]:
    """Extracts run scripts from package.json."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("scripts", {}).keys())
    except Exception:
        return []

def _find_dockerfile_entrypoint(file_path: Path) -> list[str]:
    """Extracts ENTRYPOINT or CMD from a Dockerfile."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        entrypoint_match = re.search(r"^\s*ENTRYPOINT\s+(.*)", content, re.MULTILINE | re.IGNORECASE)
        cmd_match = re.search(r"^\s*CMD\s+(.*)", content, re.MULTILINE | re.IGNORECASE)
        
        entrypoints = []
        if entrypoint_match:
            entrypoints.append(entrypoint_match.group(1).strip())
        if cmd_match:
            entrypoints.append(cmd_match.group(1).strip())
            
        return entrypoints
    except Exception:
        return []

def _find_makefile_targets(file_path: Path) -> list[str]:
    """Extracts common run targets from a Makefile."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # A simple regex to find targets like 'run:', 'start:', etc.
        return re.findall(r"^(run|start|dev|build|install):", content, re.MULTILINE)
    except Exception:
        return []


def detect_entrypoints(repo_path: Path, classified_files_path: Path, output_path: Path):
    """
    Detects potential entrypoints in the repository and saves them to a JSON file.
    """
    logger.info("Starting entrypoint detection...")
    
    try:
        with open(classified_files_path, "r", encoding="utf-8") as f:
            classified_files = json.load(f)
    except FileNotFoundError:
        logger.error(f"Classified files manifest not found at {classified_files_path}")
        return

    entrypoints = []

    for file_info in classified_files:
        file_path = repo_path / file_info["file"]
        file_type = file_info["type"]
        
        if file_type == "SOURCE_CODE" and file_info["language"].lower() == "python":
            if _find_python_main(file_path):
                entrypoints.append({
                    "file": file_info["file"],
                    "type": "python_main",
                    "confidence": "high"
                })
        elif file_path.name == "package.json":
            scripts = _find_package_json_scripts(file_path)
            if scripts:
                entrypoints.append({
                    "file": file_info["file"],
                    "type": "npm_scripts",
                    "scripts": scripts,
                    "confidence": "high"
                })
        elif file_path.name.lower() == "dockerfile":
            docker_cmds = _find_dockerfile_entrypoint(file_path)
            if docker_cmds:
                entrypoints.append({
                    "file": file_info["file"],
                    "type": "docker",
                    "commands": docker_cmds,
                    "confidence": "high"
                })
        elif file_path.name.lower() == "makefile":
            targets = _find_makefile_targets(file_path)
            if targets:
                entrypoints.append({
                    "file": file_info["file"],
                    "type": "makefile",
                    "targets": targets,
                    "confidence": "medium"
                })

    output_data = {"entrypoints": entrypoints}
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    logger.info(f"Entrypoint detection complete. Found {len(entrypoints)} potential entrypoints. Results saved to {output_path}")

if __name__ == '__main__':
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    dummy_repo = Path("./dummy_repo")
    dummy_repo.mkdir(exist_ok=True)
    
    # Create dummy files
    (dummy_repo / "app.py").write_text('if __name__ == "__main__":\n    print("hello")')
    (dummy_repo / "package.json").write_text('{"scripts": {"start": "node index.js"}}')
    (dummy_repo / "Dockerfile").write_text("CMD python app.py")
    (dummy_repo / "Makefile").write_text("run:\n\tpython app.py")
    (dummy_repo / "lib.py").write_text("def my_func():\n    return 1")

    dummy_manifest_path = Path("./outputs/dummy_repo/manifest")
    dummy_manifest_path.mkdir(parents=True, exist_ok=True)
    dummy_classified_files = dummy_manifest_path / "classified_files.json"
    dummy_classified_files.write_text(json.dumps([
        {"file": "app.py", "type": "SOURCE_CODE", "language": "python"},
        {"file": "package.json", "type": "CONFIG", "language": "json"},
        {"file": "Dockerfile", "type": "BUILD", "language": "dockerfile"},
        {"file": "Makefile", "type": "BUILD", "language": "makefile"},
        {"file": "lib.py", "type": "SOURCE_CODE", "language": "python"},
    ]))

    output_dir = Path("./outputs/dummy_repo/analysis")
    detect_entrypoints(dummy_repo, dummy_classified_files, output_dir / "entrypoints.json")

    # Clean up
    import shutil
    shutil.rmtree(dummy_repo)
    # shutil.rmtree(Path("./outputs"))
