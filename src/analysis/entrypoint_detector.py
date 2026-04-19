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
        return re.findall(r"^(run|start|dev|build|install):", content, re.MULTILINE)
    except Exception:
        return []


def detect_entrypoints(repo_path: Path, classified_files_path: Path, output_path: Path):
    """
    Detects potential entrypoints in the repository and saves them to a JSON file.

    classified_files.json schema:
        { "relative/path/to/file.py": { "category": "SOURCE_CODE", "is_test": false }, ... }
    """
    logger.info("Starting entrypoint detection...")

    try:
        with open(classified_files_path, "r", encoding="utf-8") as f:
            classified_files = json.load(f)
    except FileNotFoundError:
        logger.error(f"Classified files manifest not found at {classified_files_path}")
        return

    entrypoints = []

    # classified_files is a dict: { filepath_str: { "category": ..., "is_test": ... } }
    for file_str, metadata in classified_files.items():
        file_path = repo_path / file_str
        category = metadata.get("category", "")

        if category == "SOURCE_CODE" and file_str.endswith(".py"):
            if _find_python_main(file_path):
                entrypoints.append({
                    "file": file_str,
                    "type": "python_main",
                    "confidence": "high"
                })
        elif file_path.name == "package.json":
            scripts = _find_package_json_scripts(file_path)
            if scripts:
                entrypoints.append({
                    "file": file_str,
                    "type": "npm_scripts",
                    "scripts": scripts,
                    "confidence": "high"
                })
        elif file_path.name.lower() == "dockerfile":
            docker_cmds = _find_dockerfile_entrypoint(file_path)
            if docker_cmds:
                entrypoints.append({
                    "file": file_str,
                    "type": "docker",
                    "commands": docker_cmds,
                    "confidence": "high"
                })
        elif file_path.name.lower() == "makefile":
            targets = _find_makefile_targets(file_path)
            if targets:
                entrypoints.append({
                    "file": file_str,
                    "type": "makefile",
                    "targets": targets,
                    "confidence": "medium"
                })

    output_data = {"entrypoints": entrypoints}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(
        f"Entrypoint detection complete. Found {len(entrypoints)} potential entrypoints. "
        f"Results saved to {output_path}"
    )
