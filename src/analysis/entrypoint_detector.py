"""
Entrypoint Detector (Step 5.2).

Detects potential entrypoints in the repository:
  - Python __main__ blocks (detected via AST, not string matching)
  - package.json scripts, main, and bin fields (all three, not just scripts)
  - Dockerfile ENTRYPOINT / CMD directives
  - Makefile targets (all targets, not an allowlist of 5 words)

Key design decisions:
- Python __main__ detection uses AST so it handles all syntactically valid forms.
- Makefile: all targets are returned; semantic classification (run vs. build) is
  left to the LLM in downstream steps.
- package.json: scripts, main, and bin are all captured with different confidence
  levels to reflect how deliberately "user-facing" each field is.
- Category guards ensure only SOURCE_CODE files produce Python entrypoints, and
  only CONFIG files produce package.json / Dockerfile / Makefile entrypoints.
"""
import ast
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Python: AST-based __main__ detection
# ---------------------------------------------------------------------------

def _find_python_main(file_path: Path) -> bool:
    """
    Check if a Python file contains a top-level if __name__ == '__main__' block.
    Uses AST parsing — handles all syntactically valid forms regardless of
    quote style, spacing, or inline body placement.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except Exception:
        return False

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # Match: __name__ == '__main__'  or  '__main__' == __name__
        if isinstance(test, ast.Compare):
            left = test.left
            ops = test.ops
            comps = test.comparators
            if (
                len(ops) == 1
                and isinstance(ops[0], ast.Eq)
                and len(comps) == 1
            ):
                left_is_name = (
                    isinstance(left, ast.Name) and left.id == "__name__"
                    and isinstance(comps[0], ast.Constant)
                    and comps[0].value == "__main__"
                )
                right_is_name = (
                    isinstance(comps[0], ast.Name) and comps[0].id == "__name__"
                    and isinstance(left, ast.Constant)
                    and left.value == "__main__"
                )
                if left_is_name or right_is_name:
                    return True
    return False


# ---------------------------------------------------------------------------
# package.json: scripts, main, bin
# ---------------------------------------------------------------------------

def _find_package_json_entrypoints(file_path: Path) -> list[dict]:
    """
    Extract all entrypoint signals from a package.json file.

    Returns a list of entrypoint dicts, one per signal type:
      - bin:     confidence high  (deliberately exposed CLI executable)
      - scripts: confidence medium (tooling scripts)
      - main:    confidence low   (default module entry, not necessarily user-facing)
    """
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    results = []
    file_str = str(file_path)

    # bin: explicitly exposed CLI tools — highest signal
    bin_field = data.get("bin")
    if isinstance(bin_field, str) and bin_field:
        results.append({
            "file": file_str,
            "type": "npm_bin",
            "entry": bin_field,
            "confidence": "high",
        })
    elif isinstance(bin_field, dict) and bin_field:
        results.append({
            "file": file_str,
            "type": "npm_bin",
            "entry": bin_field,
            "confidence": "high",
        })

    # scripts: all npm scripts — medium signal
    scripts = data.get("scripts", {})
    if scripts:
        results.append({
            "file": file_str,
            "type": "npm_scripts",
            "scripts": list(scripts.keys()),
            "confidence": "medium",
        })

    # main: default module export — low signal (may just be a library)
    main = data.get("main") or data.get("exports")
    if main:
        results.append({
            "file": file_str,
            "type": "npm_main",
            "entry": main,
            "confidence": "low",
        })

    return results


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

def _find_dockerfile_entrypoint(file_path: Path) -> list[str]:
    """Extract ENTRYPOINT or CMD directives from a Dockerfile."""
    if not file_path.is_file():
        return []
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        entrypoint_match = re.search(
            r"^\s*ENTRYPOINT\s+(.*)", content, re.MULTILINE | re.IGNORECASE
        )
        cmd_match = re.search(
            r"^\s*CMD\s+(.*)", content, re.MULTILINE | re.IGNORECASE
        )
        results = []
        if entrypoint_match:
            results.append(entrypoint_match.group(1).strip())
        if cmd_match:
            results.append(cmd_match.group(1).strip())
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Makefile: all targets
# ---------------------------------------------------------------------------

def _find_makefile_targets(file_path: Path) -> list[str]:
    """
    Extract ALL named targets from a Makefile.

    Returns every target name — no allowlist filtering. Semantic classification
    of which targets are "run" vs. "build" vs. "test" is delegated to the LLM
    in downstream steps.
    """
    if not file_path.is_file():
        return []
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        # Match any target: lines starting with a word followed by a colon,
        # not indented (rules are indented with tabs, targets are not).
        return re.findall(r"^([A-Za-z0-9_\-\.]+)\s*:", content, re.MULTILINE)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_entrypoints(
    repo_path: Path,
    classified_files_path: Path,
    output_path: Path,
):
    """
    Detect potential entrypoints in the repository.

    Uses category guards so only SOURCE_CODE files produce Python entrypoints,
    and only CONFIG files produce package.json / Dockerfile / Makefile results.

    Args:
        repo_path:              Absolute path to the raw repository root.
        classified_files_path:  Path to classified_files.json.
        output_path:            Path to write entrypoints.json.
    """
    logger.info("Starting entrypoint detection...")

    try:
        with open(classified_files_path, "r", encoding="utf-8") as f:
            classified_files: dict = json.load(f)
    except FileNotFoundError:
        logger.error(f"Classified files manifest not found at {classified_files_path}")
        return

    entrypoints: list[dict] = []

    for file_str, metadata in classified_files.items():
        file_path = repo_path / file_str
        category = metadata.get("category", "")
        is_test = metadata.get("is_test", False)

        # Python __main__ — SOURCE_CODE only, non-test
        if category == "SOURCE_CODE" and not is_test and file_str.endswith(".py"):
            if _find_python_main(file_path):
                entrypoints.append({
                    "file": file_str,
                    "type": "python_main",
                    "confidence": "high",
                })

        # package.json — CONFIG category only
        elif category == "CONFIG" and file_path.name == "package.json":
            entrypoints.extend(_find_package_json_entrypoints(file_path))

        # Dockerfile — CONFIG category only
        elif category == "CONFIG" and file_path.name.lower() == "dockerfile":
            docker_cmds = _find_dockerfile_entrypoint(file_path)
            if docker_cmds:
                entrypoints.append({
                    "file": file_str,
                    "type": "docker",
                    "commands": docker_cmds,
                    "confidence": "high",
                })

        # Makefile — CONFIG category only
        elif category == "CONFIG" and file_path.name.lower() == "makefile":
            targets = _find_makefile_targets(file_path)
            if targets:
                entrypoints.append({
                    "file": file_str,
                    "type": "makefile",
                    "targets": targets,
                    "confidence": "medium",
                })

    output_data = {"entrypoints": entrypoints}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(
        f"Entrypoint detection complete. Found {len(entrypoints)} potential entrypoints → {output_path}"
    )
