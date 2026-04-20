"""
Dependency Extractor (Step 5.1).

Extracts declared external dependencies from all standard dependency manifests
found anywhere in the repository. Handles monorepos with multiple manifests.

Supported formats:
  Python:     requirements.txt, pyproject.toml, Pipfile, setup.cfg
  JavaScript: package.json

Note: setup.py is deliberately NOT parsed — it requires executing arbitrary code.

Key design decisions:
- All matching manifest files are discovered (not just the first one found).
  This correctly handles monorepos with per-package requirements.txt files.
- Results from all discovered files of the same type are merged and deduplicated.
"""
import configparser
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def _extract_from_requirements_txt(file_path: Path) -> list[str]:
    """Extract dependencies from a requirements.txt file."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            results = []
            for line in f:
                clean = line.split("#")[0].strip()
                if clean and not clean.startswith(("-r ", "-c ", "--")):
                    results.append(clean)
            return results
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def _extract_from_pyproject_toml(file_path: Path) -> list[str]:
    """Extract dependencies from a pyproject.toml file."""
    if tomllib is None or not file_path.is_file():
        return []
    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
        deps: list[str] = []
        # PEP 621 standard
        deps.extend(data.get("project", {}).get("dependencies", []))
        for group in data.get("project", {}).get("optional-dependencies", {}).values():
            deps.extend(group)
        # Poetry
        poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        deps.extend(
            f"{k}{v}" for k, v in poetry_deps.items() if k.lower() != "python"
        )
        return deps
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def _extract_from_pipfile(file_path: Path) -> list[str]:
    """Extract dependencies from a Pipfile (INI-like format)."""
    if not file_path.is_file():
        return []
    try:
        cfg = configparser.ConfigParser()
        cfg.read(file_path, encoding="utf-8")
        deps: list[str] = []
        for section in ("packages", "dev-packages"):
            if cfg.has_section(section):
                for pkg, ver in cfg.items(section):
                    ver = ver.strip().strip('"\'')
                    deps.append(f"{pkg}{ver}" if ver != "*" else pkg)
        return deps
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def _extract_from_setup_cfg(file_path: Path) -> list[str]:
    """Extract dependencies from a setup.cfg file."""
    if not file_path.is_file():
        return []
    try:
        cfg = configparser.ConfigParser()
        cfg.read(file_path, encoding="utf-8")
        raw = cfg.get("options", "install_requires", fallback="")
        # Multi-line values in configparser are joined by newlines
        deps = [line.strip() for line in raw.splitlines() if line.strip()]
        return deps
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def _extract_from_package_json(file_path: Path) -> list[str]:
    """Extract dependencies from a package.json file."""
    if not file_path.is_file():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        deps: list[str] = []
        for field in ("dependencies", "devDependencies", "peerDependencies"):
            for pkg, ver in data.get(field, {}).items():
                deps.append(f"{pkg}@{ver}")
        return deps
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


# ---------------------------------------------------------------------------
# Manifest registry: filename → (ecosystem, extractor)
# ---------------------------------------------------------------------------

_MANIFEST_REGISTRY: dict[str, tuple[str, object]] = {
    "requirements.txt":      ("python",      _extract_from_requirements_txt),
    "pyproject.toml":        ("python",      _extract_from_pyproject_toml),
    "Pipfile":               ("python",      _extract_from_pipfile),
    "setup.cfg":             ("python",      _extract_from_setup_cfg),
    "package.json":          ("javascript",  _extract_from_package_json),
}

# Directories to skip when searching — avoids virtual environments and caches
_SKIP_DIRS = {
    ".venv", "venv", "env", ".env", ".tox", "__pycache__",
    "node_modules", "build", "dist", ".mypy_cache", ".git",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_dependencies(repo_path: Path, output_path: Path):
    """
    Extract declared external dependencies from all manifest files in the repo.

    All matching manifest files are discovered (not just the first one), so
    monorepos with multiple requirements.txt or package.json files are handled
    correctly. Results are merged and deduplicated per ecosystem.

    Args:
        repo_path:    Absolute path to the raw repository root.
        output_path:  Path to write dependencies.json.
    """
    logger.info("Starting dependency extraction...")

    all_dependencies: dict[str, list[str]] = {
        "python": [],
        "javascript": [],
    }

    for manifest_name, (ecosystem, extractor) in _MANIFEST_REGISTRY.items():
        # Find ALL occurrences of this manifest file (handles monorepos).
        # rglob is safe here because we filter out known noise directories.
        found_files = [
            p for p in repo_path.rglob(manifest_name)
            if not any(skip in p.parts for skip in _SKIP_DIRS)
        ]

        if not found_files:
            continue

        logger.info(f"Found {len(found_files)} × {manifest_name}")
        for file_path in found_files:
            deps = extractor(file_path)  # type: ignore[call-arg]
            logger.debug(f"  {file_path.relative_to(repo_path)}: {len(deps)} deps")
            all_dependencies[ecosystem].extend(deps)

    # Deduplicate and sort per ecosystem for stable output
    for ecosystem in all_dependencies:
        all_dependencies[ecosystem] = sorted(set(all_dependencies[ecosystem]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_dependencies, f, indent=2)

    totals = {k: len(v) for k, v in all_dependencies.items()}
    logger.info(f"Dependency extraction complete. Totals: {totals} → {output_path}")
