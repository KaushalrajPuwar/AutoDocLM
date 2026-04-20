"""
Package root detector for the Static Analysis pipeline.

Determines the list of "package roots" — directories from which Python modules
are importable — using a priority-ordered strategy that works for the full
spectrum of repository layouts without any filesystem fuzzy matching.

Priority order:
  1. Explicit declaration in pyproject.toml or setup.cfg
  2. Flat layout: top-level directories with __init__.py → root is "."
  3. src/ layout:  src/ exists and its children have __init__.py → root is "src/"
  4. Default fallback: "."
"""
from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def detect_package_roots(repo_root: Path) -> list[str]:
    """
    Return a list of package root paths (relative to repo_root) from which
    Python modules are importable. The list is ordered by priority — callers
    should try each root in turn and stop at the first match.

    Examples:
        Flat layout:     [".]
        src/ layout:     ["src"]
        Explicit config: ["lib"]  (if pyproject.toml says package_dir = {"": "lib"})

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        List of relative directory strings (never empty — always contains at
        least "." as the final fallback).
    """
    # --- Priority 1: Explicit declaration in pyproject.toml ---
    roots = _roots_from_pyproject(repo_root)
    if roots:
        logger.debug(f"Package roots from pyproject.toml: {roots}")
        return roots

    # --- Priority 1b: Explicit declaration in setup.cfg ---
    roots = _roots_from_setup_cfg(repo_root)
    if roots:
        logger.debug(f"Package roots from setup.cfg: {roots}")
        return roots

    # --- Priority 2: Flat layout ---
    # If any top-level directory contains __init__.py, modules are importable
    # from the root. This is the most common case.
    flat_packages = [
        d for d in repo_root.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
        and not d.name.startswith(".")
        and d.name not in ("build", "dist", ".venv", "venv", "env",
                           ".tox", "__pycache__", "node_modules")
    ]
    if flat_packages:
        logger.debug(f"Flat layout detected. Package root: '.'")
        return ["."]

    # --- Priority 3: src/ layout ---
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        src_packages = [
            d for d in src_dir.iterdir()
            if d.is_dir() and (d / "__init__.py").exists()
        ]
        if src_packages:
            logger.debug(f"src/ layout detected. Package root: 'src'")
            return ["src"]

    # --- Priority 4: Default fallback ---
    logger.debug("No package layout detected. Defaulting to root '.'")
    return ["."]


def resolve_import_to_file(
    import_str: str,
    package_roots: list[str],
    all_repo_files: list[str],
) -> str | None:
    """
    Resolve a dotted Python import string to a repo-relative file path.

    Tries each package root in order, stopping at the first match. Never
    falls back to fuzzy suffix scanning.

    Strategies per root (in order):
      1. Direct module:  flask.helpers → <root>/flask/helpers.py
      2. Package init:   flask         → <root>/flask/__init__.py

    Args:
        import_str:      Dotted import string, e.g. "flask.helpers".
        package_roots:   Ordered list of roots (from detect_package_roots).
        all_repo_files:  Pre-filtered list of repo-relative file paths.

    Returns:
        Repo-relative path string, or None if not resolvable (external import).
    """
    path_fragment = import_str.replace(".", os.sep)
    file_set = set(all_repo_files)

    for root in package_roots:
        prefix = "" if root == "." else root + os.sep

        # Direct file: flask/helpers.py
        candidate = prefix + path_fragment + ".py"
        if candidate in file_set:
            return candidate

        # Package __init__: flask/__init__.py
        candidate_init = prefix + path_fragment + os.sep + "__init__.py"
        if candidate_init in file_set:
            return candidate_init

    return None  # External or unresolvable import


def file_to_module_path(file_rel: str, package_roots: list[str]) -> str:
    """
    Convert a repo-relative file path to its Python dotted module name.

    Takes the first package root that matches as a prefix.

    Examples (with root "src"):
        src/flask/globals.py → flask.globals
        src/flask/__init__.py → flask

    Examples (with root "."):
        flask/globals.py → flask.globals
        flask/__init__.py → flask

    Args:
        file_rel:       Repo-relative path, e.g. "src/flask/globals.py".
        package_roots:  Ordered list of roots from detect_package_roots.

    Returns:
        Dotted module name string.
    """
    path = file_rel.replace(os.sep, "/")

    for root in package_roots:
        prefix = "" if root == "." else root.replace(os.sep, "/") + "/"
        if path.startswith(prefix):
            relative = path[len(prefix):]
            # Strip .py extension
            if relative.endswith(".py"):
                relative = relative[:-3]
            # Strip __init__ suffix
            if relative.endswith("/__init__"):
                relative = relative[:-9]
            return relative.replace("/", ".")

    # Fallback: use the raw path as-is
    return path.replace("/", ".").removesuffix(".py")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _roots_from_pyproject(repo_root: Path) -> list[str]:
    """Parse pyproject.toml for an explicit package_dir mapping."""
    if tomllib is None:
        return []
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        # [tool.setuptools.package-dir] or [tool.setuptools] package_dir
        pkg_dir = (
            data.get("tool", {})
                .get("setuptools", {})
                .get("package-dir", {})
        )
        if isinstance(pkg_dir, dict) and "" in pkg_dir:
            root = pkg_dir[""]
            if root:  # e.g. "" = "src"
                return [root]
    except Exception as e:
        logger.debug(f"Could not parse pyproject.toml for package_dir: {e}")
    return []


def _roots_from_setup_cfg(repo_root: Path) -> list[str]:
    """Parse setup.cfg for an explicit package_dir mapping."""
    setup_cfg = repo_root / "setup.cfg"
    if not setup_cfg.is_file():
        return []
    try:
        cfg = configparser.ConfigParser()
        cfg.read(setup_cfg, encoding="utf-8")
        raw = cfg.get("options", "package_dir", fallback=None)
        if raw:
            # Format: "=src" or "''=src" or "= src"
            for part in raw.splitlines():
                part = part.strip()
                if "=" in part:
                    key, _, val = part.partition("=")
                    key, val = key.strip().strip("'\""), val.strip()
                    if key == "" and val:
                        return [val]
    except Exception as e:
        logger.debug(f"Could not parse setup.cfg for package_dir: {e}")
    return []
