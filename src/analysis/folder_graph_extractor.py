"""
Folder Graph Extractor (Step 5.6).

Aggregates the file-level import graph into a folder-level view of dependencies.
All results are written to a SINGLE output file (folder_graphs.json) rather than
one file per folder, which eliminates the filename collision risk from path
sanitization.

Key design decisions:
- Root-level files (parent == ".") are stored under the canonical key "."
  rather than the repo directory name (which is environment-dependent).
- Output is a single JSON dict keyed by folder path, not N separate files.
  This prevents collisions where data/raw and data_raw map to the same filename.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def extract_folder_graphs(
    import_graph_path: Path,
    output_path: Path,
    repo_root: Path,
):
    """
    Aggregate the file-level import graph into a folder-level dependency view.

    Output format (single JSON file):
    {
      "src/flask": {
        "folder": "src/flask",
        "internal_files": [...],
        "incoming_dependencies": [...],
        "outgoing_dependencies": [...]
      },
      ".": {
        "folder": ".",
        ...
      }
    }

    The root folder key is always "." (not the checkout directory name).

    Args:
        import_graph_path:  Path to import_graph.json.
        output_path:        Path to write folder_graphs.json (a single file).
        repo_root:          Absolute path to the repository root (used for logging only).
    """
    logger.info("Starting folder graph extraction...")

    try:
        with open(import_graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        file_graph = nx.node_link_graph(graph_data)
    except FileNotFoundError:
        logger.error(f"Import graph not found at {import_graph_path}")
        return

    if not file_graph.nodes():
        logger.warning("Import graph is empty. Skipping folder graph extraction.")
        return

    # Canonical folder key: always use POSIX-style relative paths.
    # Root-level files → "." (not the repo directory name)
    def canonical_folder(file_node: str) -> str:
        parent = str(Path(file_node).parent)
        # Path("config.py").parent is "." on all platforms
        return parent if parent != "" else "."

    folder_data: dict[str, dict] = defaultdict(lambda: {
        "folder": "",
        "internal_files": set(),
        "incoming_dependencies": set(),
        "outgoing_dependencies": set(),
    })

    for file_node in file_graph.nodes():
        folder = canonical_folder(file_node)
        folder_data[folder]["folder"] = folder
        folder_data[folder]["internal_files"].add(file_node)

    for u, v in file_graph.edges():
        folder_u = canonical_folder(u)
        folder_v = canonical_folder(v)
        if folder_u != folder_v:
            folder_data[folder_u]["outgoing_dependencies"].add(folder_v)
            folder_data[folder_v]["incoming_dependencies"].add(folder_u)

    # Serialise: sets → sorted lists for stable, diffable output
    serialisable: dict[str, dict] = {}
    for folder, data in folder_data.items():
        serialisable[folder] = {
            "folder": data["folder"],
            "internal_files": sorted(data["internal_files"]),
            "incoming_dependencies": sorted(data["incoming_dependencies"]),
            "outgoing_dependencies": sorted(data["outgoing_dependencies"]),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)

    logger.info(
        f"Folder graph extraction complete. "
        f"{len(serialisable)} folder entries → {output_path}"
    )
