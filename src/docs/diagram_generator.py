import json
import logging
import re
from pathlib import Path

from src.config import RunConfig

logger = logging.getLogger(__name__)

TOP_IMPORT_GRAPH_FILES = 30


def _sanitize_mermaid_id(value: str, prefix: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    if not safe or safe[0].isdigit():
        safe = f"{prefix}_{safe}"
    return safe


def _write_markdown(path: Path, title: str, intro: str, mermaid_lines: list[str], note: str = "") -> None:
    content: list[str] = [f"# {title}", "", intro, ""]
    if note:
        content.extend([f"> {note}", ""])
    content.extend(["```mermaid", *mermaid_lines, "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def _safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse JSON at %s: %s", path, exc)
        return None


def _fallback_scores(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, float]:
    indegree: dict[str, int] = {node: 0 for node in nodes}
    for _, target in edges:
        if target in indegree:
            indegree[target] += 1
    return {node: float(score) for node, score in indegree.items()}


def generate_component_graph_markdown(project_dir: Path) -> bool:
    folder_graphs_path = project_dir / "analysis" / "folder_graphs.json"
    output_path = project_dir / "docs" / "diagrams" / "component_graph.md"

    folder_graphs = _safe_load_json(folder_graphs_path)
    if not folder_graphs:
        _write_markdown(
            output_path,
            "Component Graph",
            "No folder-level dependency data was available for this run.",
            ["graph LR", "  no_data[\"No folder dependency graph available\"]"],
            note="This page is a deterministic fallback and indicates missing Step 5.6 output.",
        )
        return False

    folders = sorted(folder_graphs.keys())
    id_map = {folder: _sanitize_mermaid_id(folder, "folder") for folder in folders}

    # Group folders by their parent directory for subgraphs
    groups: dict[str, list[str]] = {}
    for folder in folders:
        parent = str(Path(folder).parent)
        if parent == ".":
            parent = "root"
        groups.setdefault(parent, []).append(folder)

    mermaid_lines = ["graph LR", ""]
    
    # Define styles
    mermaid_lines.extend([
        "  classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;",
        "  classDef core fill:#e1f5fe,stroke:#01579b,stroke-width:2px;",
        "  classDef util fill:#f3e5f5,stroke:#4a148c,stroke-width:1px;",
        "  classDef test fill:#fff3e0,stroke:#e65100,stroke-width:1px,stroke-dasharray: 5 5;",
        "  classDef api fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;",
        ""
    ])

    edge_pairs: set[tuple[str, str]] = set()
    for folder in folders:
        outgoing = folder_graphs.get(folder, {}).get("outgoing_dependencies", [])
        for dep in outgoing:
            if dep in id_map and dep != folder:
                edge_pairs.add((folder, dep))

    for parent, members in sorted(groups.items()):
        if len(members) > 1 or parent != "root":
            safe_parent = _sanitize_mermaid_id(parent, "group")
            mermaid_lines.append(f"  subgraph {safe_parent} [\"{parent}\"]")
            for folder in sorted(members):
                node_id = id_map[folder]
                file_count = len(folder_graphs.get(folder, {}).get("internal_files", []))
                label = Path(folder).name if folder != "." else "root"
                if file_count > 0:
                    label += f" ({file_count} files)"
                mermaid_lines.append(f'    {node_id}["{label}"]')
            mermaid_lines.append("  end")
        else:
            for folder in members:
                node_id = id_map[folder]
                file_count = len(folder_graphs.get(folder, {}).get("internal_files", []))
                label = Path(folder).name if folder != "." else "root"
                if file_count > 0:
                    label += f" ({file_count} files)"
                mermaid_lines.append(f'  {node_id}["{label}"]')

    mermaid_lines.append("")
    for src, dst in sorted(edge_pairs):
        mermaid_lines.append(f"  {id_map[src]} --> {id_map[dst]}")

    # Assign classes
    for folder in folders:
        node_id = id_map[folder]
        lower_folder = folder.lower()
        if any(x in lower_folder for x in ["test", "spec", "mock", "fixture"]):
            mermaid_lines.append(f"  class {node_id} test")
        elif any(x in lower_folder for x in ["api", "route", "server", "endpoint", "controller"]):
            mermaid_lines.append(f"  class {node_id} api")
        elif any(x in lower_folder for x in ["util", "helper", "common", "tool", "shared"]):
            mermaid_lines.append(f"  class {node_id} util")
        elif any(x in lower_folder for x in ["core", "service", "logic", "src", "app", "lib"]):
            mermaid_lines.append(f"  class {node_id} core")
        else:
            mermaid_lines.append(f"  class {node_id} default")

    _write_markdown(
        output_path,
        "Component Graph",
        "This diagram shows folder-level dependencies with grouped sub-systems and file counts.",
        mermaid_lines,
        note="The graph uses color-coding: Blue (Core/Source), Green (API/Interface), Purple (Utils/Shared), Orange Dash (Tests).",
    )
    return True


def generate_import_graph_top30_markdown(project_dir: Path) -> bool:
    import_graph_path = project_dir / "analysis" / "import_graph.json"
    centrality_path = project_dir / "analysis" / "centrality_scores.json"
    output_path = project_dir / "docs" / "diagrams" / "import_graph_top30.md"

    graph_data = _safe_load_json(import_graph_path)
    if not graph_data:
        _write_markdown(
            output_path,
            "Import Graph (Top 30)",
            "No file-level import graph was available for this run.",
            ["graph TD", "  no_data[\"No import graph available\"]"],
            note="This page is a deterministic fallback and indicates missing Step 5.3 output.",
        )
        return False

    nodes = [node.get("id") for node in graph_data.get("nodes", []) if node.get("id")]
    edges = [
        (edge.get("source"), edge.get("target"))
        for edge in graph_data.get("edges", [])
        if edge.get("source") and edge.get("target")
    ]

    if not nodes:
        _write_markdown(
            output_path,
            "Import Graph (Top 30)",
            "Import graph exists but contains no nodes.",
            ["graph TD", "  no_data[\"No import graph nodes available\"]"],
            note="This can happen when all source files were filtered out before static analysis.",
        )
        return False

    centrality = _safe_load_json(centrality_path)
    used_fallback = False
    if not centrality:
        centrality = _fallback_scores(nodes, edges)
        used_fallback = True

    score_map = {node: float(centrality.get(node, 0.0)) for node in nodes}
    top_nodes = sorted(nodes, key=lambda n: (-score_map[n], n))[:TOP_IMPORT_GRAPH_FILES]
    top_set = set(top_nodes)

    filtered_edges = sorted({(src, dst) for src, dst in edges if src in top_set and dst in top_set})

    basename_counts: dict[str, int] = {}
    for node in top_nodes:
        basename = Path(node).name
        basename_counts[basename] = basename_counts.get(basename, 0) + 1

    labels: dict[str, str] = {}
    for node in top_nodes:
        basename = Path(node).name
        labels[node] = basename if basename_counts[basename] == 1 else node

    id_map = {node: f"file_{idx}" for idx, node in enumerate(top_nodes)}

    mermaid_lines = ["graph TD"]
    for node in top_nodes:
        label = labels[node].replace('"', "'")
        mermaid_lines.append(f'  {id_map[node]}["{label}"]')

    for src, dst in filtered_edges:
        mermaid_lines.append(f"  {id_map[src]} --> {id_map[dst]}")

    note = (
        "Centrality scores were unavailable, so ranking used fallback in-degree counts."
        if used_fallback
        else "Nodes are ranked by PageRank centrality from Step 5.5."
    )

    _write_markdown(
        output_path,
        "Import Graph (Top 30)",
        "This diagram shows import dependencies between the 30 most central files.",
        mermaid_lines,
        note=note,
    )
    return True


def run_step_9(config: RunConfig, project_dir: str) -> None:
    """Generate deterministic Mermaid diagrams for architecture visualization."""
    del config  # Stage 9 is deterministic and does not require runtime model settings.

    project_path = Path(project_dir)
    logger.info("Starting Step 9: Mermaid Diagram Generation")

    component_ok = generate_component_graph_markdown(project_path)
    import_ok = generate_import_graph_top30_markdown(project_path)

    logger.info(
        "Step 9 complete. component_graph=%s import_graph_top30=%s",
        "ok" if component_ok else "fallback",
        "ok" if import_ok else "fallback",
    )
