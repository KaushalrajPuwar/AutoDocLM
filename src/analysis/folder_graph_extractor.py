import json
import logging
from pathlib import Path
import networkx as nx
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

def extract_folder_graphs(import_graph_path: Path, output_dir: Path, repo_root: Path):
    """
    Aggregates the file-level import graph into a folder-level view of dependencies.
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

    folder_dependencies = defaultdict(lambda: {
        "folder": "",
        "internal_files": set(),
        "incoming_dependencies": set(),
        "outgoing_dependencies": set(),
    })

    # Get all folders from the nodes in the graph
    all_folders = set()
    for file_node in file_graph.nodes():
        folder = str(Path(file_node).parent)
        if folder == ".":
            folder = str(repo_root.name) # Root folder
        all_folders.add(folder)
        folder_dependencies[folder]["internal_files"].add(file_node)


    for u, v in file_graph.edges():
        folder_u = str(Path(u).parent)
        folder_v = str(Path(v).parent)
        
        if folder_u == ".": folder_u = str(repo_root.name)
        if folder_v == ".": folder_v = str(repo_root.name)

        if folder_u != folder_v:
            folder_dependencies[folder_u]["outgoing_dependencies"].add(folder_v)
            folder_dependencies[folder_v]["incoming_dependencies"].add(folder_u)

    output_dir.mkdir(parents=True, exist_ok=True)

    for folder, data in folder_dependencies.items():
        data["folder"] = folder
        # Convert sets to sorted lists for stable JSON output
        data["internal_files"] = sorted(list(data["internal_files"]))
        data["incoming_dependencies"] = sorted(list(data["incoming_dependencies"]))
        data["outgoing_dependencies"] = sorted(list(data["outgoing_dependencies"]))
        
        # Sanitize folder name for use as a filename
        folder_filename = folder.replace("/", "_").replace("\\", "_") + ".json"
        output_path = output_dir / folder_filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    logger.info(f"Folder graph extraction complete. Generated {len(folder_dependencies)} folder graphs in {output_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    dummy_repo = Path("./dummy_repo")
    dummy_repo.mkdir(exist_ok=True)
    
    dummy_analysis_path = Path("./outputs/dummy_repo/analysis")
    dummy_analysis_path.mkdir(parents=True, exist_ok=True)
    import_graph_file = dummy_analysis_path / "import_graph.json"
    
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("src/main.py", "src/utils.py"), 
        ("src/main.py", "src/models/user.py"), 
        ("src/utils.py", "src/models/user.py"),
        ("src/main.py", "config.py") # Edge to root
    ])
    graph_data = nx.node_link_data(graph)

    with open(import_graph_file, "w") as f:
        json.dump(graph_data, f)

    output_folder_graphs_dir = dummy_analysis_path / "folder_graphs"
    extract_folder_graphs(import_graph_file, output_folder_graphs_dir, dummy_repo)

    # Verification
    src_graph_path = output_folder_graphs_dir / "src.json"
    with open(src_graph_path, "r") as f:
        src_data = json.load(f)
        print("Generated src folder graph:")
        print(src_data)
        assert "src/models" in src_data["outgoing_dependencies"]
        assert "dummy_repo" in src_data["outgoing_dependencies"]
    
    src_models_graph_path = output_folder_graphs_dir / "src_models.json"
    with open(src_models_graph_path, "r") as f:
        models_data = json.load(f)
        print("\nGenerated src/models folder graph:")
        print(models_data)
        assert "src" in models_data["incoming_dependencies"]
        assert not models_data["outgoing_dependencies"]


    import shutil
    shutil.rmtree(dummy_repo)
    # shutil.rmtree(Path("./outputs"))
