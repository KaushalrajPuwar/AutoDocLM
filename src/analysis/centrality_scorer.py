import json
import logging
from pathlib import Path
import networkx as nx

logger = logging.getLogger(__name__)

def score_centrality(import_graph_path: Path, output_path: Path):
    """
    Calculates the centrality of each file in the import graph.
    """
    logger.info("Starting centrality scoring...")

    try:
        with open(import_graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        
        G = nx.node_link_graph(graph_data)
    except FileNotFoundError:
        logger.error(f"Import graph not found at {import_graph_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Could not decode import graph from {import_graph_path}")
        return

    if not G.nodes():
        logger.warning("Import graph is empty. Skipping centrality scoring.")
        centrality = {}
    else:
        # We use PageRank (as defined in our architectural specs) because it
        # models a recursive "random surfer", meaning a file is important not just 
        # if it's imported often, but if it is imported by *other important files*.
        centrality = nx.pagerank(G, alpha=0.85)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(centrality, f, indent=2)

    logger.info(f"Centrality scoring complete. Results saved to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    dummy_analysis_path = Path("./outputs/dummy_repo/analysis")
    dummy_analysis_path.mkdir(parents=True, exist_ok=True)
    import_graph_file = dummy_analysis_path / "import_graph.json"
    
    # Create a dummy graph: main -> utils, main -> models, utils -> models
    # models should be most central
    graph = nx.DiGraph()
    graph.add_edges_from([("src/main.py", "src/utils.py"), ("src/main.py", "src/models.py"), ("src/utils.py", "src/models.py")])
    graph_data = nx.node_link_data(graph)

    with open(import_graph_file, "w") as f:
        json.dump(graph_data, f)

    output_file = dummy_analysis_path / "centrality_scores.json"
    score_centrality(import_graph_file, output_file)

    with open(output_file, "r") as f:
        scores = json.load(f)
        print("Generated centrality scores:")
        print(scores)
        assert scores["src/models.py"] > scores["src/utils.py"]
        assert scores["src/utils.py"] > scores["src/main.py"]

    # import shutil
    # shutil.rmtree(Path("./outputs"))
