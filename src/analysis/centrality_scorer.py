"""
Centrality Scorer (Step 5.5).

Calculates the structural importance (centrality) of each file in the import graph
using PageRank. A file's score reflects not just how often it is imported, but how
important its importers are — a standard measure of architectural significance.

The PageRank damping factor (alpha) is exposed as a parameter so it can be
controlled via RunConfig in future if needed, without changing this module.
"""
import json
import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def score_centrality(
    import_graph_path: Path,
    output_path: Path,
    alpha: float = 0.85,
):
    """
    Calculate PageRank centrality for every file node in the import graph.

    Args:
        import_graph_path:  Path to import_graph.json.
        output_path:        Path to write centrality_scores.json.
        alpha:              PageRank damping factor. Default 0.85 (standard
                            starting point; expose via RunConfig if tuning needed).
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
        centrality: dict = {}
    else:
        # PageRank models a recursive "random surfer": a file is important not
        # just if it's imported often, but if it is imported by other important files.
        centrality = nx.pagerank(G, alpha=alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(centrality, f, indent=2)

    logger.info(f"Centrality scoring complete. {len(centrality)} files scored → {output_path}")
