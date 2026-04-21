import logging
from pathlib import Path
from .component_graph import generate_component_graph
from .import_graph import generate_import_graph

logger = logging.getLogger(__name__)

def run_diagram_generation(project_dir: Path):
    logger.info("=== STEP 9: Diagram Generation ===")
    analysis_dir = project_dir / "analysis"
    diagrams_dir = project_dir / "docs" / "diagrams"
    
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    
    folder_graphs_path = analysis_dir / "folder_graphs.json"
    if folder_graphs_path.exists():
        out_path = diagrams_dir / "component_graph.md"
        generate_component_graph(folder_graphs_path, out_path)
        logger.info(f"Generated {out_path}")
    else:
        logger.warning(f"File not found: {folder_graphs_path}")
        
    import_graph_path = analysis_dir / "import_graph.json"
    centrality_scores_path = analysis_dir / "centrality_scores.json"
    if import_graph_path.exists() and centrality_scores_path.exists():
        out_path = diagrams_dir / "import_graph_top30.md"
        generate_import_graph(import_graph_path, centrality_scores_path, out_path)
        logger.info(f"Generated {out_path}")
    else:
        logger.warning(f"Files not found: {import_graph_path} or {centrality_scores_path}")
        
    logger.info("Diagram Generation Complete.")
