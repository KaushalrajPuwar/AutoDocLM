import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def build_nav_tree(docs_dir: Path):
    """
    Scrape docs directory and construct a nested nav dict for mkdocs.yml
    """
    nav = []
    
    # Homepage / Index
    if (docs_dir / "index.md").exists():
        nav.append({"Home": "index.md"})
        
    # Setup
    if (docs_dir / "setup.md").exists():
        nav.append({"Setup Guide": "setup.md"})
        
    # Architecture
    if (docs_dir / "architecture.md").exists():
        nav.append({"Architecture Overview": "architecture.md"})
        
    # Diagrams
    diagrams_dir = docs_dir / "diagrams"
    if diagrams_dir.exists():
        diagram_nav = []
        if (diagrams_dir / "component_graph.md").exists():
            diagram_nav.append({"Component Graph": "diagrams/component_graph.md"})
        if (diagrams_dir / "import_graph_top30.md").exists():
            diagram_nav.append({"Top 30 Imports": "diagrams/import_graph_top30.md"})
            
        if diagram_nav:
            nav.append({"Diagrams": diagram_nav})

    # Modules
    modules_dir = docs_dir / "modules"
    if modules_dir.exists():
        modules_nav = []
        for file_path in sorted(modules_dir.glob("*.md")):
            label = file_path.stem.replace("__", "/")
            modules_nav.append({label: f"modules/{file_path.name}"})
        if modules_nav:
            nav.append({"Modules (Folders)": modules_nav})
            
    # Files
    files_dir = docs_dir / "files"
    if files_dir.exists():
        files_nav = []
        for file_path in sorted(files_dir.glob("*.md")):
            label = file_path.stem.replace("__", "/")
            files_nav.append({label: f"files/{file_path.name}"})
        if files_nav:
            nav.append({"File Deep-Dives": files_nav})
            
    # Reference
    if (docs_dir / "reference.md").exists():
        nav.append({"Technical Reference": "reference.md"})
        
    return nav
