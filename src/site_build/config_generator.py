import yaml
from pathlib import Path
from .nav_builder import build_nav_tree

def generate_mkdocs_config(project_dir: Path, repo_name: str):
    docs_dir = project_dir / "docs"
    mkdocs_file = project_dir / "mkdocs.yml"
    
    # 1. Build nav tree
    nav_tree = build_nav_tree(docs_dir)
    
    # 2. Build configuration (excluding nav temporarily)
    config = {
        "site_name": f"{repo_name} Documentation",
        "docs_dir": "docs",
        "site_dir": "site",
        "theme": {
            "name": "material",
            "features": [
                "navigation.tabs",
                "navigation.sections",
                "navigation.expand",
                "search.suggest",
                "search.highlight"
            ],
            "palette": {
                "scheme": "slate",
                "primary": "indigo",
                "accent": "indigo"
            }
        },
        "markdown_extensions": [
            "pymdownx.highlight",
            "pymdownx.details",
            "pymdownx.magiclink",
            "pymdownx.inlinehilite",
            "admonition",
            "tables"
        ]
    }
    
    # Dump the first half of the config
    with open(mkdocs_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    # Append mermaid custom fences config (YAML !python tags are tricky)
    with open(mkdocs_file, 'a', encoding='utf-8') as f:
        f.write("- pymdownx.superfences:\n")
        f.write("    custom_fences:\n")
        f.write("      - name: mermaid\n")
        f.write("        class: mermaid\n")
        f.write("        format: !!python/name:pymdownx.superfences.fence_code_format\n\n")
        
    # Finally, append the nav tree to the end
    with open(mkdocs_file, 'a', encoding='utf-8') as f:
        yaml.dump({"nav": nav_tree}, f, default_flow_style=False, sort_keys=False)
        
    return mkdocs_file
