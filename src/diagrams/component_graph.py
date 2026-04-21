import json
from pathlib import Path

def generate_component_graph(folder_graphs_path: Path, output_file: Path):
    with open(folder_graphs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    mermaid_lines = ["graph TD"]
    for folder, info in data.items():
        for target in info.get("outgoing_dependencies", []):
            f_safe = folder.replace('.', 'root').replace('/', '_').replace('-', '_')
            t_safe = target.replace('.', 'root').replace('/', '_').replace('-', '_')
            mermaid_lines.append(f"    {f_safe}[\"{folder}\"] --> {t_safe}[\"{target}\"]")
            
    content = "```mermaid\n" + "\n".join(mermaid_lines) + "\n```\n"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
