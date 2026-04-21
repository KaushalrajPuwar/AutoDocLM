import json
from pathlib import Path

def generate_import_graph(import_graph_path: Path, centrality_scores_path: Path, output_file: Path, top_n: int = 30):
    with open(import_graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
        
    with open(centrality_scores_path, 'r', encoding='utf-8') as f:
        centrality_data = json.load(f)
        
    # Get top N files
    top_files = sorted(centrality_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_file_names = set(f[0] for f in top_files)
    
    mermaid_lines = ["graph TD"]
    for edge in graph_data.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        if src in top_file_names and tgt in top_file_names:
            s_safe = src.replace('/', '_').replace('.', '_').replace('-', '_')
            t_safe = tgt.replace('/', '_').replace('.', '_').replace('-', '_')
            mermaid_lines.append(f"    {s_safe}[\"{src}\"] --> {t_safe}[\"{tgt}\"]")
            
    content = "```mermaid\n" + "\n".join(mermaid_lines) + "\n```\n"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
