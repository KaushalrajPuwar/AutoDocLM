import ast
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CallVisitor(ast.NodeVisitor):
    def __init__(self, file_alias_map: dict):
        self.calls = {}
        self.current_function = None
        self.file_alias_map = file_alias_map

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current_function = node.name
        self.calls[self.current_function] = []
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node: ast.Call):
        if not self.current_function:
            self.generic_visit(node)
            return

        func = node.func
        call_name = None

        if isinstance(func, ast.Name): # e.g., my_function()
            call_name = func.id
        elif isinstance(func, ast.Attribute): # e.g., my_module.my_function()
            # This is a simplified resolver. A full one would trace back the origin of `func.value`.
            source_name = []
            temp_attr = func
            while isinstance(temp_attr, ast.Attribute):
                source_name.insert(0, temp_attr.attr)
                temp_attr = temp_attr.value
            if isinstance(temp_attr, ast.Name):
                source_name.insert(0, temp_attr.id)
            
            call_name = ".".join(source_name)

        if call_name:
            # Resolve alias if it exists
            # e.g., if call_name is 'us.get_user' and 'us' is an alias for 'user_service'
            # we want to find 'user_service.get_user'
            
            parts = call_name.split('.')
            if parts[0] in self.file_alias_map:
                resolved_base = self.file_alias_map[parts[0]]
                resolved_call = ".".join([resolved_base] + parts[1:])
                self.calls[self.current_function].append(resolved_call)
            else:
                self.calls[self.current_function].append(call_name)

        self.generic_visit(node)


def extract_cross_file_calls(
    repo_path: Path, 
    chunks_path: Path, 
    import_graph_path: Path, 
    output_path: Path
):
    """
    Extracts a map of function calls that cross file boundaries.
    """
    logger.info("Starting cross-file call extraction...")

    try:
        with open(import_graph_path, "r", encoding="utf-8") as f:
            import_graph_data = json.load(f)
        alias_map = import_graph_data.get("alias_map", {})
    except FileNotFoundError:
        logger.error(f"Import graph not found at {import_graph_path}")
        return

    # 1. Build a map of all defined functions/symbols in the repo
    symbol_map = {} # { "module.path.ClassName.method_name": "path/to/file.py" }
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                lang = chunk.get("language", "").lower()
                if lang == "python":
                    # We need to map the symbol's canonical path to its file
                    # e.g., src.models.User -> src/models.py
                    file_rel_path = chunk["file"]
                    module_path = file_rel_path.replace(".py", "").replace("/", ".")
                    
                    symbol_name = chunk.get("symbol")
                    if symbol_name:
                        full_symbol_path = f"{module_path}.{symbol_name}"
                        symbol_map[full_symbol_path] = file_rel_path
                    
                    # Also map the file-level module itself
                    symbol_map[module_path] = file_rel_path

    except FileNotFoundError:
        logger.error(f"Chunks file not found at {chunks_path}")
        return

    cross_calls = {}

    # 2. Iterate through each file and find calls
    for file_str, file_aliases in alias_map.items():
        file_path = repo_path / file_str
        if not file_path.is_file():
            continue

        cross_calls[file_str] = {}
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=file_str)
            
            visitor = CallVisitor(file_aliases)
            visitor.visit(tree)

            # 3. For each call, check if it resolves to a different file
            for calling_func, called_symbols in visitor.calls.items():
                for called_symbol in called_symbols:
                    # Find the file this symbol belongs to
                    target_file = symbol_map.get(called_symbol)
                    
                    if target_file and target_file != file_str:
                        if calling_func not in cross_calls[file_str]:
                            cross_calls[file_str][calling_func] = {"calls": []}
                        
                        call_info = {
                            "file": target_file,
                            "function": called_symbol.split('.')[-1] # Just the function name
                        }
                        if call_info not in cross_calls[file_str][calling_func]["calls"]:
                            cross_calls[file_str][calling_func]["calls"].append(call_info)

        except Exception as e:
            logger.error(f"Could not parse {file_str} for cross-file calls: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cross_calls, f, indent=2)

    logger.info(f"Cross-file call extraction complete. Results saved to {output_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dummy_repo = Path("./dummy_repo")
    dummy_repo.mkdir(exist_ok=True)
    
    (dummy_repo / "src").mkdir(exist_ok=True)
    (dummy_repo / "src" / "main.py").write_text("from . import utils\n\ndef start():\n    utils.do_work()")
    (dummy_repo / "src" / "utils.py").write_text("def do_work():\n    print('doing work')")

    # Dummy chunks.jsonl
    dummy_chunks_path = Path("./outputs/dummy_repo/chunks")
    dummy_chunks_path.mkdir(parents=True, exist_ok=True)
    chunks_file = dummy_chunks_path / "chunks.jsonl"
    with open(chunks_file, "w") as f:
        f.write(json.dumps({"file": "src/main.py", "symbol": "start", "language": "python"}) + "\n")
        f.write(json.dumps({"file": "src/utils.py", "symbol": "do_work", "language": "python"}) + "\n")

    # Dummy import_graph.json
    dummy_analysis_path = Path("./outputs/dummy_repo/analysis")
    dummy_analysis_path.mkdir(parents=True, exist_ok=True)
    import_graph_file = dummy_analysis_path / "import_graph.json"
    import_graph_file.write_text(json.dumps({
        "nodes": [{"id": "src/main.py"}, {"id": "src/utils.py"}],
        "links": [{"source": "src/main.py", "target": "src/utils.py"}],
        "alias_map": {
            "src/main.py": {"utils": "src.utils"},
            "src/utils.py": {}
        }
    }))

    output_file = dummy_analysis_path / "cross_file_calls.json"
    extract_cross_file_calls(dummy_repo, chunks_file, import_graph_file, output_file)

    import shutil
    shutil.rmtree(dummy_repo)
    # shutil.rmtree(Path("./outputs"))
