import ast
import json
import logging
from pathlib import Path
import networkx as nx
import os

logger = logging.getLogger(__name__)

class ImportVisitor(ast.NodeVisitor):
    """
    AST visitor to find all imports in a Python file.
    """
    def __init__(self, file_path: str, repo_root: Path):
        self.imports = set()
        self.aliases = {}
        self.file_path = file_path
        self.repo_root = repo_root
        self.current_file_dir = Path(file_path).parent

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.asname:
                self.aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module
        if module is None: # e.g., from . import something
            return

        # Handle relative imports
        if node.level > 0:
            # from ..utils import something
            # level 1: from . import ...
            # level 2: from .. import ...
            
            # Resolve the relative path to an absolute path from the repo root
            base_path = self.current_file_dir
            for _ in range(node.level -1):
                base_path = base_path.parent
            
            # The module name is appended to the resolved base path
            # e.g., from .foo import bar -> current_dir/foo.py
            # e.g., from ..foo import bar -> parent_dir/foo.py
            full_module_path = (base_path / module.replace('.', os.sep)).resolve()
            
            # Make it relative to the repo root
            try:
                module_as_path = full_module_path.relative_to(self.repo_root)
                module_name = str(module_as_path).replace(os.sep, '.')
                # This might result in something like src.utils, which is what we want
            except ValueError:
                # The import is outside the repo structure, treat as external
                module_name = module

        else: # Absolute import
            module_name = module

        self.imports.add(module_name)
        for alias in node.names:
            # The full name of the imported symbol is module.name
            full_symbol_name = f"{module_name}.{alias.name}"
            if alias.asname:
                self.aliases[alias.asname] = full_symbol_name
            else:
                # This allows us to resolve `something` to `my.module.something`
                self.aliases[alias.name] = full_symbol_name
        self.generic_visit(node)


def _resolve_import_to_file(imp: str, all_files: list[str], repo_root: Path) -> str | None:
    """
    Tries to resolve an import string to a file path within the repository.
    """
    # imp could be 'src.utils.helpers' or 'src.models'
    # We need to check for 'src/utils/helpers.py' or 'src/models/__init__.py'
    
    # Path variant 1: src/utils/helpers.py
    potential_file_path = imp.replace('.', os.sep) + ".py"
    if potential_file_path in all_files:
        return potential_file_path

    # Path variant 2: src/utils/helpers/__init__.py
    potential_init_path = os.path.join(imp.replace('.', os.sep), "__init__.py")
    if potential_init_path in all_files:
        return potential_init_path
        
    return None


def build_import_graph(repo_path: Path, classified_files_path: Path, output_path: Path, include_tests: bool = False):
    """
    Builds a directed graph of all internal module dependencies in the repository.
    """
    logger.info("Starting import graph extraction...")
    
    try:
        with open(classified_files_path, "r", encoding="utf-8") as f:
            classified_files = json.load(f)
    except FileNotFoundError:
        logger.error(f"Classified files manifest not found at {classified_files_path}")
        return

    graph = nx.DiGraph()
    alias_map = {}

    source_files = [
        f["file"] for f in classified_files 
        if f["type"] == "SOURCE_CODE" and f["language"].lower() == "python"
    ]
    
    if not include_tests:
        source_files = [f for f in source_files if not any(
            test_pattern in f for test_pattern in ["/tests/", "/test/", "test_", "_test.py"]
        )]

    all_repo_files_relative = [str(p.relative_to(repo_path)) for p in repo_path.rglob("*.py")]

    for file_str in source_files:
        file_path = repo_path / file_str
        graph.add_node(file_str, type="file")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=file_str)
            
            visitor = ImportVisitor(file_str, repo_path)
            visitor.visit(tree)
            
            alias_map[file_str] = visitor.aliases
            
            for imp in visitor.imports:
                resolved_file = _resolve_import_to_file(imp, all_repo_files_relative, repo_path)
                if resolved_file and resolved_file != file_str:
                    # This is an internal import
                    graph.add_node(resolved_file, type="file")
                    graph.add_edge(file_str, resolved_file)

        except Exception as e:
            logger.error(f"Could not parse {file_str} for imports: {e}")

    graph_data = nx.node_link_data(graph)
    graph_data["alias_map"] = alias_map

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    logger.info(f"Import graph extraction complete. Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges. Results saved to {output_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dummy_repo = Path("./dummy_repo")
    dummy_repo.mkdir(exist_ok=True)
    
    (dummy_repo / "src").mkdir(exist_ok=True)
    (dummy_repo / "src" / "main.py").write_text("from . import utils\nfrom .models import User")
    (dummy_repo / "src" / "utils.py").write_text("import os\n\nclass Helper:\n    pass")
    (dummy_repo / "src" / "models.py").write_text("class User:\n    pass")
    (dummy_repo / "tests").mkdir(exist_ok=True)
    (dummy_repo / "tests" / "test_main.py").write_text("from src import main")

    dummy_manifest_path = Path("./outputs/dummy_repo/manifest")
    dummy_manifest_path.mkdir(parents=True, exist_ok=True)
    dummy_classified_files = dummy_manifest_path / "classified_files.json"
    dummy_classified_files.write_text(json.dumps([
        {"file": "src/main.py", "type": "SOURCE_CODE", "language": "python", "is_test": False},
        {"file": "src/utils.py", "type": "SOURCE_CODE", "language": "python", "is_test": False},
        {"file": "src/models.py", "type": "SOURCE_CODE", "language": "python", "is_test": False},
        {"file": "tests/test_main.py", "type": "SOURCE_CODE", "language": "python", "is_test": True},
    ]))

    output_dir = Path("./outputs/dummy_repo/analysis")
    build_import_graph(dummy_repo, dummy_classified_files, output_dir / "import_graph.json")

    import shutil
    shutil.rmtree(dummy_repo)
    # shutil.rmtree(Path("./outputs"))
