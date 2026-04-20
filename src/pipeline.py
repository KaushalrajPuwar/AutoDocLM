import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.config import RunConfig
from src.ingest.clone_repo import ingest_repo
from src.ingest.file_filter import filter_files
from src.ingest.classify_files import classify_files
from src.chunking.orchestrator import chunk_repo
from src.analysis.dependency_extractor import extract_dependencies
from src.analysis.entrypoint_detector import detect_entrypoints
from src.analysis.import_graph_builder import build_import_graph
from src.analysis.cross_file_calls import extract_cross_file_calls
from src.analysis.centrality_scorer import score_centrality
from src.analysis.folder_graph_extractor import extract_folder_graphs

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.run_timestamp = datetime.utcnow().isoformat() + "Z"
        self.repo_name = self._determine_repo_name()
        self.project_dir = Path("outputs") / self.repo_name
        self.metadata = {
             "repo_name": self.repo_name,
             "timestamp": self.run_timestamp,
             "config": self.config.model_dump(),
             "commit_hash": None # Populated during step 1
        }
        
    def _determine_repo_name(self) -> str:
        """Extract a usable directory name from the Git URL or zip path."""
        if self.config.repo_url:
            # Example: https://github.com/pallets/flask.git -> flask
            name = self.config.repo_url.strip("/").split("/")[-1]
            if name.endswith(".git"):
                name = name[:-4]
            return name
        elif self.config.repo_zip:
             # Example: ./test_assets/repo.zip -> repo
             return Path(self.config.repo_zip).stem
        else:
             return "unknown_repo"

    def _setup_directories(self):
        """Create necessary outputs structure."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        # Assuming other directories will be created by stages when needed

    def _save_metadata(self):
        """Save the run metadata to metadata.json."""
        metadata_path = self.project_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved run metadata to {metadata_path}")


    def run(self):
        """Execute the full pipeline."""
        logger.info(f"Starting AutoDocLM Orchestrator for repo: {self.repo_name}")
        self._setup_directories()

        # Step 1: Repo Ingestion
        logger.info("=== STEP 1: Repo Ingestion ===")
        raw_repo_path = ingest_repo(self.config, self.metadata, self.project_dir)
        logger.info(f"Repo available at: {raw_repo_path}")

        # Save metadata after step 1 so it includes commit hash
        self._save_metadata()

        logger.info("Pipeline Step 0 and 1 completed successfully.")
        
        # Step 2: File Filtering
        logger.info("=== STEP 2: File Filtering ===")
        manifest_path = filter_files(self.config, raw_repo_path, self.project_dir)
        logger.info(f"File manifest created at: {manifest_path}")
        
        # Step 3: File Classification
        logger.info("=== STEP 3: File Classification ===")
        classified_path = classify_files(self.config, manifest_path, self.project_dir)
        logger.info(f"Classified files mapped at: {classified_path}")
        
        logger.info("Pipeline Step 2 and 3 completed successfully.")

        # Step 4: Chunking
        logger.info("=== STEP 4: Chunking ===")
        chunks_path = chunk_repo(self.config, self.project_dir)
        logger.info(f"Chunks written to: {chunks_path}")

        # Step 5: Static Analysis
        logger.info("=== STEP 5: Static Analysis ===")
        self.run_static_analysis(raw_repo_path, classified_path, chunks_path)

        logger.info("Pipeline Steps 0–5 completed successfully.")
        # Future steps will follow here.

    def run_static_analysis(self, repo_path: Path, classified_files_path: Path, chunks_path: Path):
        """Runs all static analysis modules."""
        analysis_dir = self.project_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # 5.1: Dependency Extraction
        deps_path = analysis_dir / "dependencies.json"
        extract_dependencies(repo_path, deps_path)

        # 5.2: Entrypoint Detection
        entrypoints_path = analysis_dir / "entrypoints.json"
        detect_entrypoints(repo_path, classified_files_path, entrypoints_path)

        # 5.3: Import Graph
        import_graph_path = analysis_dir / "import_graph.json"
        build_import_graph(repo_path, classified_files_path, import_graph_path, self.config.include_tests)

        # 5.4: Cross-File Calls
        cross_calls_path = analysis_dir / "cross_file_calls.json"
        extract_cross_file_calls(repo_path, chunks_path, import_graph_path, cross_calls_path)

        # 5.5: Centrality Scoring
        centrality_path = analysis_dir / "centrality_scores.json"
        score_centrality(import_graph_path, centrality_path)

        # 5.6: Folder Graphs (single bundled JSON, not a directory)
        folder_graphs_path = analysis_dir / "folder_graphs.json"
        extract_folder_graphs(import_graph_path, folder_graphs_path, repo_path)

        logger.info("Static analysis step completed.")
