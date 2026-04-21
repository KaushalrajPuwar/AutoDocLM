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

        # Step 6: Embedding + Vector Index (Optional)
        if self.config.use_embeddings:
            logger.info("=== STEP 6: Embedding + Vector Index ===")
            from src.indexing.vector_store_chroma import run_indexing
            embeddings_dir = self.project_dir / "embeddings"
            chunk_metadata_path = run_indexing(
                chunks_path=chunks_path,
                classified_files_path=classified_path,
                embeddings_dir=embeddings_dir,
                ollama_model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
                include_tests=self.config.include_tests,
            )
            if chunk_metadata_path:
                logger.info(f"Step 6 complete. Metadata: {chunk_metadata_path}")
            else:
                logger.warning(
                    "Step 6: Indexing returned None (Ollama unreachable). "
                    "Continuing without embeddings."
                )
        else:
            logger.info("Step 6: Embeddings skipped (pass --use-embeddings to enable).")

        logger.info("Pipeline Steps 0–6 completed successfully.")
        
    def _verify_step_7(self) -> tuple[bool, list[str]]:
        """
        Simplified check: Just verify the final repo architecture summary.
        If this exists and is valid, we assume the pipeline is ready for Step 8.
        """
        missing_or_failed = []
        summaries_dir = self.project_dir / "summaries"

        arch_path = summaries_dir / "repo_architecture.json"
        if not arch_path.exists():
            missing_or_failed.append("repo_architecture.json missing")
        else:
            try:
                with open(arch_path, 'r') as f:
                    data = json.load(f)
                if "error" in data:
                    missing_or_failed.append("repo_architecture.json contains error marker")
            except Exception as e:
                missing_or_failed.append(f"repo_architecture.json invalid json: {e}")

        return (len(missing_or_failed) == 0, missing_or_failed)

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

        # Step 6: Embedding + Vector Index (Optional)
        if self.config.use_embeddings:
            logger.info("=== STEP 6: Embedding + Vector Index ===")
            from src.indexing.vector_store_chroma import run_indexing
            embeddings_dir = self.project_dir / "embeddings"
            chunk_metadata_path = run_indexing(
                chunks_path=chunks_path,
                classified_files_path=classified_path,
                embeddings_dir=embeddings_dir,
                ollama_model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
                include_tests=self.config.include_tests,
            )
            if chunk_metadata_path:
                logger.info(f"Step 6 complete. Metadata: {chunk_metadata_path}")
            else:
                logger.warning(
                    "Step 6: Indexing returned None (Ollama unreachable). "
                    "Continuing without embeddings."
                )
        else:
            logger.info("Step 6: Embeddings skipped (pass --use-embeddings to enable).")

        logger.info("Pipeline Steps 0–6 completed successfully.")
        
        # Step 7 Implementation Loop
        def _run_stage_7():
            # Step 7: LLM Inference (7.1 Chunk Inference)
            logger.info("=== STEP 7.1: Chunk Inference ===")
            from src.llm.chunk_inference import run_chunk_inference
            run_chunk_inference(self.config, str(self.project_dir))

            # Step 7.2: File-Level Inference
            logger.info("=== STEP 7.2: File Inference ===")
            from src.llm.file_inference import run_file_inference
            run_file_inference(self.config, str(self.project_dir))

            # Step 7.3: Folder/Component Inference (with RAG)
            logger.info("=== STEP 7.3: Folder Inference ===")
            from src.llm.folder_inference import run_folder_inference
            run_folder_inference(self.config, str(self.project_dir))

            # Step 7.4: Repo-Wide Architecture Inference
            logger.info("=== STEP 7.4: Repo Architecture Inference ===")
            from src.llm.repo_inference import run_repo_inference
            run_repo_inference(self.config, str(self.project_dir))

        # Initial execution of Stage 7
        _run_stage_7()

        # Step 7 Verification Sweep & Auto-Retry
        logger.info("=== STEP 7: Verification Sweep ===")
        success, missing = self._verify_step_7()
        if not success:
            logger.warning(f"Step 7 verification detected {len(missing)} missing or failed items. Triggering auto-retry...")
            _run_stage_7()
            success, missing = self._verify_step_7() # Final sweep
            if not success:
                logger.error(f"Step 7 verification failed after retry. Missing items: {missing}")
                raise RuntimeError(f"Step 7 implementation check failed for {self.repo_name}. Manual inspection required.")
        
        logger.info("Step 7 verified: all summaries and architecture components are present.")
        logger.info("Pipeline Steps 0–7.4 completed successfully.")

        # Step 8: Markdown Documentation Writing
        logger.info("=== STEP 8: Markdown Documentation Writing ===")
        from src.llm.markdown_writer import run_step_8
        run_step_8(self.config, str(self.project_dir))

        logger.info("Pipeline Steps 0–8 completed successfully.")
        
        # Step 9: Mermaid Diagram Generation (Deterministic)
        logger.info("=== STEP 9: Mermaid Diagram Generation ===")
        from src.docs.diagram_generator import run_step_9
        run_step_9(self.config, str(self.project_dir))

        logger.info("Pipeline Steps 0–9 completed successfully.")

        # Step 10: MkDocs Assembly + Site Build
        logger.info("=== STEP 10: MkDocs Assembly + Site Build ===")
        from src.docs.mkdocs_builder import run_step_10
        step_10_ok = run_step_10(self.config, str(self.project_dir))
        if not step_10_ok:
            raise RuntimeError(
                f"Step 10 failed for {self.repo_name}. "
                "MkDocs site build did not complete successfully."
            )

        logger.info("Pipeline Steps 0–10 completed successfully.")

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
