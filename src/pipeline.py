import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.config import RunConfig
from src.ingest.clone_repo import ingest_repo
from src.ingest.file_filter import filter_files
from src.ingest.classify_files import classify_files

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
        # Future steps will follow here.
