import logging
import subprocess
from pathlib import Path
from .config_generator import generate_mkdocs_config

logger = logging.getLogger(__name__)

def run_site_build(project_dir: Path, repo_name: str):
    logger.info("=== STEP 10: MkDocs Assembly and Site Build ===")
    
    # 1. Generate Config
    mkdocs_file = generate_mkdocs_config(project_dir, repo_name)
    logger.info(f"Generated MkDocs Configuration at {mkdocs_file}")
    
    # 2. Run Site Build
    try:
        import sys
        logger.info("Running `mkdocs build`...")
        result = subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "-f", str(mkdocs_file.absolute())],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(result.stdout)
        logger.info(f"Site built successfully at {project_dir / 'site'}")
    except subprocess.CalledProcessError as e:
        logger.error(f"MkDocs build failed with exit code: {e.returncode}")
        logger.error(f"Stdout:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        raise RuntimeError("Step 10: MkDocs Site Assembly failed. Check mkdocs dependency.")
    
    logger.info("Site Assembly Complete.")
