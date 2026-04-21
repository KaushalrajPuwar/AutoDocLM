import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from src.config import RunConfig
from src.llm.inference_client import InferenceClient
from src.llm.caching import get_cache_key, read_cache, write_cache
from src.llm.prompts import (
    PROMPT_VERSION,
    WRITING_SYSTEM_PROMPT_HEADER,
    FILE_WRITE_USER_PROMPT,
    FOLDER_WRITE_USER_PROMPT,
    ARCH_WRITE_USER_PROMPT,
    SETUP_WRITE_USER_PROMPT,
    INDEX_WRITE_USER_PROMPT,
    REFERENCE_WRITE_USER_PROMPT
)

logger = logging.getLogger(__name__)

class MarkdownWriter:
    def __init__(self, config: RunConfig, project_dir: str):
        self.config = config
        self.project_dir = Path(project_dir)
        self.docs_dir = self.project_dir / "docs"
        self.client = InferenceClient(config)
        self.semaphore = asyncio.Semaphore(config.writing_concurrency)

    async def run(self):
        """Orchestrate the documentation writing process."""
        logger.info("Starting Step 8: Markdown Documentation Writing")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        (self.docs_dir / "files").mkdir(exist_ok=True)
        (self.docs_dir / "modules").mkdir(exist_ok=True)

        tasks = []

        # 8.1: File Docs
        tasks.append(self.generate_file_pages())

        # 8.2: Folder Docs
        tasks.append(self.generate_folder_pages())

        # 8.3: Architecture Overview
        tasks.append(self.generate_architecture_page())

        # 8.4: Setup / Installation
        tasks.append(self.generate_setup_page())

        await asyncio.gather(*tasks)

        # 8.5: Reference Page (Site Map)
        await self.generate_reference_page()

        # 8.6: Homepage / Index (Depends on knowing which pages were generated)
        await self.generate_index_page()

        logger.info(f"Step 8 complete. Documentation generated at: {self.docs_dir}")

    async def _write_prose(self, stage: str, prompt_version: str, user_prompt: str, cache_input: str) -> str:
        """Helper to manage caching and LLM calls for prose generation."""
        cache_key = get_cache_key(cache_input, self.config.writing_model, prompt_version)
        cached = read_cache(str(self.project_dir), stage, cache_key)
        
        if cached and "markdown" in cached:
            return cached["markdown"]

        async with self.semaphore:
            content = await self.client.generate_markdown_async(
                model=self.config.writing_model,
                system_prompt=WRITING_SYSTEM_PROMPT_HEADER,
                user_prompt=user_prompt,
                stage=stage
            )
            
            # Post-process for "unknown" safety
            if "unknown" in content.lower() and "not confirmed in repository" not in content.lower():
                # The prompt rule should handle this, but we can do a best-effort replacement if needed.
                pass

            write_cache(str(self.project_dir), stage, cache_key, {"markdown": content})
            return content

    def _get_safe_filename(self, file_path: str) -> str:
        """Mirroring Step 7 file summary naming convention."""
        return file_path.replace("/", "__").replace("\\", "__").rsplit(".", 1)[0] + ".md"

    def _get_safe_folder_name(self, folder_path: str) -> str:
        """Mirroring Step 7 folder summary naming convention."""
        return folder_path.replace("/", "__").replace("\\", "__") + ".md"

    async def generate_file_pages(self):
        """Generate documentation for every file summary found."""
        summaries_dir = self.project_dir / "summaries" / "files"
        if not summaries_dir.exists():
            logger.warning("No file summaries found. Skipping file docs.")
            return

        tasks = []
        for summary_file in summaries_dir.glob("*.json"):
            tasks.append(self._generate_single_file_page(summary_file))
        await asyncio.gather(*tasks)

    async def _generate_single_file_page(self, summary_path: Path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_json = json.load(f)
        
        file_path = summary_json.get("file", summary_path.stem)
        logger.info(f"Generating file doc: {file_path}")
        
        user_prompt = FILE_WRITE_USER_PROMPT.format(
            file_summary_json=json.dumps(summary_json, indent=2)
        )
        
        content = await self._write_prose(
            stage="write_file",
            prompt_version=PROMPT_VERSION["write_file"],
            user_prompt=user_prompt,
            cache_input=json.dumps(summary_json)
        )
        
        safe_name = self._get_safe_filename(file_path)
        out_path = self.docs_dir / "files" / safe_name
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

    async def generate_folder_pages(self):
        """Generate documentation for every module summary found."""
        summaries_dir = self.project_dir / "summaries" / "modules"
        if not summaries_dir.exists():
            logger.warning("No module summaries found. Skipping folder docs.")
            return

        tasks = []
        for summary_file in summaries_dir.glob("*.json"):
            tasks.append(self._generate_single_folder_page(summary_file))
        await asyncio.gather(*tasks)

    async def _generate_single_folder_page(self, summary_path: Path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_json = json.load(f)
        
        folder_path = summary_json.get("folder", summary_path.stem)
        logger.info(f"Generating module doc: {folder_path}")

        # GATHER CHILD FILE SUMMARIES
        child_summaries = []
        file_sum_dir = self.project_dir / "summaries" / "files"
        if file_sum_dir.exists():
            for f_sum in file_sum_dir.glob("*.json"):
                try:
                    with open(f_sum, 'r', encoding='utf-8') as fs:
                        f_data = json.load(fs)
                        # Check if file belongs to this folder
                        f_path = f_data.get("file", "")
                        if f_path.startswith(folder_path + "/") or f_path == folder_path:
                            # Keep it condensed: Role and APIs are enough for synergy mapping
                            child_summaries.append({
                                "file": f_path,
                                "role": f_data.get("role", "unknown"),
                                "public_api": f_data.get("public_api_surface", [])
                            })
                except Exception as e:
                    logger.debug(f"Error reading child summary {f_sum}: {e}")

        user_prompt = FOLDER_WRITE_USER_PROMPT.format(
            module_summary_json=json.dumps(summary_json, indent=2),
            file_summaries_json=json.dumps(child_summaries, indent=2)
        )
        
        content = await self._write_prose(
            stage="write_folder",
            prompt_version=PROMPT_VERSION["write_folder"],
            user_prompt=user_prompt,
            cache_input=json.dumps(summary_json) + json.dumps(child_summaries)
        )
        
        safe_name = self._get_safe_folder_name(folder_path)
        out_path = self.docs_dir / "modules" / safe_name
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

    async def generate_architecture_page(self):
        """Generate the main architecture overview page."""
        logger.info("Generating architecture doc: architecture.md")
        summary_path = self.project_dir / "summaries" / "repo_architecture.json"
        if not summary_path.exists():
            logger.warning("Repo architecture summary not found. Skipping architecture page.")
            return

        with open(summary_path, 'r', encoding='utf-8') as f:
            arch_json = json.load(f)

        # GATHER TOP-CENTRAlITY MODULES
        top_modules = []
        try:
            centrality_path = self.project_dir / "analysis" / "centrality_scores.json"
            if centrality_path.exists():
                with open(centrality_path, 'r', encoding='utf-8') as f:
                    scores = json.load(f)
                # scores is traditionally a list or dict of {file: score}
                # But Step 7.3 folder inference already worked on them.
                # Let's just get the top 10 from summaries/modules
                mod_dir = self.project_dir / "summaries" / "modules"
                if mod_dir.exists():
                    # We'll just take the top 10 by file size or name for now if scores are hard to parse
                    # Actually, let's just take all module summaries if there are < 15, or top 10.
                    all_mods = list(mod_dir.glob("*.json"))
                    all_mods.sort(key=lambda x: os.path.getsize(x), reverse=True)
                    for m_path in all_mods[:10]:
                        with open(m_path, 'r', encoding='utf-8') as mf:
                            top_modules.append(json.load(mf))
        except Exception as e:
            logger.debug(f"Error gathering top modules: {e}")

        user_prompt = ARCH_WRITE_USER_PROMPT.format(
            repo_architecture_json=json.dumps(arch_json, indent=2),
            top_module_summaries_json=json.dumps(top_modules, indent=2),
            architecture_style=arch_json.get("architecture_style", "unknown")
        )
        
        content = await self._write_prose(
            stage="write_arch",
            prompt_version=PROMPT_VERSION["write_arch"],
            user_prompt=user_prompt,
            cache_input=json.dumps(arch_json) + json.dumps(top_modules)
        )
        
        with open(self.docs_dir / "architecture.md", 'w', encoding='utf-8') as f:
            f.write(content)

    async def generate_setup_page(self):
        """Generate setup and installation documentation."""
        logger.info("Generating setup doc: setup.md")
        # Evidence Gathering
        analysis_dir = self.project_dir / "analysis"
        
        entrypoints = {}
        if (analysis_dir / "entrypoints.json").exists():
            with open(analysis_dir / "entrypoints.json", 'r', encoding='utf-8') as f:
                entrypoints = json.load(f)

        deps = {}
        if (analysis_dir / "dependencies.json").exists():
            with open(analysis_dir / "dependencies.json", 'r', encoding='utf-8') as f:
                deps = json.load(f)

        # Raw evidence from repo
        raw_repo = self.project_dir / "raw_repo"
        readme_text = ""
        build_file_text = ""

        # Priority search for README
        for name in ["README.md", "README", "readme.md"]:
            p = raw_repo / name
            if p.exists():
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    readme_text = f.read(4000) # Read first 4K
                break
        
        # Priority search for build files
        for name in ["Dockerfile", "Makefile", "docker-compose.yml"]:
            p = raw_repo / name
            if p.exists():
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    build_file_text += f"\n--- {name} ---\n" + f.read(2000)

        user_prompt = SETUP_WRITE_USER_PROMPT.format(
            entrypoints_json=json.dumps(entrypoints, indent=2),
            dependencies_json=json.dumps(deps, indent=2),
            readme_text=readme_text if readme_text else "None found.",
            build_file_text=build_file_text if build_file_text else "None found."
        )
        
        content = await self._write_prose(
            stage="write_setup",
            prompt_version=PROMPT_VERSION["write_setup"],
            user_prompt=user_prompt,
            cache_input=user_prompt
        )
        
        with open(self.docs_dir / "setup.md", 'w', encoding='utf-8') as f:
            f.write(content)

    async def generate_reference_page(self):
        """Generate a complete technical reference index (site map) with grounded roles."""
        logger.info("Generating reference doc: reference.md")
        
        doc_pages = []
        
        # 1. Gather Modules with Roles
        mod_summ_dir = self.project_dir / "summaries" / "modules"
        mod_docs_dir = self.docs_dir / "modules"
        if mod_docs_dir.exists():
            for p in sorted(mod_docs_dir.glob("*.md")):
                # Map back to summary to get the role
                summary_stem = p.stem # e.g. src__flask
                summary_path = mod_summ_dir / f"{summary_stem}.json"
                role = "unknown"
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            role = data.get("component_role", "unknown")
                    except: pass
                
                name = p.stem.replace("__", "/")
                doc_pages.append(f"Module: {name} (modules/{p.name}) - Role: {role}")

        # 2. Gather Files with Roles
        file_summ_dir = self.project_dir / "summaries" / "files"
        file_docs_dir = self.docs_dir / "files"
        if file_docs_dir.exists():
            for p in sorted(file_docs_dir.glob("*.md")):
                summary_stem = p.stem
                summary_path = file_summ_dir / f"{summary_stem}.json"
                role = "unknown"
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            role = data.get("role", "unknown")
                    except: pass
                
                name = p.stem.replace("__", "/")
                doc_pages.append(f"File: {name} (files/{p.name}) - Role: {role}")

        user_prompt = REFERENCE_WRITE_USER_PROMPT.format(
            doc_pages_list="\n".join(doc_pages)
        )
        
        # Use a higher max_tokens for the reference page to avoid truncation
        content = await self.client.generate_markdown_async(
            model=self.config.writing_model,
            system_prompt=WRITING_SYSTEM_PROMPT_HEADER,
            user_prompt=user_prompt,
            max_tokens=4096, # Explicitly high for the full index
            stage="write_reference"
        )
        
        # Cache management (Manual since we bypassed _write_prose for max_tokens)
        cache_key = get_cache_key("\n".join(doc_pages), self.config.writing_model, PROMPT_VERSION["write_reference"])
        write_cache(str(self.project_dir), "write_reference", cache_key, {"markdown": content})
        
        with open(self.docs_dir / "reference.md", 'w', encoding='utf-8') as f:
            f.write(content)

    async def generate_index_page(self):
        """Generate the landing page for the site."""
        logger.info("Generating index doc: index.md")
        arch_path = self.project_dir / "summaries" / "repo_architecture.json"
        arch_json = {}
        if arch_path.exists():
            with open(arch_path, 'r', encoding='utf-8') as f:
                arch_json = json.load(f)

        # Build list of available pages
        doc_pages = []
        if (self.docs_dir / "architecture.md").exists():
            doc_pages.append("Architecture Overview: architecture.md")
        if (self.docs_dir / "setup.md").exists():
            doc_pages.append("Setup and Installation: setup.md")
        
        # Add modules links (using flat names)
        mod_dir = self.docs_dir / "modules"
        if mod_dir.exists():
            for p in mod_dir.glob("*.md"):
                doc_pages.append(f"Module: modules/{p.name}")

        user_prompt = INDEX_WRITE_USER_PROMPT.format(
            repo_architecture_json=json.dumps(arch_json, indent=2)
        )
        
        content = await self._write_prose(
            stage="write_index",
            prompt_version=PROMPT_VERSION["write_index"],
            user_prompt=user_prompt,
            cache_input=json.dumps(arch_json)
        )
        
        with open(self.docs_dir / "index.md", 'w', encoding='utf-8') as f:
            f.write(content)

def run_step_8(config: RunConfig, project_dir: str):
    """Entry point for Step 8 (Synchronous wrapper)."""
    async def _run():
        writer = MarkdownWriter(config, project_dir)
        await writer.run()
    
    asyncio.run(_run())
