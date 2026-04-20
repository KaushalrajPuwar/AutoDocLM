"""
Step 7.3 — Folder/Component Inference (with RAG)

For each folder that contains at least one successfully summarised file, synthesizes
the folder's architectural role using:
  - Centrality-ranked file summaries (top-5 full, remainder abbreviated)
  - Folder graph edges (deterministic — incoming/outgoing/internal)
  - Live ChromaDB semantic evidence (RAG — top-5 relevant code chunks)

Switches model to Qwen/Qwen2.5-Coder-32B-Instruct (arch_model).
Output: outputs/<repo>/summaries/modules/<safe_folder_name>.json
"""
import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import RunConfig
from src.llm.prompts import (
    FOLDER_SUMMARY_SYSTEM_PROMPT,
    FOLDER_SUMMARY_USER_PROMPT,
    PROMPT_VERSION,
)
from src.llm.caching import get_cache_key, read_cache, write_cache
from src.llm.inference_client import InferenceClient
from src.llm.rag_retriever import load_collection, retrieve_evidence

logger = logging.getLogger(__name__)

MAX_FULL_FILES = 5  # Files above this count are abbreviated in the prompt


def _safe_folder_name(folder_path: str) -> str:
    """
    Convert a folder path to a flat safe filename.
    e.g. 'src/api' -> 'src__api.json'
    """
    return folder_path.replace("/", "__").replace("\\", "__") + ".json"


def _file_safe_to_folder(safe_name: str) -> str:
    """
    Extract the folder path from a safe filename.
    e.g. 'src__api__routes.json' -> 'src/api'
    """
    # Remove .json extension
    no_ext = safe_name[:-5] if safe_name.endswith(".json") else safe_name
    # Split on double-underscore, drop the last segment (filename)
    parts = no_ext.split("__")
    if len(parts) <= 1:
        return ""  # Top-level file, no folder
    return "/".join(parts[:-1])


def _load_file_summaries(out_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all file summaries from summaries/files/*.json.
    Returns: {file_path: summary_dict}
    """
    files_dir = os.path.join(out_dir, "summaries", "files")
    summaries: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(files_dir):
        logger.warning(f"summaries/files/ directory not found at {files_dir}.")
        return summaries

    for fname in os.listdir(files_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(files_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Skip error markers
            if "error" in data:
                continue
            # Recover the original file path from the summary's own "file" field
            # (more reliable than decoding the filename)
            file_path = data.get("file", "")
            if file_path:
                summaries[file_path] = data
        except Exception as e:
            logger.debug(f"Failed to load file summary {fpath}: {e}")

    return summaries


def _load_centrality_scores(out_dir: str) -> Dict[str, float]:
    """
    Load centrality_scores.json.
    Returns: {file_path: score (0.0–1.0)}
    """
    path = os.path.join(out_dir, "analysis", "centrality_scores.json")
    if not os.path.exists(path):
        logger.warning("centrality_scores.json not found. All files will be treated equally.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_folder_graphs(out_dir: str) -> Dict[str, Any]:
    """
    Load folder_graphs.json.
    Returns: {folder_path: {internal: [...], incoming: [...], outgoing: [...]}}
    """
    path = os.path.join(out_dir, "analysis", "folder_graphs.json")
    if not os.path.exists(path):
        logger.warning("folder_graphs.json not found. Folder graph evidence will be empty.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_import_graph_edges(out_dir: str) -> List[Dict[str, str]]:
    """Load raw edges from import_graph.json."""
    path = os.path.join(out_dir, "analysis", "import_graph.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("edges", [])


def _get_external_callers(
    folder_path: str,
    edges: List[Dict[str, str]],
    file_summaries: Dict[str, Dict[str, Any]]
) -> str:
    """
    Identify external files that import from this folder and return their summaries.
    """
    callers = []
    seen_files = set()
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        # If target starts with folder_path/ (or is folder_path) AND source does not
        is_target_in_folder = tgt == folder_path or tgt.startswith(folder_path + "/")
        is_source_outside = not (src == folder_path or src.startswith(folder_path + "/"))
        
        if is_target_in_folder and is_source_outside and src in file_summaries and src not in seen_files:
            summary = file_summaries[src]
            entry = {
                "file": src,
                "role": summary.get("role", "unknown"),
                "architectural_role": summary.get("architectural_role", "unknown"),
                "public_api_surface": summary.get("public_api_surface", [])
            }
            callers.append(entry)
            seen_files.add(src)
            if len(callers) >= 5: # Cap to 5 external callers to preserve space
                break

    return json.dumps(callers, indent=2)


def _inject_folder_evidence(
    folder_path: str,
    folder_files: List[str],
    centrality_scores: Dict[str, float],
    out_dir: str
) -> str:
    """
    Find top centrality files in folder and inject their top-5 chunk summaries.
    """
    # Coverage rule: < 5 files -> 1, 5-15 files -> 2, > 15 files -> 1
    num_to_inject = 1
    if 5 <= len(folder_files) <= 15:
        num_to_inject = 2

    sorted_files = sorted(folder_files, key=lambda f: centrality_scores.get(f, 0.0), reverse=True)
    targets = sorted_files[:num_to_inject]
    
    evidence_blocks = []
    chunks_dir = os.path.join(out_dir, "summaries", "chunks")
    
    # We need to know which chunks belong to which file
    # This is slightly inefficient but given the scale of Step 7 (parallelized) it's acceptable
    # For a real project we'd pass this in
    chunks_path = os.path.join(out_dir, "chunks", "chunks.jsonl")
    file_to_chunks = defaultdict(list)
    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            for line in f:
                c = json.loads(line)
                if c["file"] in targets:
                    file_to_chunks[c["file"]].append(c["chunk_id"])

    for file_path in targets:
        # Load and sort chunks for this file
        chunk_ids = file_to_chunks.get(file_path, [])
        chunk_summaries = []
        for cid in chunk_ids:
            cp = os.path.join(chunks_dir, f"{cid}.json")
            if os.path.exists(cp):
                with open(cp, "r") as f:
                    chunk_summaries.append(json.load(f))
        
        # Sort chunks by role priority (highest first)
        role_priority = {"application_core": 0, "entry_point":1, "request_handler":2, "infrastructure":3}
        chunk_summaries.sort(key=lambda c: role_priority.get(c.get("role_type", ""), 10))
        
        subset = chunk_summaries[:5]
        if subset:
            block = f"## HIGH-CENTRALITY FILE [centrality score: {centrality_scores.get(file_path, 0.0):.4f}]\n"
            block += f"## Chunk-level evidence for: {file_path}\n"
            block += json.dumps(subset, indent=2)
            evidence_blocks.append(block)

    return "\n\n".join(evidence_blocks)


def _group_files_by_folder(
    file_summaries: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Group file paths by their parent folder.
    Returns: {folder_path: [file_path, ...]}
    """
    folder_map: Dict[str, List[str]] = defaultdict(list)
    for file_path in file_summaries:
        # Resolve folder path
        parts = file_path.replace("\\", "/").split("/")
        if len(parts) > 1:
            folder = "/".join(parts[:-1])
        else:
            folder = ""  # Top-level file
        folder_map[folder].append(file_path)
    return folder_map


def _build_file_summaries_block(
    folder_files: List[str],
    file_summaries: Dict[str, Dict[str, Any]],
    centrality_scores: Dict[str, float],
) -> tuple[str, str]:
    """
    Apply centrality-based selection and relative truncation (bottom 30%).
    """
    # Sort files by centrality score descending
    sorted_files = sorted(
        folder_files,
        key=lambda f: centrality_scores.get(f, 0.0),
        reverse=True,
    )

    # NO truncation: Keep 100% in full to ensure signal fidelity
    full_files = sorted_files
    abbreviated_files = []

    truncation_note = ""
    if abbreviated_files:
        truncation_note = (
            f"NOTE: This folder has {len(folder_files)} files. "
            f"Top {len(full_files)} are shown in full. "
            f"Bottom {len(abbreviated_files)} (30% lowest centrality) are abbreviated (labels only)."
        )

    result_entries = []

    for fp in full_files:
        summary = file_summaries[fp]
        result_entries.append(summary)

    for fp in abbreviated_files:
        summary = file_summaries[fp]
        abbreviated = {
            "file": summary.get("file", fp),
            "role": summary.get("role", "unknown"),
            "architectural_role": summary.get("architectural_role", "unknown"),
            "public_api_surface": summary.get("public_api_surface", []),
            "_note": "[ABBREVIATED — low centrality]",
        }
        result_entries.append(abbreviated)

    return truncation_note, json.dumps(result_entries, indent=2)


async def process_folder(
    folder_path: str,
    folder_files: List[str],
    client: InferenceClient,
    config: RunConfig,
    out_dir: str,
    semaphore: asyncio.Semaphore,
    file_summaries: Dict[str, Dict[str, Any]],
    centrality_scores: Dict[str, float],
    folder_graphs: Dict[str, Any],
    import_edges: List[Dict[str, str]],
    collection: Optional[object],
    index: int,
    total: int
) -> None:
    """
    Process a single folder: build evidence, call LLM, write output.
    """
    model = config.arch_model  # 32B for folder inference
    stage = "folder"
    prompt_version = PROMPT_VERSION.get("folder", "v1")

    out_file = os.path.join(out_dir, "summaries", "modules", _safe_folder_name(folder_path))
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                data = json.load(f)
            if "error" not in data:
                return  # Already done in a previous run
        except Exception:
            pass

    logger.info(f"Processing folder {index}/{total}: {folder_path}")

    # 1. Get valid file summaries for this folder
    valid_summaries = {fp: file_summaries[fp] for fp in folder_files if fp in file_summaries}
    if not valid_summaries:
        logger.warning(f"No valid file summaries for folder '{folder_path}'. Skipping.")
        return

    # 2. Centrality-based file selection
    truncation_note, file_summaries_json = _build_file_summaries_block(
        list(valid_summaries.keys()), file_summaries, centrality_scores
    )

    # 3. Folder graph evidence
    folder_graph = folder_graphs.get(folder_path, {"internal": [], "incoming": [], "outgoing": []})
    folder_graph_json = json.dumps(folder_graph, indent=2)

    # 4. External Callers Evidence (Structural)
    external_callers_json = _get_external_callers(folder_path, import_edges, file_summaries)

    # 5. Raw Evidence Injection (High-Centrality Chunks)
    raw_chunk_evidence = _inject_folder_evidence(folder_path, folder_files, centrality_scores, out_dir)

    # 6. RAG semantic evidence (Cross-Corpus, excluding this folder)
    folder_name = folder_path.split("/")[-1]
    query = f"{folder_name} usage and implementation details"
    exclude_filter = folder_files # Exclude chunks from inside this folder

    rag_evidence_text = await asyncio.to_thread(
        retrieve_evidence,
        collection,
        query,
        None, # file_filter = None (Scan whole corpus)
        exclude_filter,
        3, # Fewer results for less noise
        config.embedding_model,
    )
    
    # Combine raw evidence + RAG
    semantic_evidence_text = f"{raw_chunk_evidence}\n\n## SUPPLEMENTARY RAG EVIDENCE\n{rag_evidence_text}"

    # 7. Format prompt
    try:
        user_prompt = FOLDER_SUMMARY_USER_PROMPT.format(
            folder_path=folder_path,
            folder_graph_json=folder_graph_json,
            external_callers_json=external_callers_json,
            semantic_evidence_text=semantic_evidence_text,
            truncation_note=truncation_note,
            file_summaries_json=file_summaries_json,
        )
    except KeyError as e:
        logger.warning(f"Prompt format error for folder {folder_path}: {e}")
        return

    # 6. Cache check
    c_key = get_cache_key(user_prompt, model, prompt_version)
    cached_data = read_cache(out_dir, stage, c_key)

    if cached_data:
        response_data = cached_data
    else:
        # 7. Call LLM (32B, higher max_tokens for richer synthesis)
        async with semaphore:
            response_data = await client.generate_json_async(
                model=model,
                system_prompt=FOLDER_SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=4096,
                seed=42,
                stage=stage,
            )
        if "error" not in response_data:
            write_cache(out_dir, stage, c_key, response_data)

    # 8. Write output
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2)


async def run_folder_inference_async(config: RunConfig, out_dir: str) -> None:
    # Load all data needed upfront
    file_summaries = _load_file_summaries(out_dir)
    if not file_summaries:
        logger.warning("No file summaries found. Skipping folder inference.")
        return

    centrality_scores = _load_centrality_scores(out_dir)
    folder_graphs = _load_folder_graphs(out_dir)
    import_edges = _load_import_graph_edges(out_dir)
    folder_map = _group_files_by_folder(file_summaries)

    # Load ChromaDB collection once and share across all folder tasks
    collection = load_collection(out_dir)

    client = InferenceClient(config)
    # Use a lower concurrency for folder inference — the 32B model has longer latency
    # and each folder prompt is much bigger. Half the chunk concurrency is a safe default.
    folder_concurrency = max(1, config.inference_concurrency // 2)
    semaphore = asyncio.Semaphore(folder_concurrency)

    logger.info(
        f"Starting folder inference for {len(folder_map)} folders "
        f"with concurrency {folder_concurrency} ({config.arch_model})..."
    )

    valid_folders = [(fp, ff) for fp, ff in folder_map.items() if fp]
    for pass_idx in range(1, 4):
        # 1. Identify which folders need processing
        folders_to_process = []
        for folder_path, folder_files in valid_folders:
            out_file = os.path.join(out_dir, "summaries", "modules", _safe_folder_name(folder_path))
            needs_work = True
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r") as f:
                        data = json.load(f)
                    if "error" not in data:
                        needs_work = False
                except Exception:
                    pass
            if needs_work:
                folders_to_process.append((folder_path, folder_files))

        if not folders_to_process:
            if pass_idx > 1:
                logger.info(f"All folders successfully processed after pass {pass_idx-1}.")
            break

        if pass_idx > 1:
            logger.info(f"Starting pass {pass_idx}/3 to retry {len(folders_to_process)} failed/missing folders...")

        tasks = [
            process_folder(
                folder_path, folder_files, client, config, out_dir,
                semaphore, file_summaries, centrality_scores, folder_graphs, import_edges, collection,
                i + 1, len(folders_to_process)
            )
            for i, (folder_path, folder_files) in enumerate(folders_to_process)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Folder task failed: {r}")
    logger.info("Folder inference completed.")


def run_folder_inference(config: RunConfig, out_dir: str) -> None:
    """Synchronous wrapper called by the orchestrator."""
    asyncio.run(run_folder_inference_async(config, out_dir))
