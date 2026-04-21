"""
Step 7.2 — File-Level Semantic Inference

For each unique source file in chunks.jsonl, collects all of its chunk summaries,
augments with static analysis evidence (imports + cross-file calls), and calls the
7B model to produce a file-level semantic summary JSON.

Output: outputs/<repo>/summaries/files/<safe_name>.json
"""
import asyncio
import json
import logging
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional

from src.config import RunConfig
from src.llm.prompts import (
    FILE_SUMMARY_SYSTEM_PROMPT,
    FILE_SUMMARY_USER_PROMPT,
    PROMPT_VERSION,
)
from src.llm.caching import get_cache_key, read_cache, write_cache
from src.llm.inference_client import InferenceClient

logger = logging.getLogger(__name__)


def _safe_filename(file_path: str) -> str:
    """
    Convert a file path to a safe flat filename.
    e.g. 'src/api/routes.py' -> 'src__api__routes.json'
    """
    return file_path.replace("/", "__").replace("\\", "__").rsplit(".", 1)[0] + ".json"


def _load_chunk_to_file_map(out_dir: str) -> Dict[str, List[str]]:
    """
    Read chunks.jsonl and build a map: {file_path: [chunk_id, ...]}
    """
    chunks_path = os.path.join(out_dir, "chunks", "chunks.jsonl")
    file_to_chunks: Dict[str, List[str]] = defaultdict(list)

    if not os.path.exists(chunks_path):
        logger.warning(f"chunks.jsonl not found at {chunks_path}.")
        return file_to_chunks

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                file_path = chunk.get("file", "")
                chunk_id = chunk.get("chunk_id", "")
                if file_path and chunk_id:
                    file_to_chunks[file_path].append(chunk_id)
            except json.JSONDecodeError:
                continue

    return file_to_chunks


def _load_import_graph(out_dir: str) -> Dict[str, List[str]]:
    """
    Load import_graph.json and return a map: {file_path: [imported_modules]}
    """
    path = os.path.join(out_dir, "analysis", "import_graph.json")
    if not os.path.exists(path):
        logger.warning("import_graph.json not found. Proceeding without import evidence.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # Build file -> imports list from edges: [{source, target}, ...]
    imports_map: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.get("edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt:
            imports_map[src].append(tgt)

    return imports_map


def _load_cross_file_calls(out_dir: str) -> Dict[str, Any]:
    """
    Load cross_file_calls.json and return the full dict.
    Structure: {calling_file: {calling_function: {calls: [{file, function}]}}}
    """
    path = os.path.join(out_dir, "analysis", "cross_file_calls.json")
    if not os.path.exists(path):
        logger.warning("cross_file_calls.json not found. Proceeding without cross-call evidence.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rank_chunk_summaries(chunk_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort chunks by architectural importance based on role_type and chunk_type.
    Order: core -> entry -> handler -> infra -> interface -> model -> config -> utility -> test
    """
    role_priority = {
        "application_core": 0,
        "entry_point": 1,
        "request_handler": 2,
        "infrastructure": 3,
        "interface_or_mixin": 4,
        "data_model": 5,
        "configuration": 6,
        "utility": 7,
        "test": 8,
        "unknown": 9,
    }
    type_priority = {
        "class": 0,
        "function": 1,
        "method": 1,
        "block": 2,
        "unknown": 3,
    }

    def sort_key(c):
        rp = role_priority.get(c.get("role_type", "unknown"), 9)
        tp = type_priority.get(c.get("chunk_type", "unknown"), 3)
        return (rp, tp, c.get("symbol", ""))

    return sorted(chunk_summaries, key=sort_key)


async def process_file(
    file_path: str,
    chunk_ids: List[str],
    client: InferenceClient,
    config: RunConfig,
    out_dir: str,
    semaphore: asyncio.Semaphore,
    imports_map: Dict[str, List[str]],
    cross_file_calls: Dict[str, Any],
    index: int,
    total: int
) -> None:
    """
    Process a single file: collect chunk summaries, build prompt, call LLM, write output.
    """
    model = config.chunk_model  # 7B for file inference
    stage = "file"
    prompt_version = PROMPT_VERSION.get("file", "v1")

    out_file = os.path.join(out_dir, "summaries", "files", _safe_filename(file_path))
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                data = json.load(f)
            if "error" not in data:
                return  # Already done in a previous run
        except Exception:
            pass

    # 1. Load chunk summaries for this file
    chunk_summaries = []
    chunks_dir = os.path.join(out_dir, "summaries", "chunks")
    for chunk_id in chunk_ids:
        chunk_path = os.path.join(chunks_dir, f"{chunk_id}.json")
        if not os.path.exists(chunk_path):
            logger.debug(f"Chunk summary not found: {chunk_path}")
            continue
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Skip error chunks — they have no useful content
            if "error" not in data:
                chunk_summaries.append(data)
        except Exception as e:
            logger.debug(f"Failed to read chunk summary {chunk_path}: {e}")

    if not chunk_summaries:
        logger.warning(f"No valid chunk summaries for {file_path}. Writing error marker.")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"error": "all_chunks_failed", "file": file_path}, f, indent=2)
        return

    # 2. Rank chunk summaries by role importance
    ranked_summaries = _rank_chunk_summaries(chunk_summaries)

    # 3. Build imports_list string
    imports = imports_map.get(file_path, [])
    imports_list = json.dumps(imports) if imports else "[]"

    # 4. Extract cross-file call evidence for this file
    file_cross_calls = cross_file_calls.get(file_path, {})
    cross_file_calls_json = json.dumps(file_cross_calls, indent=2) if file_cross_calls else "{}"

    # 5. Format prompt
    chunk_summaries_json = json.dumps(ranked_summaries, indent=2)
    try:
        user_prompt = FILE_SUMMARY_USER_PROMPT.format(
            file=file_path,
            imports_list=imports_list,
            cross_file_calls_json=cross_file_calls_json,
            chunk_summaries_json=chunk_summaries_json,
        )
    except KeyError as e:
        logger.warning(f"Prompt format error for file {file_path}: {e}")
        return

    # 6. Cache check
    c_key = get_cache_key(user_prompt, model, prompt_version)
    cached_data = read_cache(out_dir, stage, c_key)

    if cached_data:
        response_data = cached_data
    else:
        # 7. Call LLM
        async with semaphore:
            logger.info(f"Processing file {index}/{total}: {file_path}")
            response_data = await client.generate_json_async(
                model=model,
                system_prompt=FILE_SUMMARY_SYSTEM_PROMPT,
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
    logger.info(f"Done file {index}/{total}: {file_path}")


async def run_file_inference_async(config: RunConfig, out_dir: str) -> None:
    file_to_chunks = _load_chunk_to_file_map(out_dir)
    if not file_to_chunks:
        logger.warning("No file-to-chunk mapping found. Skipping file inference.")
        return

    imports_map = _load_import_graph(out_dir)
    cross_file_calls = _load_cross_file_calls(out_dir)

    client = InferenceClient(config)
    semaphore = asyncio.Semaphore(config.inference_concurrency)

    logger.info(
        f"Starting file inference for {len(file_to_chunks)} files "
        f"with concurrency {config.inference_concurrency} ({config.chunk_model})..."
    )

    for pass_idx in range(1, 4):
        # 1. Identify which files need processing
        files_to_process = []
        for i, (file_path, chunk_ids) in enumerate(file_to_chunks.items()):
            out_file = os.path.join(out_dir, "summaries", "files", _safe_filename(file_path))
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
                files_to_process.append((file_path, chunk_ids))

        if not files_to_process:
            if pass_idx > 1:
                logger.info(f"All files successfully processed after pass {pass_idx-1}.")
            break

        if pass_idx > 1:
            logger.info(f"Starting pass {pass_idx}/3 to retry {len(files_to_process)} failed/missing files...")

        tasks = [
            process_file(
                file_path, chunk_ids, client, config, out_dir,
                semaphore, imports_map, cross_file_calls, i + 1, len(files_to_process)
            )
            for i, (file_path, chunk_ids) in enumerate(files_to_process)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"File task failed: {r}")
    logger.info("File inference completed.")
    await client.aclose()


def run_file_inference(config: RunConfig, out_dir: str) -> None:
    """Synchronous wrapper called by the orchestrator."""
    asyncio.run(run_file_inference_async(config, out_dir))
