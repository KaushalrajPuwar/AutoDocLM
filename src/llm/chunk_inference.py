import asyncio
import json
import logging
import os
from typing import List, Dict, Any

from src.config import RunConfig
from src.llm.prompts import CHUNK_SUMMARY_SYSTEM_PROMPT, CHUNK_SUMMARY_USER_PROMPT, PROMPT_VERSION
from src.llm.caching import get_cache_key, read_cache, write_cache
from src.llm.inference_client import InferenceClient

logger = logging.getLogger(__name__)

async def process_chunk(
    chunk: Dict[str, Any],
    client: InferenceClient,
    config: RunConfig,
    out_dir: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int
) -> None:
    """
    Process a single chunk. Check cache first, then call LLM if missing, and output to designated directory.
    """
    chunk_id = chunk["chunk_id"]
    model = config.chunk_model
    stage = "chunk"
    prompt_version = PROMPT_VERSION.get("chunk", "v1")
    out_file = os.path.join(out_dir, "summaries", "chunks", f"{chunk_id}.json")
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                data = json.load(f)
            if "error" not in data:
                # Already fully completed and written to disk from a previous run
                return
        except Exception:
            pass

    logger.info(f"Processing chunk {index}/{total}: {chunk_id}")

    # 1. Calculate public hint (heuristic based on leading underscore)
    symbol_name = chunk.get("symbol", chunk.get("name", "unknown"))
    is_public_hint = not symbol_name.startswith("_")

    # 1. Format User Prompt
    try:
        user_prompt = CHUNK_SUMMARY_USER_PROMPT.format(
            chunk_id=chunk_id,
            file=chunk["file"],
            symbol=symbol_name,
            parent_class=chunk.get("parent_class", "null"),
            decorators=json.dumps(chunk.get("decorators", [])),
            chunk_type=chunk.get("chunk_type", "unknown"),
            language="python", # Hardcoding for now, could be passed dynamically
            start_line=chunk.get("line_start", 0),
            end_line=chunk.get("line_end", 0),
            is_public_hint=is_public_hint,
            code=chunk["chunk_text"]
        )
    except KeyError as e:
        logger.warning(f"KeyError formatting prompt for chunk_id {chunk_id}: {e}")
        return

    # 2. Check Cache
    c_key = get_cache_key(user_prompt, model, prompt_version)
    cached_data = read_cache(out_dir, stage, c_key)
    
    if cached_data:
        response_data = cached_data
    else:
        # 3. Call LLM with concurrency control
        async with semaphore:
            response_data = await client.generate_json_async(
                model=model,
                system_prompt=CHUNK_SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=1024,
                seed=42, # Fixed seed for reproducible output
                stage=stage
            )
        
        # Write to cache
        if "error" not in response_data:
            write_cache(out_dir, stage, c_key, response_data)

    # 4. Write final output file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2)


async def run_chunk_inference_async(config: RunConfig, repo_out_dir: str):
    """
    Asynchronous executor for all chunks in the repository.
    """
    chunks_path = os.path.join(repo_out_dir, "chunks", "chunks.jsonl")
    if not os.path.exists(chunks_path):
        logger.warning(f"chunks.jsonl not found at {chunks_path}. Skipping chunk inference.")
        return

    # Load chunks
    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
    if not chunks:
        logger.info("No chunks to process for LLM inference.")
        return

    client = InferenceClient(config)
    semaphore = asyncio.Semaphore(config.inference_concurrency)
    
    logger.info(f"Starting chunk inference for {len(chunks)} chunks with concurrency {config.inference_concurrency} ({config.chunk_model})...")
    
    for pass_idx in range(1, 4):
        # 1. Identify which chunks need processing (missing or error)
        chunks_to_process = []
        for chunk in chunks:
            out_file = os.path.join(repo_out_dir, "summaries", "chunks", f"{chunk['chunk_id']}.json")
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
                chunks_to_process.append(chunk)

        if not chunks_to_process:
            if pass_idx > 1:
                logger.info(f"All chunks successfully processed after pass {pass_idx-1}.")
            break

        if pass_idx > 1:
            logger.info(f"Starting pass {pass_idx}/3 to retry {len(chunks_to_process)} failed/missing chunks...")

        tasks = []
        for i, chunk in enumerate(chunks_to_process, 1):
            tasks.append(process_chunk(chunk, client, config, repo_out_dir, semaphore, i, len(chunks_to_process)))
            
        # Execute batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors that occurred
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Chunk task failed: {r}")
    
    logger.info("Chunk inference completed.")

def run_chunk_inference(config: RunConfig, repo_out_dir: str):
    """
    Synchronous wrapper for orchestrator to call.
    """
    asyncio.run(run_chunk_inference_async(config, repo_out_dir))
