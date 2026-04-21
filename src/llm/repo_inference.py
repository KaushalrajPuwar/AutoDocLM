"""
Step 7.4 — Repo-Wide Architecture Inference

Single-call inference that assembles a structured 6-block prompt from all
Step 7.3 folder summaries, entrypoints, dependencies, folder-level import
graph, and a classified repo tree, then calls the 32B model once to produce
a unified architectural JSON for the whole repository.

Output: outputs/<repo>/summaries/repo_architecture.json
"""
import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import RunConfig
from src.llm.prompts import REPO_ARCH_SYSTEM_PROMPT, REPO_ARCH_USER_PROMPT, PROMPT_VERSION
from src.llm.caching import get_cache_key, read_cache, write_cache
from src.llm.inference_client import InferenceClient
from src.llm.rag_retriever import load_collection, retrieve_evidence
import math

logger = logging.getLogger(__name__)

# Top 40% of folders by avg centrality get the "high centrality" slot in the prompt
_HIGH_CENTRALITY_FRACTION = 0.40


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_json(path: str, label: str) -> Any:
    """Load a JSON file, returning None and logging a warning if missing."""
    if not os.path.exists(path):
        logger.warning(f"{label} not found at {path}. Injecting empty placeholder.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_module_summaries(out_dir: str) -> List[Dict[str, Any]]:
    """
    Load all folder/module summaries from summaries/modules/*.json.
    Error markers are filtered out.
    """
    modules_dir = os.path.join(out_dir, "summaries", "modules")
    summaries = []

    if not os.path.exists(modules_dir):
        logger.warning(f"summaries/modules/ not found at {modules_dir}.")
        return summaries

    for fname in os.listdir(modules_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(modules_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "error" not in data:
                summaries.append(data)
        except Exception as e:
            logger.debug(f"Failed to load module summary {fpath}: {e}")

    return summaries


def _compute_folder_centrality(
    module_summaries: List[Dict[str, Any]],
    centrality_scores: Dict[str, float],
) -> List[Tuple[Dict[str, Any], float]]:
    """
    For each folder summary, compute average centrality of its files using
    centrality_scores.json. Returns list of (summary, avg_score) sorted
    by score descending.
    """
    # Build a file -> folder map from module summaries' key_files field
    result = []
    for summary in module_summaries:
        folder = summary.get("folder", "")
        key_files = summary.get("key_files", {})

        # key_files is {file_path: role_string}
        file_paths = list(key_files.keys()) if key_files else []

        if file_paths:
            scores = [centrality_scores.get(fp, 0.0) for fp in file_paths]
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0.0

        result.append((summary, avg_score))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _split_module_summaries(
    module_summaries: List[Dict[str, Any]],
    centrality_scores: Dict[str, float],
) -> Tuple[str, str]:
    """
    Split folder summaries into two prompt blocks:
    - Top 40% by avg centrality → high-centrality block
    - Remaining 60% → remaining block

    Returns (high_centrality_json, remaining_json).
    """
    if not module_summaries:
        placeholder = "(No module summaries available.)"
        return placeholder, placeholder

    ranked = _compute_folder_centrality(module_summaries, centrality_scores)
    n_total = len(ranked)
    n_high = max(1, round(n_total * _HIGH_CENTRALITY_FRACTION))

    high = [item[0] for item in ranked[:n_high]]
    remaining = [item[0] for item in ranked[n_high:]]

    high_json = json.dumps(high, indent=2)
    remaining_json = (
        json.dumps(remaining, indent=2)
        if remaining
        else "(No remaining modules — all included in high-centrality set.)"
    )

    logger.info(
        f"Module split: {n_high} high-centrality (top 40%) + "
        f"{len(remaining)} remaining out of {n_total} total folders."
    )
    return high_json, remaining_json


def _compress_to_folder_edges(import_graph: Optional[Dict]) -> str:
    """
    Compress file-level import_graph edges to unique folder-level edges.
    e.g. src/api/routes.py → src/services/user_service.py
    becomes: src/api → src/services
    """
    if not import_graph:
        return "[]"

    folder_edges = set()
    for edge in import_graph.get("edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        src_folder = "/".join(src.replace("\\", "/").split("/")[:-1])
        tgt_folder = "/".join(tgt.replace("\\", "/").split("/")[:-1])
        if src_folder and tgt_folder and src_folder != tgt_folder:
            folder_edges.add((src_folder, tgt_folder))

    edges_list = [{"from": s, "to": t} for s, t in sorted(folder_edges)]
    logger.info(f"Compressed import graph: {len(import_graph.get('edges', []))} file edges → {len(edges_list)} folder edges.")
    return json.dumps(edges_list, indent=2)


def _build_repo_tree(out_dir: str) -> str:
    """Built a text tree representation (omitted for brevity in this replace)"""
    manifest_path = os.path.join(out_dir, "manifest", "classified_files.json")
    if not os.path.exists(manifest_path):
        return "(Repo tree unavailable)"

    with open(manifest_path, "r", encoding="utf-8") as f:
        classified = json.load(f)

    file_paths = sorted(classified.keys())
    tree_lines = ["(Analysed files only)", ""]
    dir_map: Dict[str, List[str]] = defaultdict(list)
    top_level_files = []

    for fp in file_paths:
        parts = fp.replace("\\", "/").split("/")
        if len(parts) == 1:
            top_level_files.append(fp)
        else:
            dir_map[parts[0]].append(fp)

    for f in top_level_files:
        tree_lines.append(f"  {f}")
    for top_dir, paths in sorted(dir_map.items()):
        tree_lines.append(f"  {top_dir}/")
        for fp in paths:
            parts = fp.replace("\\", "/").split("/")
            indent = "  " * len(parts)
            tree_lines.append(f"{indent}{parts[-1]}")
    return "\n".join(tree_lines)


def _get_top_files_json(out_dir: str, centrality_scores: Dict[str, float]) -> str:
    """
    Identify top 3 non-test files via centrality * log(chunk_count + 1).
    Returns their full 7.2 file summaries as JSON.
    """
    # 1. Map file to chunk count
    chunks_path = os.path.join(out_dir, "chunks", "chunks.jsonl")
    chunk_counts = defaultdict(int)
    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            for line in f:
                c = json.loads(line)
                chunk_counts[c["file"]] += 1
    
    # 2. Get all file summaries
    files_dir = os.path.join(out_dir, "summaries", "files")
    summaries = []
    if os.path.exists(files_dir):
        for fname in os.listdir(files_dir):
            if fname.endswith(".json"):
                with open(os.path.join(files_dir, fname), "r") as f:
                    data = json.load(f)
                    if "error" not in data:
                        summaries.append(data)
    
    # 3. Score and rank (filtering out tests)
    scored = []
    for s in summaries:
        fp = s.get("file", "")
        # Filter out test/example roles
        role = s.get("architectural_role", "").lower()
        if "test" in role or "example" in role:
            continue
        
        cs = centrality_scores.get(fp, 0.0)
        cc = chunk_counts.get(fp, 0)
        score = cs * math.log(cc + 1)
        scored.append((s, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_3 = [item[0] for item in scored[:3]]
    return json.dumps(top_3, indent=2)


def _get_narrative_import_graph(import_graph: Optional[Dict]) -> str:
    """Convert folder edges to narrative dependency sentences."""
    if not import_graph:
        return "No folder-level dependency data available."
    
    folder_edges = defaultdict(set)
    for edge in import_graph.get("edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        sf = "/".join(src.replace("\\", "/").split("/")[:-1])
        tf = "/".join(tgt.replace("\\", "/").split("/")[:-1])
        if sf and tf and sf != tf:
            folder_edges[sf].add(tf)
    
    lines = []
    for sf in sorted(folder_edges.keys()):
        targets = ", ".join(sorted(list(folder_edges[sf])))
        lines.append(f"- {sf} depends on: {targets}")
    return "\n".join(lines) if lines else "No cross-folder dependencies detected."


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_repo_inference(config: RunConfig, out_dir: str) -> None:
    """
    Assemble all evidence blocks, call the 32B model once, and write
    summaries/repo_architecture.json.
    """
    out_file = os.path.join(out_dir, "summaries", "repo_architecture.json")
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                data = json.load(f)
            if "error" not in data:
                logger.info("repo_architecture.json already exists. Skipping Step 7.4.")
                return
        except Exception:
            pass

    model = config.arch_model  # 32B
    stage = "repo"
    prompt_version = PROMPT_VERSION.get("repo", "v1")

    # 1. Load entrypoints
    entrypoints_raw = _load_json(
        os.path.join(out_dir, "analysis", "entrypoints.json"),
        "entrypoints.json"
    )
    entrypoints_json = json.dumps(entrypoints_raw, indent=2) if entrypoints_raw else "{}"

    # 2. Load and split module summaries by centrality
    module_summaries = _load_module_summaries(out_dir)
    if not module_summaries:
        logger.warning("No module summaries found. Step 7.4 cannot run meaningfully.")

    centrality_scores_raw = _load_json(
        os.path.join(out_dir, "analysis", "centrality_scores.json"),
        "centrality_scores.json"
    )
    centrality_scores: Dict[str, float] = centrality_scores_raw or {}

    high_centrality_json, remaining_json = _split_module_summaries(module_summaries, centrality_scores)

    # 3. Load dependencies
    deps_raw = _load_json(
        os.path.join(out_dir, "analysis", "dependencies.json"),
        "dependencies.json"
    )
    dependencies_json = json.dumps(deps_raw, indent=2) if deps_raw else "{}"

    # 4. Narrative folder-level import graph
    import_graph_raw = _load_json(
        os.path.join(out_dir, "analysis", "import_graph.json"),
        "import_graph.json"
    )
    narrative_import_graph_text = _get_narrative_import_graph(import_graph_raw)

    # 5. Build repo tree
    repo_tree_text = _build_repo_tree(out_dir)

    # 6. High-signal evidence injection
    top_files_json = _get_top_files_json(out_dir, centrality_scores)

    # 7. Assemble and call
    async def _call_inference():
        # Global RAG (unfiltered mission/architecture query)
        collection = load_collection(out_dir)
        global_rag_evidence = await asyncio.to_thread(
            retrieve_evidence,
            collection,
            "project architecture summary, core execution loop, and main entry points",
            None, # file_filter
            None, # exclude_filter
            5,    # n_results
            config.embedding_model
        )

        try:
            user_prompt = REPO_ARCH_USER_PROMPT.format(
                entrypoints_json=entrypoints_json,
                top_files_json=top_files_json, # NEW
                high_centrality_module_summaries_json=high_centrality_json,
                narrative_import_graph_text=narrative_import_graph_text, # NEW
                global_rag_evidence=global_rag_evidence, # NEW
                remaining_module_summaries_json=remaining_json,
                repo_tree_text=repo_tree_text,
            )
        except KeyError as e:
            logger.error(f"Prompt format error in Step 7.4: {e}")
            return {"error": f"format_error: {e}"}

        # Cache check / Call LLM
        c_key = get_cache_key(user_prompt, model, prompt_version)
        cached_data = read_cache(out_dir, stage, c_key)
        if cached_data:
            return cached_data
        
        logger.info(f"Step 7.4: Calling {model} for repo-wide architecture inference...")
        client = InferenceClient(config)
        resp = await client.generate_json_async(
            model=model,
            system_prompt=REPO_ARCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=8192,
            seed=42,
            stage=stage
        )
        if "error" not in resp:
            write_cache(out_dir, stage, c_key, resp)
        await client.aclose()
        return resp

    response_data = asyncio.run(_call_inference())

    # 8. Write output
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2)

    if "error" in response_data:
        logger.error(f"Step 7.4 inference failed. Error marker written to {out_file}.")
    else:
        logger.info(f"Step 7.4 complete. Repo architecture written to {out_file}.")
