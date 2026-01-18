import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json_any(path: str) -> Any:
    """Read JSON or JSONL (one JSON object per line)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty dataset file: {path}")

    # Standard JSON
    if raw.startswith("{") or raw.startswith("["):
        return json.loads(raw)

    # JSONL fallback
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def flatten_to_str(x):
    """
    Normalize ground_truth / target_new to a plain string.
    """
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]

    if not isinstance(x, str):
        raise ValueError(f"Expected string after flattening, got {type(x)}: {x}")

    return x.strip()

def normalize_knowedit_record(x: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a KnowEdit-style editing record to our minimal schema:
    subject, prompt, target_new, ground_truth

    The HF KnowEdit benchmark files may omit `ground_truth` and store it under aliases
    like `target_true`, `target`, `answer`, or inside locality/portability structures.
    """
    out = dict(x)

    # --- Required: subject, prompt, target_new ---
    subject = out.get("subject")
    prompt = out.get("prompt")
    target_new = out.get("target_new")

    if subject is None or prompt is None or target_new is None:
        missing = [k for k in ["subject", "prompt", "target_new"] if out.get(k) is None]
        raise KeyError(
            f"Missing required KnowEdit fields: {missing}. Available keys: {list(out.keys())[:30]}"
        )

    # --- Ground truth: direct or aliases ---
    ground_truth = out.get("ground_truth")

    if ground_truth is None:
        # common aliases
        for k in ["target_true", "target", "answer", "ground", "label"]:
            if k in out and out[k] is not None:
                ground_truth = out[k]
                break

    # --- Ground truth: try to infer from locality/portability blocks ---
    # Some benchmarks store the original answer as "target" inside locality/portability dicts.
    if ground_truth is None:
        for block_key in ["locality", "portability"]:
            blk = out.get(block_key)
            if isinstance(blk, dict):
                # Look for nested "target" or "ground_truth"
                for kk in ["ground_truth", "target_true", "target", "answer"]:
                    if kk in blk and blk[kk] is not None:
                        ground_truth = blk[kk]
                        break
                if ground_truth is not None:
                    break

                # Sometimes locality/portability contain lists of dicts
                for vv in blk.values():
                    if isinstance(vv, list) and vv:
                        v0 = vv[0]
                        if isinstance(v0, dict):
                            for kk in ["ground_truth", "target_true", "target", "answer"]:
                                if kk in v0 and v0[kk] is not None:
                                    ground_truth = v0[kk]
                                    break
                        if ground_truth is not None:
                            break
            if ground_truth is not None:
                break

    if ground_truth is None:
        raise KeyError(
            "Missing required KnowEdit fields: ['ground_truth']. "
            f"Available keys: {list(out.keys())[:30]}"
        )

    # Final normalized keys
    out["subject"] = subject
    out["prompt"] = prompt
    out["ground_truth"] = flatten_to_str(ground_truth)
    out["target_new"] = flatten_to_str(target_new)


    out.setdefault("case_id", out.get("case_id", out.get("id", "")))
    return out



def load_knowedit_records_flexible(
    *,
    dataset_path: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_split: str = "test",
    hf_subset: Optional[str] = None,
    seed: int = 42,
    sample_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load KnowEdit records from local JSON/JSONL or HuggingFace."""
    if dataset_path:
        data = read_json_any(dataset_path)
        if isinstance(data, dict) and "data" in data:
            records = data["data"]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(f"Unsupported dataset structure in {dataset_path}")
    elif hf_dataset:
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_dataset, name=hf_subset, split=hf_split)
            records = [dict(r) for r in ds]
        except ValueError:
            # Fallback: download split file via huggingface_hub and parse locally
            records = _load_from_hf_files(hf_dataset, hf_split)
    else:
        raise ValueError("Provide either dataset_path or hf_dataset")

    records = [normalize_knowedit_record(r) for r in records]

    if sample_k is not None:
        rng = random.Random(seed)
        records = rng.sample(records, min(sample_k, len(records)))

    return records

def _hf_pick_split_file(hf_dataset: str, repo_files: List[str], split: str) -> str:
    """
    Pick the best file for knowledge editing examples.
    We avoid probe/blender files like {ent,pos,neg} by inspecting the schema of a few candidates.
    Preference order:
      1) files containing editing-like fields (prompt/subject/target_new/ground_truth or known aliases)
      2) then prefer jsonl over json
    """
    split_l = split.lower()

    # Candidate pool: only json/jsonl and split-related names
    candidates = []
    for f in repo_files:
        fl = f.lower()
        if not fl.endswith((".jsonl", ".json")):
            continue
        if (f"/{split_l}" in fl) or (f"_{split_l}." in fl) or (fl.endswith(f"{split_l}.json")) or (fl.endswith(f"{split_l}.jsonl")):
            candidates.append(f)

    if not candidates:
        candidates = [f for f in repo_files if f.lower().endswith((".jsonl", ".json"))]

    if not candidates:
        raise FileNotFoundError(f"No .json/.jsonl files found in HF repo for split='{split}'")

    # Try to inspect a few candidates quickly by downloading and peeking at first record keys
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        # No hub available here; fallback to filename heuristic
        candidates.sort(key=lambda x: (0 if x.lower().endswith(".jsonl") else 1, len(x)))
        return candidates[0]

    def looks_like_editing_file(filename: str) -> bool:
        # Download and inspect first record keys
        local_path = hf_hub_download(repo_id=hf_dataset, repo_type="dataset", filename=filename)
        data = read_json_any(local_path)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        if not isinstance(data, list) or not data:
            return False
        r0 = data[0]
        if not isinstance(r0, dict):
            return False
        keys = set(r0.keys())

        # Editing-like keys (direct or aliases)
        direct = {"prompt", "subject", "target_new", "ground_truth"}
        aliases = {"requested_rewrite", "paraphrase_prompts", "target_true", "target_new", "subject", "prompt"}
        return bool(keys & direct) or bool(keys & aliases)

    # Rank candidates by schema
    good = []
    for f in candidates[:30]:  # cap to avoid too many downloads
        try:
            if looks_like_editing_file(f):
                good.append(f)
        except Exception:
            continue

    chosen_pool = good if good else candidates
    chosen_pool.sort(key=lambda x: (0 if x.lower().endswith(".jsonl") else 1, len(x)))
    return chosen_pool[0]

    return candidates[0]


def _load_from_hf_files(hf_dataset: str, hf_split: str) -> List[Dict[str, Any]]:
    """
    Download the split file from a HuggingFace dataset repo and load as JSON/JSONL locally.
    This avoids `datasets` glob/pattern issues.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "Fallback HF download requires `huggingface_hub`. Install with: pip install huggingface_hub"
        ) from e

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=hf_dataset, repo_type="dataset")
    picked = _hf_pick_split_file(hf_dataset, repo_files, hf_split)
    print_log(f"[INFO] HF fallback picked file: {picked}")

    local_path = hf_hub_download(repo_id=hf_dataset, repo_type="dataset", filename=picked)
    data = read_json_any(local_path)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported dataset structure in downloaded file: {picked}")

def print_log(log_msg: str) -> None:
    """
    Print a log message in gray color to the console.
    Uses ANSI escape codes (works on most modern terminals).
    """
    GRAY = "\033[90m"
    RESET = "\033[0m"
    print(f"{GRAY}{log_msg}{RESET}")
