import json
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------
# Ensure import paths (repo root + scripts folder)
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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

def _load_records(cfg_path: Path, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load raw records from either:
      - existing KnowEdit flexible loader (local/HF), or
      - CounterFact via datasets/local JSON.

    Returns: (records, dataset_source_str)
    """
    dataset_type = _as_str(cfg.get("exp_dataset_type", "knowedit"), "knowedit").strip().lower()

    dataset_path = _as_str(cfg.get("exp_dataset_path", ""), "").strip()
    hf_dataset = cfg.get("hf_dataset", None)
    hf_split = _as_str(cfg.get("hf_split", "test"), "test").strip()
    hf_subset = cfg.get("hf_subset", None)

    use_local = bool(dataset_path)
    use_hf = bool(hf_dataset)

    if use_local and use_hf:
        raise ValueError("Config error: provide either exp_dataset_path OR hf_dataset, not both.")
    if (not use_local) and (not use_hf):
        raise ValueError("Config error: provide exp_dataset_path (local) or hf_dataset (HuggingFace).")

    seed = _as_int(cfg.get("exp_seed", 42), 42)
    sample_k_raw = cfg.get("exp_sample_k", None)
    sample_k = None
    if sample_k_raw is not None:
        sample_k = _as_int(sample_k_raw, 0)
        if sample_k <= 0:
            raise ValueError(f"exp_sample_k must be > 0 when provided. Got: {sample_k_raw}")

    # KnowEdit path: reuse your existing robust loader
    if dataset_type == "knowedit":
        records = load_knowedit_records_flexible(
            dataset_path=str(_resolve_path_like_cfg(cfg_path, dataset_path)) if use_local else None,
            hf_dataset=hf_dataset if use_hf else None,
            hf_split=hf_split,
            hf_subset=hf_subset,
            seed=seed,
            sample_k=sample_k,
        )
        return records, ("local" if use_local else "hf")

    # CounterFact: load via HF datasets or local JSON/JSONL
    if dataset_type == "counterfact":
        raw: List[Dict[str, Any]] = []
        if use_hf:
            try:
                from datasets import load_dataset  # type: ignore
            except Exception as e:
                raise RuntimeError("Missing dependency 'datasets'. Install it to use hf_dataset with counterfact.") from e

            ds = load_dataset(hf_dataset, hf_subset, split=hf_split)
            raw = [dict(x) for x in ds]
        else:
            p = _resolve_path_like_cfg(cfg_path, dataset_path)
            if not p.exists():
                raise_path_error("Local dataset file", p)

            # JSONL or JSON list
            text = p.read_text(encoding="utf-8").strip()
            if p.suffix.lower() in (".jsonl", ".jsonl.txt"):
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))
            else:
                obj = json.loads(text)
                if isinstance(obj, list):
                    raw = obj
                else:
                    raise ValueError(f"Local counterfact JSON must be a list of records. Got: {type(obj)}")

        # Optional subsample (deterministic)
        if sample_k is not None and sample_k < len(raw):
            import random
            rnd = random.Random(seed)
            idxs = list(range(len(raw)))
            rnd.shuffle(idxs)
            raw = [raw[i] for i in idxs[:sample_k]]

        return raw, ("local" if use_local else "hf")

    raise ValueError(f"Unsupported exp_dataset_type='{dataset_type}'.")

def _format_prompt_with_subject(prompt: str, subject: str) -> str:
    """
    CounterFact often stores prompt templates like: "{} was born in"
    This function tries to fill placeholders robustly.
    """
    p = (prompt or "").strip()
    s = (subject or "").strip()
    if not p:
        return p
    if "{}" in p:
        try:
            return p.format(s)
        except Exception:
            # If formatting fails, fallback to naive replace
            return p.replace("{}", s)
    if "<SUBJECT>" in p:
        return p.replace("<SUBJECT>", s)
    return p

def _extract_target_str(x: Any) -> str:
    """
    CounterFact targets may be:
    - {"str": "..."} or {"text": "..."} or plain string
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "str" in x and isinstance(x["str"], str):
            return x["str"]
        if "text" in x and isinstance(x["text"], str):
            return x["text"]
        if "name" in x and isinstance(x["name"], str):
            return x["name"]
    return str(x)

def _extract_list_of_prompts(value: Any, subject: str) -> List[str]:
    """
    Accept:
    - list[str]
    - list[dict] with key "prompt"
    - dict with key "prompts"
    """
    out: List[str] = []
    if value is None:
        return out

    if isinstance(value, dict):
        value = value.get("prompts", value.get("prompt", None))

    if isinstance(value, str):
        out.append(_format_prompt_with_subject(value, subject))
        return out

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                out.append(_format_prompt_with_subject(item, subject))
            elif isinstance(item, dict):
                p = item.get("prompt", item.get("text", item.get("template", "")))
                if p:
                    out.append(_format_prompt_with_subject(str(p), subject))
    return [p for p in out if p.strip()]

def _normalize_record(rec: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
    """
    Convert dataset-specific schema into a canonical record:
      - prompt, subject, ground_truth, target_new, case_id (optional)
      - locality_prompts (optional list[str])
      - portability_prompts (optional list[str])
    """
    dt = (dataset_type or "knowedit").strip().lower()

    if dt == "knowedit":
        # Expect canonical fields already (your loader does this for zjunlp/KnowEdit).
        out = {
            "case_id": rec.get("case_id", ""),
            "prompt": rec["prompt"],
            "subject": rec["subject"],
            "ground_truth": rec["ground_truth"],
            "target_new": rec["target_new"],
        }
        # Optional (if you later extend KnowEdit-like records)
        if "locality_prompts" in rec:
            out["locality_prompts"] = list(rec["locality_prompts"] or [])
        if "portability_prompts" in rec:
            out["portability_prompts"] = list(rec["portability_prompts"] or [])
        return out

    if dt == "counterfact":
        # azhx/counterfact schema keys (observed):
        # ['case_id', 'pararel_idx', 'requested_rewrite', 'paraphrase_prompts',
        #  'neighborhood_prompts', 'attribute_prompts', 'generation_prompts']

        rr = rec.get("requested_rewrite", rec)

        subject = _as_str(rr.get("subject", rec.get("subject", ""))).strip()
        prompt_t = _as_str(rr.get("prompt", rec.get("prompt", ""))).strip()
        prompt = _format_prompt_with_subject(prompt_t, subject)

        gt = _extract_target_str(rr.get("target_true", rr.get("ground_truth", rec.get("ground_truth", ""))))
        tn = _extract_target_str(rr.get("target_new", rr.get("target", rec.get("target_new", ""))))

        # Portability (paraphrases)
        portability_raw = (
            rec.get("paraphrase_prompts", None)
            or rec.get("paraphrases", None)
            or rec.get("portability_prompts", None)
            or rec.get("portability", None)
        )

        # Locality (neighborhood)
        locality_raw = (
            rec.get("neighborhood_prompts", None)
            or rec.get("neighborhood", None)
            or rec.get("locality_prompts", None)
            or rec.get("locality", None)
        )

        portability_prompts = _extract_list_of_prompts(portability_raw, subject)
        locality_prompts = _extract_list_of_prompts(locality_raw, subject)

        return {
            "case_id": rec.get("case_id", rec.get("id", "")),
            "prompt": prompt,
            "subject": subject,
            "ground_truth": gt,
            "target_new": tn,
            "locality_prompts": locality_prompts,
            "portability_prompts": portability_prompts,
        }

    raise ValueError(f"Unsupported exp_dataset_type='{dataset_type}'. Use 'knowedit' or 'counterfact'.")

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

def _as_bool(v, default=False) -> bool:
    """Parse common YAML bool representations safely."""
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return bool(default)

def _as_int(v, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, bool):
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)

def _as_str(v, default: str = "") -> str:
    if v is None:
        return default
    return str(v)

def _resolve_path_like_cfg(base_cfg_path: Path, maybe_rel: str) -> Path:
    """
    Resolve a path in config. If relative, resolve it from repo root.
    """
    p = Path(maybe_rel)
    if p.is_absolute():
        return p
    return REPO_ROOT / p

def raise_path_error(label: str, bad_path: Path):
    RED = "\033[91m"
    RESET = "\033[0m"
    raise FileNotFoundError(f"[CONFIG ERROR] {label} not found at path: {RED}{bad_path}{RESET}")