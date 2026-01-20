import sys
import json
import inspect
from pathlib import Path
import argparse
import yaml
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

from thesis_experiments.scripts.utils_io import load_knowedit_records_flexible, print_log
from thesis_experiments.scripts.pretty_print_utilities import print_metrics_table, print_hparams_table, print_color

# ----------------------------
# Clean warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# ------------------------------------------------------------
# Ensure import paths (repo root + scripts folder)
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ke_core import (  # noqa: E402
    load_hparams,
    force_hf_home,
    get_tokenizer,
    generate_completion,
    apply_edit,
)
from easyeditor import BaseEditor  # noqa: E402

# ----------------------------
# Helpers
# ----------------------------
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


def _call_apply_edit(editor: BaseEditor, **kwargs):
    """
    Call apply_edit(...) safely:
    - If apply_edit supports extra kwargs (locality/portability), pass them.
    - Otherwise, silently drop unsupported keys.
    """
    sig = inspect.signature(apply_edit)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if accepts_var_kw:
        return apply_edit(editor, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = set(kwargs.keys()) - set(filtered.keys())

    if dropped:
        print_color(f"[WARN] apply_edit does not support keys, dropped: {sorted(dropped)}", "yellow")

    return apply_edit(editor, **filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Run sample with forward + inverse (rollback). Uses separate experiment config + hparams YAML."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--mode", default="both", choices=["forward", "inverse", "both"])
    args = parser.parse_args()

    # ----------------------------
    # Experiment config loading
    # ----------------------------
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise_path_error("Experiment config file", cfg_path)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure (expected mapping) in: {cfg_path}")

    method = str(cfg.get("exp_method", "rome")).lower().strip()
    if method not in ("rome", "memit"):
        raise ValueError(f"Invalid exp_method='{method}'. Must be 'rome' or 'memit'.")

    dataset_type = _as_str(cfg.get("exp_dataset_type", "knowedit"), "knowedit").strip().lower()

    # ----------------------------
    # Hparams config (separate file)
    # ----------------------------
    exp_hparams_path_raw = str(cfg.get("exp_hparams_path", "")).strip()
    if not exp_hparams_path_raw:
        raise ValueError("Missing exp_hparams_path in experiment config (must point to a separate hparams YAML).")

    exp_hparams_path = _resolve_path_like_cfg(cfg_path, exp_hparams_path_raw)
    if not exp_hparams_path.exists():
        raise_path_error("Hparams config file", exp_hparams_path)

    # Optional sanity-check: ensure hparams file has alg_name
    hparams_cfg = yaml.safe_load(exp_hparams_path.read_text(encoding="utf-8")) or {}
    if not isinstance(hparams_cfg, dict):
        raise ValueError(f"Invalid hparams YAML structure (expected mapping) in: {exp_hparams_path}")
    if "alg_name" not in hparams_cfg:
        raise ValueError(f"Hparams YAML must contain 'alg_name'. Missing in: {exp_hparams_path}")

    # ----------------------------
    # Sample selection + runner params
    # ----------------------------
    sample_index = _as_int(cfg.get("exp_sample_index", 0), 0)
    forced_model_name = str(cfg.get("exp_forced_model_name", "gpt2-xl")).strip()
    max_new_tokens = _as_int(cfg.get("exp_max_new_tokens", 20), 20)
    temperature = float(cfg.get("exp_temperature", 1.0) or 1.0)
    do_sample = _as_bool(cfg.get("exp_do_sample", False), False)
    suppress_internal = _as_bool(cfg.get("exp_suppress_internal_prints", True), True)
    verbose = _as_bool(cfg.get("exp_verbose", False), False)

    if max_new_tokens <= 0:
        raise ValueError(f"exp_max_new_tokens must be > 0. Got: {max_new_tokens}")

    # ----------------------------
    # Load dataset records
    # ----------------------------
    print_log(f"\n[INFO] Loading records (dataset_type={dataset_type})...")
    raw_records, dataset_source = _load_records(cfg_path, cfg)

    if not raw_records:
        raise RuntimeError("No records loaded from dataset (empty list).")
    if sample_index < 0 or sample_index >= len(raw_records):
        raise IndexError(f"exp_sample_index={sample_index} out of range. Dataset size={len(raw_records)}")

    rec_raw = raw_records[sample_index]
    rec = _normalize_record(rec_raw, dataset_type)

    # Minimal dot-access wrapper
    class _Sample:
        def __init__(self, d: Dict[str, Any]):
            self.case_id = d.get("case_id", "")
            self.prompt = d["prompt"]
            self.subject = d["subject"]
            self.ground_truth = d["ground_truth"]
            self.target_new = d["target_new"]
            self.locality_prompts = d.get("locality_prompts", [])
            self.portability_prompts = d.get("portability_prompts", [])

    sample = _Sample(rec)
    sample_prompt = str(sample.prompt).rstrip()

    # ----------------------------
    # Load hparams + build editor
    # ----------------------------
    print_log("\n[INFO] Loading hparams + building editor...")
    print_log(f"[INFO] Using hparams file: {exp_hparams_path}")

    force_hf_home()
    hparams = load_hparams(method, str(exp_hparams_path))

    # Force model_name to avoid local-cache weirdness
    hparams.model_name = forced_model_name

    # Avoid custom paths that may conflict with your environment
    for attr in ["model_path", "cache_dir", "model_dir"]:
        if hasattr(hparams, attr):
            setattr(hparams, attr, None)

    # Ensure stats_dir exists if present in hparams (ROME uses it)
    if hasattr(hparams, "stats_dir"):
        stats_dir = getattr(hparams, "stats_dir")
        if stats_dir in (None, "", "null"):
            raise ValueError("stats_dir in hparams is empty/None. Set stats_dir to a valid folder path in the hparams YAML.")
        Path(str(stats_dir)).mkdir(parents=True, exist_ok=True)

    editor: BaseEditor = BaseEditor.from_hparams(hparams)
    tok = get_tokenizer(editor, forced_model_name)

    # Print hparams in a compact table if verbose.
    if verbose:
        print_hparams_table(hparams)

    # ----------------------------
    # Print sample summary
    # ----------------------------
    print("\n=== SAMPLE ===")
    print(f"method: {method}")
    print(f"dataset_type: {dataset_type}")
    print(f"dataset_source: {dataset_source}")
    if dataset_source == "local":
        print(f"dataset_path: {cfg.get('exp_dataset_path', '')}")
    else:
        print(f"hf_dataset: {cfg.get('hf_dataset', None)}")
        print(f"hf_split: {cfg.get('hf_split', 'test')}")
        print(f"hf_subset: {cfg.get('hf_subset', None)}")
    print(f"dataset_size_loaded: {len(raw_records)}")
    print(f"sample_index: {sample_index}")
    print(f"case_id: {sample.case_id}")
    print(f"prompt: {sample_prompt}")
    print(f"subject: {sample.subject}")
    print(f"ground_truth: {sample.ground_truth}")
    print(f"target_new: {sample.target_new}")
    print(f"locality_prompts: {len(sample.locality_prompts)}")
    print(f"portability_prompts: {len(sample.portability_prompts)}")

    # ----------------------------
    # M0: baseline
    # ----------------------------
    print_log("\n[INFO] Behavioral probe (M0)...")
    comp0 = generate_completion(editor.model, tok, sample_prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
    print("\n=== BEHAVIORAL (M0) ===")
    print(comp0)

    metrics_fwd = None
    metrics_inv = None
    edited_model = None

    # ----------------------------
    # Forward edit
    # ----------------------------
    if args.mode in ("forward", "both"):
        print_log("\n[INFO] Applying FORWARD edit (GT -> NEW)...")

        extra_eval = {}
        if isinstance(sample.locality_prompts, list) and sample.locality_prompts:
            extra_eval["locality_prompts"] = sample.locality_prompts
        if isinstance(sample.portability_prompts, list) and sample.portability_prompts:
            extra_eval["portability_prompts"] = sample.portability_prompts

        metrics_fwd, edited_model = _call_apply_edit(
            editor,
            prompt=sample_prompt,
            subject=sample.subject,
            ground_truth=sample.ground_truth,
            target_new=sample.target_new,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
            **extra_eval,
        )
        print_metrics_table(metrics_fwd, title="FORWARD METRICS")

        print_log("\n[INFO] Behavioral probe (M1)...")
        comp1 = generate_completion(edited_model, tok, sample_prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (M1) ===")
        print(comp1)

    # ----------------------------
    # Inverse / rollback
    # ----------------------------
    if args.mode == "inverse":
        print_log("\n[INFO] Applying INVERSE edit only (NEW -> GT) on M0...")

        extra_eval = {}
        if isinstance(sample.locality_prompts, list) and sample.locality_prompts:
            extra_eval["locality_prompts"] = sample.locality_prompts
        if isinstance(sample.portability_prompts, list) and sample.portability_prompts:
            extra_eval["portability_prompts"] = sample.portability_prompts

        metrics_inv, inv_model = _call_apply_edit(
            editor,
            prompt=sample_prompt,
            subject=sample.subject,
            ground_truth=sample.target_new,
            target_new=sample.ground_truth,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
            **extra_eval,
        )
        print_metrics_table(metrics_inv, title="INVERSE METRICS")

        print_log("\n[INFO] Behavioral probe (after inverse-only)...")
        comp2 = generate_completion(inv_model, tok, sample_prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (after inverse-only) ===")
        print(comp2)

    if args.mode == "both":
        if edited_model is None:
            raise RuntimeError("edited_model is None. Forward edit did not run, cannot perform rollback.")

        print_log("\n[INFO] Switching editor to edited model for rollback...")
        editor.model = edited_model

        print_log("\n[INFO] Applying INVERSE edit (rollback: NEW -> GT) on M1...")

        extra_eval = {}
        if isinstance(sample.locality_prompts, list) and sample.locality_prompts:
            extra_eval["locality_prompts"] = sample.locality_prompts
        if isinstance(sample.portability_prompts, list) and sample.portability_prompts:
            extra_eval["portability_prompts"] = sample.portability_prompts

        metrics_inv, rollback_model = _call_apply_edit(
            editor,
            prompt=sample_prompt,
            subject=sample.subject,
            ground_truth=sample.target_new,
            target_new=sample.ground_truth,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
            **extra_eval,
        )
        print_metrics_table(metrics_inv, title="INVERSE (ROLLBACK) METRICS")

        print_log("\n[INFO] Behavioral probe (M2)...")
        comp2 = generate_completion(rollback_model, tok, sample_prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (M2) ===")
        print(comp2)


if __name__ == "__main__":
    main()
