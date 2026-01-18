import sys
from pathlib import Path
import argparse
import yaml
import warnings

from thesis_experiments.scripts.utils_io import load_knowedit_records_flexible, print_log

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

from ke_core import (
    load_hparams,
    force_hf_home,
    get_tokenizer,
    generate_completion,
    apply_edit,
    format_metrics_nude,
)
from easyeditor import BaseEditor


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


def _resolve_path_like_cfg(base_cfg_path: Path, maybe_rel: str) -> Path:
    """
    Resolve a path in config. If relative, resolve it from repo root.
    """
    p = Path(maybe_rel)
    if p.is_absolute():
        return p
    # Resolve relative to repo root (stable when running from anywhere)
    return REPO_ROOT / p

def raise_path_error(label: str, bad_path: Path):
    RED = "\033[91m"
    RESET = "\033[0m"
    raise FileNotFoundError(
        f"[CONFIG ERROR] {label} not found at path: {RED}{bad_path}{RESET}"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Run KnowEdit sample with forward + inverse (rollback). Uses separate experiment config + hparams YAML."
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
    # Dataset config (local or HF)
    # ----------------------------
    dataset_path = str(cfg.get("exp_dataset_path", "")).strip()  # local JSON/JSONL
    hf_dataset = cfg.get("hf_dataset", None)                     # e.g. "zjunlp/KnowEdit"
    hf_split = str(cfg.get("hf_split", "test")).strip()
    hf_subset = cfg.get("hf_subset", None)

    use_local = bool(dataset_path)
    use_hf = bool(hf_dataset)

    if use_local and use_hf:
        raise ValueError("Config error: provide either exp_dataset_path OR hf_dataset, not both.")
    if (not use_local) and (not use_hf):
        raise ValueError("Config error: provide exp_dataset_path (local) or hf_dataset (HuggingFace).")

    sample_index = _as_int(cfg.get("exp_sample_index", 0), 0)
    seed = _as_int(cfg.get("exp_seed", 42), 42)

    sample_k_raw = cfg.get("exp_sample_k", None)
    sample_k = None
    if sample_k_raw is not None:
        sample_k = _as_int(sample_k_raw, 0)
        if sample_k <= 0:
            raise ValueError(f"exp_sample_k must be > 0 when provided. Got: {sample_k_raw}")

    forced_model_name = str(cfg.get("exp_forced_model_name", "gpt2-xl")).strip()
    max_new_tokens = _as_int(cfg.get("exp_max_new_tokens", 20), 20)
    suppress_internal = _as_bool(cfg.get("exp_suppress_internal_prints", True), True)
    verbose = _as_bool(cfg.get("exp_verbose", False), False)

    if max_new_tokens <= 0:
        raise ValueError(f"exp_max_new_tokens must be > 0. Got: {max_new_tokens}")

    # ----------------------------
    # Load dataset records
    # ----------------------------
    print_log("\n[INFO] Loading KnowEdit records...")
    records = load_knowedit_records_flexible(
        dataset_path=str(_resolve_path_like_cfg(cfg_path, dataset_path)) if use_local else None,
        hf_dataset=hf_dataset if use_hf else None,
        hf_split=hf_split,
        hf_subset=hf_subset,
        seed=seed,
        sample_k=sample_k,
    )

    if not records:
        raise RuntimeError("No records loaded from dataset (empty list).")

    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(f"exp_sample_index={sample_index} out of range. Dataset size={len(records)}")

    rec = records[sample_index]

    # Minimal dot-access wrapper
    class _Sample:
        def __init__(self, d):
            self.case_id = d.get("case_id", "")
            self.prompt = d["prompt"]
            self.subject = d["subject"]
            self.ground_truth = d["ground_truth"]
            self.target_new = d["target_new"]

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
            raise ValueError(
                "stats_dir in hparams is empty/None. Set stats_dir to a valid folder path in the hparams YAML."
            )
        Path(str(stats_dir)).mkdir(parents=True, exist_ok=True)

    editor: BaseEditor = BaseEditor.from_hparams(hparams)
    tok = get_tokenizer(editor, forced_model_name)

    # ----------------------------
    # Print sample summary
    # ----------------------------
    print("\n=== SAMPLE ===")
    print(f"method: {method}")
    print(f"dataset_source: {'local' if use_local else 'hf'}")
    if use_local:
        print(f"dataset_path: {dataset_path}")
    else:
        print(f"hf_dataset: {hf_dataset}")
        print(f"hf_split: {hf_split}")
        print(f"hf_subset: {hf_subset}")
    print(f"dataset_size_loaded: {len(records)}")
    print(f"sample_index: {sample_index}")
    print(f"case_id: {sample.case_id}")
    print(f"prompt: {sample_prompt}")
    print(f"subject: {sample.subject}")
    print(f"ground_truth: {sample.ground_truth}")
    print(f"target_new: {sample.target_new}")

    # ----------------------------
    # M0: baseline
    # ----------------------------
    print_log("\n[INFO] Behavioral probe (M0)...")
    comp0 = generate_completion(editor.model, tok, sample_prompt, max_new_tokens=max_new_tokens)
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
        metrics_fwd, edited_model = apply_edit(
            editor,
            prompt=sample_prompt,
            subject=sample.subject,
            ground_truth=sample.ground_truth,
            target_new=sample.target_new,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
        )
        print("\n=== FORWARD METRICS ===")
        print(format_metrics_nude(metrics_fwd))

        print_log("\n[INFO] Behavioral probe (M1)...")
        comp1 = generate_completion(edited_model, tok, sample_prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (M1) ===")
        print(comp1)

    # ----------------------------
    # Inverse / rollback
    # ----------------------------
    if args.mode == "inverse":
        print_log("\n[INFO] Applying INVERSE edit only (NEW -> GT) on M0...")
        metrics_inv, inv_model = apply_edit(
            editor,
            prompt=sample_prompt,
            subject=sample.subject,
            ground_truth=sample.target_new,
            target_new=sample.ground_truth,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
        )
        print("\n=== INVERSE METRICS ===")
        print(format_metrics_nude(metrics_inv))

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
        metrics_inv, rollback_model = apply_edit(
            editor,
            prompt=sample_prompt,
            subject=sample.subject,
            ground_truth=sample.target_new,
            target_new=sample.ground_truth,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
        )
        print("\n=== INVERSE (ROLLBACK) METRICS ===")
        print(format_metrics_nude(metrics_inv))

        print_log("\n[INFO] Behavioral probe (M2)...")
        comp2 = generate_completion(rollback_model, tok, sample_prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (M2) ===")
        print(comp2)


if __name__ == "__main__":
    main()
