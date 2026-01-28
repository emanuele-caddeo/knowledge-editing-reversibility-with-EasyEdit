import sys
import inspect
from pathlib import Path
import argparse
import yaml
import warnings
from typing import Any, Dict, List

import torch

# ----------------------------
# Clean warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

from thesis_experiments.scripts.utils_io import (
    print_log,
    _resolve_path_like_cfg,
    raise_path_error,
    _load_records,
    _normalize_record,
    _as_str,
    _as_int,
    _as_bool,
)

from thesis_experiments.scripts.pretty_print_utilities import (
    print_metrics_table,
    print_hparams_table,
    print_color,
)

from thesis_experiments.scripts.butterfly_effect_ppl import (
    load_ppl_texts_from_json,
    compute_ppl,
    butterfly_report,
    is_collapse,
)

from ke_core import (
    load_hparams,
    force_hf_home,
    get_tokenizer,
    generate_completion,
    apply_edit,
)

from easyeditor import BaseEditor


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
        description="Run sample with forward + inverse (rollback) + Butterfly Effect (PPL)."
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
    # Butterfly Effect (PPL) config
    # ----------------------------
    be_enabled = _as_bool(cfg.get("be_enabled", True), True)

    be_ppl_data_path_raw = _as_str(cfg.get("be_ppl_data_path", ""), "").strip()
    be_ppl_text_key = _as_str(cfg.get("be_ppl_text_key", "Text"), "Text").strip()
    be_ppl_max_items_raw = cfg.get("be_ppl_max_items", None)
    be_ppl_batch_size = _as_int(cfg.get("be_ppl_batch_size", 16), 16)
    be_ppl_add_start_token = _as_bool(cfg.get("be_ppl_add_start_token", False), False)
    be_ppl_max_length_raw = cfg.get("be_ppl_max_length", None)

    be_collapse_rel_threshold = float(cfg.get("be_collapse_rel_threshold", 1.5) or 1.5)
    be_collapse_abs_threshold = cfg.get("be_collapse_abs_threshold", None)

    be_ppl_max_items = None
    if be_ppl_max_items_raw is not None:
        be_ppl_max_items = _as_int(be_ppl_max_items_raw, 0)
        if be_ppl_max_items <= 0:
            raise ValueError(f"be_ppl_max_items must be > 0 when provided. Got: {be_ppl_max_items_raw}")

    be_ppl_max_length = None
    if be_ppl_max_length_raw is not None:
        be_ppl_max_length = _as_int(be_ppl_max_length_raw, 0)
        if be_ppl_max_length <= 0:
            raise ValueError(f"be_ppl_max_length must be > 0 when provided. Got: {be_ppl_max_length_raw}")

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
    print_log("\n[INFO] Loading hparams + building editor.")
    print_log(f"[INFO] Using hparams file: {exp_hparams_path}")

    force_hf_home()
    hparams = load_hparams(method, str(exp_hparams_path))

    hparams.model_name = forced_model_name

    for attr in ["model_path", "cache_dir", "model_dir"]:
        if hasattr(hparams, attr):
            setattr(hparams, attr, None)

    if hasattr(hparams, "stats_dir"):
        stats_dir = getattr(hparams, "stats_dir")
        if stats_dir in (None, "", "null"):
            raise ValueError("stats_dir in hparams is empty/None. Set stats_dir to a valid folder path in the hparams YAML.")
        Path(str(stats_dir)).mkdir(parents=True, exist_ok=True)

    editor: BaseEditor = BaseEditor.from_hparams(hparams)
    tok = get_tokenizer(editor, forced_model_name)

    if verbose:
        print_hparams_table(hparams)

    # ----------------------------
    # BE: load PPL texts + choose device
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl_texts: List[str] = []
    ppl_m0 = None

    if be_enabled:
        if not be_ppl_data_path_raw:
            raise ValueError("be_enabled=True but 'be_ppl_data_path' is missing in config YAML.")

        be_ppl_data_path = _resolve_path_like_cfg(cfg_path, be_ppl_data_path_raw)
        if not be_ppl_data_path.exists():
            raise_path_error("BE PPL dataset file", be_ppl_data_path)

        ppl_texts = load_ppl_texts_from_json(
            path=be_ppl_data_path,
            text_key=be_ppl_text_key,
            max_items=be_ppl_max_items,
        )

        print_log(f"[INFO] BE enabled. Loaded {len(ppl_texts)} PPL texts from: {be_ppl_data_path}")
        print_log(f"[INFO] BE device: {device}")

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
    # BE: PPL on M0 (baseline)
    # ----------------------------
    if be_enabled:
        print_log("\n[INFO] BE: computing PPL on M0 (baseline).")
        ppl_m0, _ = compute_ppl(
            texts=ppl_texts,
            model=editor.model,
            tokenizer=tok,
            device=device,
            batch_size=be_ppl_batch_size,
            add_start_token=be_ppl_add_start_token,
            max_length=be_ppl_max_length,
        )
        print(f"\n=== BE (M0) ===\nmean_ppl: {ppl_m0:.4f}")

    # ----------------------------
    # M0: behavioral probe
    # ----------------------------
    print_log("\n[INFO] Behavioral probe (M0).")
    comp0 = generate_completion(
        editor.model,
        tok,
        sample_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    print("\n=== BEHAVIORAL (M0) ===")
    print(comp0)

    metrics_fwd = None
    metrics_inv = None
    edited_model = None

    # ----------------------------
    # Forward edit
    # ----------------------------
    if args.mode in ("forward", "both"):
        print_log("\n[INFO] Applying FORWARD edit (GT -> NEW).")

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

        # ----------------------------
        # BE: PPL on M1 (after forward edit)
        # ----------------------------
        if be_enabled and ppl_m0 is not None:
            print_log("\n[INFO] BE: computing PPL on M1 (after forward edit).")
            ppl_m1, _ = compute_ppl(
                texts=ppl_texts,
                model=edited_model,
                tokenizer=tok,
                device=device,
                batch_size=be_ppl_batch_size,
                add_start_token=be_ppl_add_start_token,
                max_length=be_ppl_max_length,
            )
            rep = butterfly_report(ppl_before=ppl_m0, ppl_after=ppl_m1)
            collapse = is_collapse(
                rep,
                rel_threshold=be_collapse_rel_threshold,
                abs_threshold=be_collapse_abs_threshold,
            )

            metrics_fwd = metrics_fwd or {}
            metrics_fwd["butterfly_effect"] = {
                "stage": "M1_after_forward",
                **rep.to_dict(),
                "collapse_rel_threshold": float(be_collapse_rel_threshold),
                "collapse_abs_threshold": (float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None),
                "is_collapse": bool(collapse),
            }

            print(
                "\n=== BE (M1) ===\n"
                f"mean_ppl: {ppl_m1:.4f}\n"
                f"delta_abs: {rep.ppl_delta_abs:.4f}\n"
                f"delta_rel: {rep.ppl_delta_rel:.4f}\n"
                f"is_collapse: {collapse}"
            )

        print_metrics_table(metrics_fwd, title="FORWARD METRICS")

        print_log("\n[INFO] Behavioral probe (M1).")
        comp1 = generate_completion(
            edited_model,
            tok,
            sample_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print("\n=== BEHAVIORAL (M1) ===")
        print(comp1)

    # ----------------------------
    # Inverse / rollback
    # ----------------------------
    if args.mode == "inverse":
        print_log("\n[INFO] Applying INVERSE edit only (NEW -> GT) on M0.")

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

        print_log("\n[INFO] Behavioral probe (after inverse-only).")
        comp2 = generate_completion(
            inv_model,
            tok,
            sample_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print("\n=== BEHAVIORAL (after inverse-only) ===")
        print(comp2)

    if args.mode == "both":
        if edited_model is None:
            raise RuntimeError("edited_model is None. Forward edit did not run, cannot perform rollback.")

        print_log("\n[INFO] Switching editor to edited model for rollback.")
        editor.model = edited_model

        print_log("\n[INFO] Applying INVERSE edit (rollback: NEW -> GT) on M1.")

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

        # ----------------------------
        # BE: PPL on M2 (after rollback)
        # ----------------------------
        if be_enabled and ppl_m0 is not None:
            print_log("\n[INFO] BE: computing PPL on M2 (after rollback).")
            ppl_m2, _ = compute_ppl(
                texts=ppl_texts,
                model=rollback_model,
                tokenizer=tok,
                device=device,
                batch_size=be_ppl_batch_size,
                add_start_token=be_ppl_add_start_token,
                max_length=be_ppl_max_length,
            )
            rep2 = butterfly_report(ppl_before=ppl_m0, ppl_after=ppl_m2)
            collapse2 = is_collapse(
                rep2,
                rel_threshold=be_collapse_rel_threshold,
                abs_threshold=be_collapse_abs_threshold,
            )

            metrics_inv = metrics_inv or {}
            metrics_inv["butterfly_effect"] = {
                "stage": "M2_after_rollback",
                **rep2.to_dict(),
                "collapse_rel_threshold": float(be_collapse_rel_threshold),
                "collapse_abs_threshold": (float(be_collapse_abs_threshold) if be_collapse_abs_threshold is not None else None),
                "is_collapse": bool(collapse2),
            }

            print(
                "\n=== BE (M2) ===\n"
                f"mean_ppl: {ppl_m2:.4f}\n"
                f"delta_abs: {rep2.ppl_delta_abs:.4f}\n"
                f"delta_rel: {rep2.ppl_delta_rel:.4f}\n"
                f"is_collapse: {collapse2}"
            )

        print_metrics_table(metrics_inv, title="INVERSE (ROLLBACK) METRICS")

        print_log("\n[INFO] Behavioral probe (M2).")
        comp2 = generate_completion(
            rollback_model,
            tok,
            sample_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print("\n=== BEHAVIORAL (M2) ===")
        print(comp2)


if __name__ == "__main__":
    main()
