import sys
import os
import io
import argparse
import warnings
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import torch
from transformers import AutoTokenizer

# ----------------------------
# Clean warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# ----------------------------
# Add repo root to PYTHONPATH
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Optional: force HF cache location (safe)
HF_CACHE = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = HF_CACHE

from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams


def load_hparams(method: str, hparams_path: str):
    method = method.lower()
    if method == "rome":
        return ROMEHyperParams.from_hparams(hparams_path)
    if method == "memit":
        return MEMITHyperParams.from_hparams(hparams_path)
    raise ValueError("Unsupported method. Use 'rome' or 'memit'.")


def hp_get(hparams, name: str, default):
    return getattr(hparams, name, default)


def get_tokenizer(editor: BaseEditor, model_name: str):
    tok = getattr(editor, "tok", None)
    if tok is not None:
        return tok
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def apply_edit(
    editor: BaseEditor,
    *,
    prompt: str,
    subject: str,
    ground_truth: str,
    target_new: str,
    verbose: bool,
    suppress_internal_prints: bool,
):
    """
    Apply one edit with persistent weights.
    NOTE: sequential_edit=True is HARDCODED by design.
    """
    kwargs = dict(
        prompts=[prompt],
        ground_truth=[ground_truth],
        target_new=[target_new],
        subject=[subject],
        sequential_edit=True,   # <-- IMPORTANT: persistent edit
        verbose=verbose,
        test_generation=False,
    )

    if suppress_internal_prints:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            metrics, edited_model, _ = editor.edit(**kwargs)
        return metrics, edited_model
    else:
        metrics, edited_model, _ = editor.edit(**kwargs)
        return metrics, edited_model

def unwrap_metric(value):
    # Unwrap list like [np.float64(1.0)]
    if isinstance(value, list) and len(value) == 1:
        value = value[0]

    # Convert numpy scalar to Python float
    try:
        value = float(value)
    except Exception:
        pass

    return value

def print_metrics(metrics):
    for i, case in enumerate(metrics):
        print(f"\n[CASE {i}]")

        print("  PRE:")
        for k, v in case["pre"].items():
            print(f"    {k}: {replace_with_color(unwrap_metric(v), 'green')}")

        print("  POST:")
        for k, v in case["post"].items():
            print(f"    {k}: {replace_with_color(unwrap_metric(v), 'green')}")



def print_color(text: str, color_name: str):
    # Simple function to print colored text in the terminal
    color_codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
    }
    color_code = color_codes.get(color_name, "")
    print(f"\033[{color_code}m{text}\033[0m")

def color_pattern_in_text(text: str, pattern: str, color_name: str) -> str:
    # Simple function to color all occurrences of a pattern in the text
    color_codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
    }
    color_code = color_codes.get(color_name, "")
    colored_pattern = f"\033[{color_code}m{pattern}\033[0m"
    return text.replace(pattern, colored_pattern)

def replace_with_color(text: str, color_name: str) -> str:
    color_codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
    }
    color_code = color_codes.get(color_name, "")
    return f"\033[{color_code}m{text}\033[0m"


def main():
    parser = argparse.ArgumentParser(description="EasyEdit single test with rollback (minimal)")
    parser.add_argument("--method", required=True, choices=["rome", "memit"])
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--mode", default="both", choices=["forward", "inverse", "both"])
    args = parser.parse_args()

    print_color("[INFO] Loading hyperparameters...", "gray")
    
    hparams = load_hparams(args.method, args.hparams)

    # Read only what we need from YAML
    max_new_tokens = int(hp_get(hparams, "exp_max_new_tokens", 20))
    suppress_internal = bool(hp_get(hparams, "exp_suppress_internal_prints", True))
    verbose = bool(hp_get(hparams, "exp_verbose", False))

    print_color(f"[INFO] YAML params: max_new_tokens={max_new_tokens}, suppress_internal_prints={suppress_internal}, verbose={verbose}", "gray")

    # Fix model_name issues (your setup)
    forced_model_name = "gpt2-xl"
    print_color("[INFO] Forcing HuggingFace model_name...", "gray")
    hparams.model_name = forced_model_name
    for attr in ["model_path", "cache_dir", "model_dir"]:
        if hasattr(hparams, attr):
            setattr(hparams, attr, None)

    print_color("[INFO] Building editor...", "gray")
    editor = BaseEditor.from_hparams(hparams)
    tok = get_tokenizer(editor, forced_model_name)

    # Keep prompt consistent (trailing space)
    prompt = "The capital of Canada is"
    subject = "Canada"
    gt = "Ottawa"
    new = "Toronto"

    # ----------------------------
    # M0: baseline
    # ----------------------------
    print_color("[INFO] Behavioral probe (M0)...", "gray")
    comp0 = generate_completion(editor.model, tok, prompt, max_new_tokens=max_new_tokens)
    print("\n=== BEHAVIORAL (M0) ===")
    print(comp0)

    metrics_fwd = None
    metrics_inv = None

    # ----------------------------
    # Forward edit
    # ----------------------------
    if args.mode in ("forward", "both"):
        print_color("[INFO] Applying FORWARD edit (GT -> NEW)...", "gray")
        metrics_fwd, edited_model = apply_edit(
            editor,
            prompt=prompt,
            subject=subject,
            ground_truth=gt,
            target_new=new,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
        )

        print("\n=== FORWARD METRICS ===")
        print(type(metrics_fwd))
        print_metrics(metrics_fwd)

        print_color("[INFO] Behavioral probe (M1)...", "gray")
        comp1 = generate_completion(edited_model, tok, prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (M1) ===")
        print(comp1)

    # ----------------------------
    # Inverse edit (rollback attempt)
    # ----------------------------
    if args.mode == "inverse":
        print_color("[INFO] Applying INVERSE edit only (NEW -> GT) on M0...", "gray")
        metrics_inv, inv_model = apply_edit(
            editor,
            prompt=prompt,
            subject=subject,
            ground_truth=new,
            target_new=gt,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
        )

        print("\n=== INVERSE METRICS ===")
        print(metrics_inv)

        print_color("[INFO] Behavioral probe (after inverse-only)...", "gray")
        comp2 = generate_completion(inv_model, tok, prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (after inverse-only) ===")
        print(comp2)

    if args.mode == "both":
        print_color("[INFO] Switching editor to edited model for rollback...", "gray")
        editor.model = edited_model

        print_color("[INFO] Applying INVERSE edit (rollback: NEW -> GT) on M1...", "gray")
        metrics_inv, rollback_model = apply_edit(
            editor,
            prompt=prompt,
            subject=subject,
            ground_truth=new,
            target_new=gt,
            verbose=verbose,
            suppress_internal_prints=suppress_internal,
        )

        print("\n=== INVERSE (ROLLBACK) METRICS ===")
        # print the inv metrics but in a readable way
        print_metrics(metrics_inv) 

        print_color("[INFO] Behavioral probe (M2)...", "gray")
        comp2 = generate_completion(rollback_model, tok, prompt, max_new_tokens=max_new_tokens)
        print("\n=== BEHAVIORAL (M2) ===")
        print(comp2)


if __name__ == "__main__":
    main()
