import io
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams


# Keep console clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")


def load_hparams(method: str, hparams_path: str):
    method = method.lower()
    if method == "rome":
        return ROMEHyperParams.from_hparams(hparams_path)
    if method == "memit":
        return MEMITHyperParams.from_hparams(hparams_path)
    raise ValueError("Unsupported method. Use 'rome' or 'memit'.")


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
    Persistent edit by design: sequential_edit=True.
    """
    kwargs = dict(
        prompts=[prompt],
        ground_truth=[ground_truth],
        target_new=[target_new],
        subject=[subject],
        sequential_edit=True,
        verbose=verbose,
        test_generation=False,
    )

    if suppress_internal_prints:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            metrics, edited_model, _ = editor.edit(**kwargs)
        return metrics, edited_model
    metrics, edited_model, _ = editor.edit(**kwargs)
    return metrics, edited_model


def _unwrap_metric(value: Any):
    # Unwrap list like [np.float64(1.0)]
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    try:
        return float(value)
    except Exception:
        return value


def format_metrics_nude(metrics: List[Dict[str, Any]]) -> str:
    """
    Returns a clean string with only scalar values (no np.float64, no [x]).
    Keeps empty dicts as-is.
    """
    lines: List[str] = []
    for i, case in enumerate(metrics):
        lines.append(f"[CASE {i}]")

        pre = case.get("pre", {})
        post = case.get("post", {})

        lines.append("  PRE:")
        for k, v in pre.items():
            lines.append(f"    {k}: {_unwrap_metric(v)}")

        lines.append("  POST:")
        for k, v in post.items():
            lines.append(f"    {k}: {_unwrap_metric(v)}")

    return "\n".join(lines)


def force_hf_home():
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HOME"] = hf_cache
