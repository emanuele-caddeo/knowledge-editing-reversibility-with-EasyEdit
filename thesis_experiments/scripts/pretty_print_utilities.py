from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

def _to_float(x: Any) -> Optional[float]:
    """Best-effort conversion of numeric-like values to float."""
    if x is None:
        return None
    if isinstance(x, np.generic):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def _value_to_str(v: Any) -> str:
    """Convert a metric leaf value into a compact string."""
    f = _to_float(v)
    if f is not None:
        return f"{f:.4f}"

    if isinstance(v, list):
        # If list is numeric -> show mean and length.
        floats = [x for x in (_to_float(xi) for xi in v) if x is not None]
        if floats and len(floats) == len(v):
            mean = sum(floats) / len(floats)
            return f"mean={mean:.4f} (n={len(floats)})"
        # Otherwise, fallback to stringified list.
        return str(v)

    if isinstance(v, dict):
        # Flatten nested dict on a single line.
        parts = []
        for kk, vv in v.items():
            parts.append(f"{kk}={_value_to_str(vv)}")
        return ", ".join(parts) if parts else "{}"

    return str(v)


def metrics_to_str_dict(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Convert a metrics dict into key -> string values."""
    return {str(k): _value_to_str(v) for k, v in (metrics or {}).items()}


def _iter_metric_cases(metrics: Any) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Normalize metrics into a list of (case_label, pre_dict, post_dict).
    Supports:
      - list[{'pre':..., 'post':..., 'case_id':...}]
      - {'pre':..., 'post':...}
      - {'case_0': {'pre':..., 'post':...}, ...}
    """
    if metrics is None:
        return []

    # 1) list of cases
    if isinstance(metrics, list):
        out: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for i, m in enumerate(metrics):
            if isinstance(m, dict):
                label = str(m.get("case_id", m.get("case", i)))
                pre = m.get("pre", {}) if isinstance(m.get("pre", {}), dict) else {}
                post = m.get("post", {}) if isinstance(m.get("post", {}), dict) else {}
                out.append((label, pre, post))
            else:
                out.append((str(i), {}, {}))
        return out

    # 2) single dict with pre/post
    if isinstance(metrics, dict) and "pre" in metrics and "post" in metrics:
        pre = metrics.get("pre", {}) if isinstance(metrics.get("pre", {}), dict) else {}
        post = metrics.get("post", {}) if isinstance(metrics.get("post", {}), dict) else {}
        label = str(metrics.get("case_id", metrics.get("case", 0)))
        return [(label, pre, post)]

    # 3) dict of cases
    if isinstance(metrics, dict):
        out2: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for k, v in metrics.items():
            if isinstance(v, dict) and "pre" in v and "post" in v:
                pre = v.get("pre", {}) if isinstance(v.get("pre", {}), dict) else {}
                post = v.get("post", {}) if isinstance(v.get("post", {}), dict) else {}
                out2.append((str(k), pre, post))
        if out2:
            return out2

    # Fallback: unknown structure
    return [("0", {}, {})]


def print_metrics_table(metrics: Any, *, title: str) -> None:
    """Print PRE vs POST metrics in a readable table."""
    cases = _iter_metric_cases(metrics)
    print(f"\n=== {title} ===")

    if not cases:
        print("(no metrics)")
        return

    for case_label, pre_dict, post_dict in cases:
        pre = metrics_to_str_dict(pre_dict)
        post = metrics_to_str_dict(post_dict)

        keys = sorted(set(pre.keys()) | set(post.keys()))
        rows = [[k, pre.get(k, "-"), post.get(k, "-")] for k in keys]

        print(f"\n[CASE {case_label}]")
        print(tabulate(rows, headers=["metric", "pre", "post"], tablefmt="github"))


def print_hparams_table(hparams: Any) -> None:
    """Print hparams as a table: name | value."""
    print("\n=== HPARAMS ===")

    if hparams is None:
        print("(no hparams)")
        return

    if isinstance(hparams, dict):
        data = hparams
    elif hasattr(hparams, "__dict__"):
        data = vars(hparams)
    else:
        data = {"hparams": str(hparams)}

    rows = [[str(k), str(data[k])] for k in sorted(data.keys())]
    print(tabulate(rows, headers=["name", "value"], tablefmt="github"))


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