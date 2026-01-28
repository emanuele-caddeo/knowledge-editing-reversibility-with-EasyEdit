import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class KnowEditSample:
    case_id: int
    prompt: str
    subject: str
    ground_truth: str
    target_new: str


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_knowedit_records(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"KnowEdit file not found: {path}")

    data = _read_json(path)
    if isinstance(data, dict):
        # Some datasets store records under a key like "data" or similar.
        # Try common fallbacks.
        for k in ("data", "records", "examples"):
            if k in data and isinstance(data[k], list):
                return data[k]
        raise ValueError("Unsupported KnowEdit JSON format: dict without a records list.")
    if isinstance(data, list):
        return data

    raise ValueError("Unsupported KnowEdit JSON format: expected list or dict.")


def _get_first_existing(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_knowedit_record(rec: Dict[str, Any], *, case_id: int) -> KnowEditSample:
    """
    Tries to support multiple common field namings.
    Expected final fields: prompt, subject, ground_truth, target_new.
    """
    # Direct-style fields
    prompt = _get_first_existing(rec, ["prompt", "src", "query", "input", "request_prompt"])
    subject = _get_first_existing(rec, ["subject", "entity", "topic"])
    ground_truth = _get_first_existing(rec, ["ground_truth", "target_true", "answer", "gold"])
    target_new = _get_first_existing(rec, ["target_new", "target", "new_answer", "edit_target"])

    # EasyEdit-style nested field (often returned in metrics)
    rr = rec.get("requested_rewrite")
    if isinstance(rr, dict):
        prompt = prompt or rr.get("prompt")
        subject = subject or rr.get("subject")
        ground_truth = ground_truth or rr.get("ground_truth")
        target_new = target_new or rr.get("target_new")

    missing = [k for k, v in {
        "prompt": prompt,
        "subject": subject,
        "ground_truth": ground_truth,
        "target_new": target_new,
    }.items() if not v]

    if missing:
        raise ValueError(f"KnowEdit record missing fields {missing}. Record keys: {list(rec.keys())}")

    return KnowEditSample(
        case_id=case_id,
        prompt=str(prompt),
        subject=str(subject),
        ground_truth=str(ground_truth),
        target_new=str(target_new),
    )


def load_one_sample(dataset_path: str, index: int) -> KnowEditSample:
    records = load_knowedit_records(dataset_path)
    if not (0 <= index < len(records)):
        raise IndexError(f"Index out of range: {index} (dataset size={len(records)})")

    rec = records[index]
    if not isinstance(rec, dict):
        raise ValueError("KnowEdit record must be a dict/object.")
    return normalize_knowedit_record(rec, case_id=index)
