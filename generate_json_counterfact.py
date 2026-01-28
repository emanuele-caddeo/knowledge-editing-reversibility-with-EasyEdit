# export_counterfact_to_local_json.py
from pathlib import Path
import json
from datasets import load_dataset

# Must match your experiment config:
HF_DATASET = "azhx/counterfact"
HF_SPLIT = "test"

# Local path you will put in the experiment YAML as:
# exp_dataset_path: data_/counterfact/test.json
OUTPUT_PATH = Path("thesis_experiments/data_/counterfact/test.json")

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from HF: {HF_DATASET} (split={HF_SPLIT})")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    print(f"Loaded {len(ds)} rows")

    records = [dict(r) for r in ds]

    print(f"Writing JSON to: {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
