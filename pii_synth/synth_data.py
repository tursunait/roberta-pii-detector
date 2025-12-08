# generate_data.py
from pathlib import Path

from .config_and_labels import N_SAMPLES
from .generation import generate_jsonl
from .build_datasets import build_and_save_datasets
from .write_conll import write_conll


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    raw_jsonl = RAW_DIR / "pii.jsonl"

    # 1. Generate synthetic English-only PII JSONL (pos + O-only + hard negatives)
    generate_jsonl(raw_jsonl, n_samples=N_SAMPLES)

    # 2. Convert to HF datasets (Arrow) with text, spans, input_ids, attention_mask, labels, tokens
    build_and_save_datasets(raw_jsonl, PROC_DIR)

    # 3. CoNLL exports – these should look like your pii.train/val/test.conll
    write_conll(PROC_DIR / "train", PROC_DIR / "pii.train.conll")
    write_conll(PROC_DIR / "val", PROC_DIR / "pii.val.conll")
    write_conll(PROC_DIR / "test", PROC_DIR / "pii.test.conll")


if __name__ == "__main__":
    main()
