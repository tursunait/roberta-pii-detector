# pii_synth/build_datasets.py
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .config_and_labels import (
    TOKENIZER_NAME,
    MAX_LENGTH,
    TRAIN_RATIO,
    VAL_RATIO,
    LABEL2ID,
)
from .config_and_labels import save_label_maps


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def spans_to_token_labels(
    text: str,
    spans: List[Dict[str, Any]],
    tokenizer,
) -> Dict[str, Any]:
    """
    Convert character-level spans to token-level BILOU labels.

    Works with DeBERTa tokenizer using offsets_mapping.
    """
    encoded = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets: List[Tuple[int, int]] = encoded["offset_mapping"]

    # build token spans -> entity indices
    # each entity span: {'start': int, 'end': int, 'label': 'EMAIL'|...}
    token_labels = ["O"] * len(input_ids)

    # map tokens to entity index
    for ent in spans:
        e_start = ent["start"]
        e_end = ent["end"]
        ent_label = ent["label"]  # e.g., "EMAIL"

        # collect token indices overlapping this char span
        ent_token_indices: List[int] = []
        for tidx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:
                continue  # special tokens
            if tok_end <= e_start:
                continue
            if tok_start >= e_end:
                continue
            ent_token_indices.append(tidx)

        if not ent_token_indices:
            continue

        if len(ent_token_indices) == 1:
            t = ent_token_indices[0]
            token_labels[t] = f"U-{ent_label}"
        else:
            first = ent_token_indices[0]
            last = ent_token_indices[-1]
            token_labels[first] = f"B-{ent_label}"
            token_labels[last] = f"L-{ent_label}"
            for t in ent_token_indices[1:-1]:
                token_labels[t] = f"I-{ent_label}"

    # convert label strings to ids, using -100 for special tokens to ignore in loss
    label_ids: List[int] = []
    for tidx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end == 0:
            # [CLS], [SEP], padding etc.
            label_ids.append(-100)
        else:
            lab = token_labels[tidx]
            label_ids.append(LABEL2ID.get(lab, LABEL2ID["O"]))

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return {
        "text": text,
        "spans": spans,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_ids,
        "tokens": tokens,
    }


def build_and_save_datasets(
    raw_jsonl: str | Path,
    out_dir: str | Path,
    seed: int = 42,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    records = load_jsonl(raw_jsonl)

    hf_records = [
        spans_to_token_labels(rec["text"], rec["spans"], tokenizer) for rec in records
    ]

    ds = Dataset.from_list(hf_records).shuffle(seed=seed)

    n = len(ds)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    n_test = n - n_train - n_val

    train_ds = ds.select(range(0, n_train))
    val_ds = ds.select(range(n_train, n_train + n_val))
    test_ds = ds.select(range(n_train + n_val, n_train + n_val + n_test))

    dsd = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    dsd["train"].save_to_disk(out_dir / "train")
    dsd["val"].save_to_disk(out_dir / "val")
    dsd["test"].save_to_disk(out_dir / "test")

    # write label2id.json / id2label.json like you already have
    save_label_maps(out_dir)
