# pii_synth/config_and_labels.py
from pathlib import Path
import json

#  DATASET SIZE / SPLIT
N_SAMPLES = 120_000  # total examples (pos + O-only + hard negatives)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # test = 0.1

#  POS/NEG RATIOS
# fraction of records that are all-O (no spans)
O_ONLY_RATIO = 0.50
# fraction that are “hard negatives” (hashes, GUIDs, invalid cards etc.)
HARDNEG_RATIO = 0.15

#  TOKENIZER
TOKENIZER_NAME = "microsoft/deberta-base"
MAX_LENGTH = 512  # Allow variable lengths, don't force all to be same

#  NOISE / TYPO CONFIG (outside spans)
NOISE_CHAR_SUB_PROB = 0.08  # random replacement
NOISE_SWAP_PROB = 0.03  # random adjacent swap
NOISE_CASE_PROB = 0.05  # random upper/lower flip

#  ENTITY LABELS (BILOU)
ENTITY_TYPES = [
    "EMAIL",
    "PHONE",
    "SSN",
    "CREDIT_CARD",
    "PERSON",
    "ORG",
    "ADDRESS",
    "DATE",
    "AGE",
]

LABEL_LIST = ["O"]
for ent in ENTITY_TYPES:
    LABEL_LIST += [f"B-{ent}", f"I-{ent}", f"L-{ent}", f"U-{ent}"]



LABEL2ID = {lab: i for i, lab in enumerate(LABEL_LIST)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}


def save_label_maps(out_dir: str | Path) -> None:
    """
    Save label2id.json / id2label.json in the same format you already use.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, indent=2)

    with open(out_dir / "id2label.json", "w", encoding="utf-8") as f:
        # HF typically stores keys as strings
        json.dump({str(i): lab for i, lab in ID2LABEL.items()}, f, indent=2)
