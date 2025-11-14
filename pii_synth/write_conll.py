# pii_synth/write_conll.py
from pathlib import Path
from datasets import load_from_disk

from .config_and_labels import ID2LABEL


def write_conll(split_dir: str | Path, out_path: str | Path) -> None:
    """
    Convert HF dataset split (with 'tokens' and 'labels') into CoNLL format:
    token<TAB>LABEL lines, blank line between examples.
    """
    split_dir = Path(split_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(split_dir))

    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            tokens = ex["tokens"]
            labels = ex["labels"]
            for tok, lid in zip(tokens, labels):
                if lid == -100:
                    # skip padding/special tokens in CoNLL
                    continue
                label = ID2LABEL[lid]
                f.write(f"{tok}\t{label}\n")
            f.write("\n")
