## Synthetic PII Data Generation Pipeline

This document describes how our synthetic PII dataset is generated, how the codebase is organized, and how teammates can reproduce or extend the dataset before moving on to **fine-tuning DeBERTa-base for Named Entity Recognition**.

**Currently there are 120 000 raw sentances generated.** 


## 1. Overview

Because we cannot use real PII (emails, phone numbers, SSNs, credit cards, addresses, names), we generate a fully synthetic dataset using:

* **Faker (`en_US`)** for English-only realistic entities
* **Template-based text generation** (e.g., “You can reach {person} at {email}”)
* **Span-level supervision** (`start`, `end`, `label`)
* **Token-level BILOU labels** generated using the DeBERTa tokenizer
* **Noise injection** **only outside PII spans** to improve robustness
* **Hard negatives** (IBANs, GUIDs, MAC addresses, random IDs) to prevent overfitting
* **HuggingFace datasets** for saving train/val/test splits

The goal is to deliver a **model-ready dataset** that can be directly loaded into a HuggingFace `Trainer`.



## 2. Directory Structure

```
deberta-pii-detector/
│
├── synth_data.py                    # Entry point to run the generator
│
├── pii_synth/
│   ├── generation.py                # Builds synthetic examples + noise + hard negatives
│   ├── patterns.py                  # Regex utilities for PII and hard negatives
│   ├── noise.py                     # Typo/keyboard/case/swap noise (span-safe)
│   ├── templates.py                 # English template sentences
│   ├── utils.py                     # Span conversion and token-label mapping
│
└── data/
    ├── raw/
    │   └── pii.jsonl                # Un-tokenized synthetic examples
    │
    ├── processed/
        ├── train/                   # HF dataset split
        ├── val/
        ├── test/
        ├── label2id.json            # BILOU → ID map
        ├── id2label.json            # ID → BILOU map
        └── *.conll                  # for debugging token tags
```



## 3. How to Generate a New Dataset

From the project root:

```bash
python synth_data.py
```

This runs the entire pipeline:

1. **Build raw examples** → `data/raw/pii.jsonl`
2. **Apply noise** to non-PII text
3. **Tokenize with DeBERTa-base**
4. **Map tokens to BILOU IDs**
5. **Split into train/val/test**
6. **Save HuggingFace datasets** to `data/processed/*`
7. **Export debug CoNLL files**

### If you want to silence HuggingFace tokenizer warnings:

```bash
export TOKENIZERS_PARALLELISM=false
python synth_data.py
```


## 4. What the Dataset Contains

Each example in the HF dataset contains:

```python
{
  "text": "... original text ...",
  "spans": [
      {"start": 14, "end": 25, "label": "PERSON"},
      {"start": 48, "end": 67, "label": "EMAIL"},
      ...
  ],
  "tokens": [...],         # decode/check only
  "input_ids": [...],      
  "attention_mask": [...],
  "labels": [...]          # BILOU IDs, -100 for padding
}
```

### PII classes

We support 8 entity types:

```
PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, ADDRESS, ORG, DATE
```

Each is expanded into BILOU tags (e.g., `B-PERSON`, `I-PERSON`, `L-PERSON`, `U-PERSON`).

### Hard negatives

We include **non-PII strings that look like PII**:

* random numbers
* IBANs
* MAC addresses
* UUIDs
* ticket/order IDs
* partial phone-like sequences
* email-like patterns without domain
* invalid credit card patterns

All receive label `O` to ensure the model does not overfit to formats.

### Noise (applied only outside PII spans)

* random character substitutions
* case flips
* swaps of adjacent characters
* controlled typos

**We do not modify spans**, ensuring their indices remain valid.



## 5. How to Load the Dataset for Fine-Tuning

From any training script or notebook:

```python
from datasets import load_from_disk

train_ds = load_from_disk("data/processed/train")
val_ds   = load_from_disk("data/processed/val")
test_ds  = load_from_disk("data/processed/test")

print(train_ds[0])  
print(train_ds.features)
```

Then load the label maps:

```python
import json

with open("data/processed/label2id.json") as f:
    label2id = json.load(f)
with open("data/processed/id2label.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
```

Create the model:

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/deberta-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)
```

Now you can train using HuggingFace `Trainer`.


## 6. Outputs You Will Use in Fine-Tuning

Your **only need these** for fine-tuning:

```
data/processed/train/
data/processed/val/
data/processed/test/
data/processed/label2id.json
data/processed/id2label.json
```

Everything else is for generation/debugging only.



## 7. Regeneration Guidelines

When *should* we regenerate the dataset:

* We add new templates
* We add new PII types
* We tune noise or hard-negatives
* We want more variety or more samples

We **should NOT** regenerate if:

* We already trained a model and want reproducibility
* We are debugging training hyperparameters
* We changed model settings but not the data pipeline





