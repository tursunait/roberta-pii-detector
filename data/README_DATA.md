
# Synthetic PII Dataset — Usage Guide

This folder contains the **synthetic PII dataset** used for training our DeBERTa-based Named Entity Recognition (NER) model. This documentation will guide you—step by step—on how to:

1. **Set up your environment**
2. **Install dependencies**
3. **Access and load the dataset**
4. **Understand the dataset structure**
5. **Use the dataset for model fine-tuning**

This README is written for teammates who have **never used Hugging Face before**. Just follow the steps in order.


# 1. 🔧 Environment Setup

You may use **Conda** or **pip**.

### Option A — Using Conda

```bash
conda create -n pii python=3.10 -y
conda activate pii
pip install -r requirements.txt
```

### Option B — Using Pip only

```bash
python3 -m venv pii_env
source pii_env/bin/activate
pip install -r requirements.txt
```

# 2. 📄 Install Dependencies

This project uses the following dependencies (already in `requirements.txt`):

```
numpy
pandas
transformers
datasets
tokenizers
accelerate
huggingface_hub
faker
regex
seqeval
torch
tqdm
```

Install everything with:

```bash
pip install -r requirements.txt
```


# 3. 📥 Loading the Dataset From Hugging Face

The dataset is hosted on the HuggingFace Hub at:

**[https://huggingface.co/datasets/tursunait/deberta-pii-synth](https://huggingface.co/datasets/tursunait/deberta-pii-synth)**

You do **not** need to clone the Hugging Face repo manually.
Just load the dataset directly in Python:

```python
from datasets import load_dataset

ds = load_dataset("tursunait/deberta-pii-synth")
train = ds["train"]
val = ds["validation"]
test = ds["test"]

print(train[0])
```

This will automatically download and cache the Arrow files.


# 4. 📂 Dataset Structure

The dataset contains **three splits**:

| Split    | Size          | Purpose                                |
| -------- | ------------- | -------------------------------------- |
| `train/` | ~96k examples | Used for model training                |
| `val/`   | ~12k examples | Used for validation and early stopping |
| `test/`  | ~12k examples | Final evaluation                       |

Each example contains:

```json
{
  "text": "Hi, my name is John Smith and my email is john.smith@gmail.com.",
  "spans": [
    {"start": 18, "end": 28, "label": "PERSON"},
    {"start": 49, "end": 70, "label": "EMAIL"}
  ],
  "tokens": [...],
  "input_ids": [...],
  "attention_mask": [...],
  "labels": [...]
}
```

### Field meanings:

| Field              | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| **text**           | Original input sentence                                     |
| **spans**          | Character-level span annotations (start, end, entity label) |
| **tokens**         | DeBERTa tokenized text                                      |
| **input_ids**      | Token IDs for the model                                     |
| **attention_mask** | Masking for padded tokens                                   |
| **labels**         | Integer NER labels in BILOU format                          |

You do NOT need to regenerate these.
They are pre-tokenized and ready for training.


# 5. Entity Types Included

The synthetic dataset contains **8 PII categories**:

| Label         | Example                                   |
| ------------- | ----------------------------------------- |
| `PERSON`      | "John Smith"                              |
| `EMAIL`       | "[john@gmail.com](mailto:john@gmail.com)" |
| `PHONE`       | "(555)-321-9999"                          |
| `ADDRESS`     | "123 Main Street"                         |
| `CREDIT_CARD` | "3434 9923 2333 4455"                     |
| `SSN`         | "123-45-6789"                             |
| `ORG`         | "Acme Corp"                               |
| `DATE`        | "2020-09-01"                              |

The BILOU scheme is used:

* **B-XXX** = beginning of entity
* **I-XXX** = inside of entity
* **L-XXX** = last token
* **U-XXX** = unit-length entity
* **O** = outside any entity


# 6. How to Use This Dataset for DeBERTa Fine-Tuning

Below is the minimal working script your teammates can use.

### Step 1 — Load model + tokenizer

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tursunait/deberta-pii-synth")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=NUM_LABELS
)
```

### Step 2 — Define training args

```python
args = TrainingArguments(
    output_dir="pii-ner-model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)
```

### Step 3 — Create Trainer

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)
```

### Step 4 — Train the model

```python
trainer.train()
```


# 7. Summary

You now have:

* A clean dataset of synthetic PII
* Full ready-to-train Arrow files
* A Hugging Face dataset you can load in one line
* BILOU-tokenized labels
* Everything required to fine-tune a DeBERTa token-classification model

You only need to:

1. Install requirements
2. Load dataset via `load_dataset("tursunait/deberta-pii-synth")`
3. Train the model using HuggingFace Trainer


# 8. If You Need Help

📘 **Hugging Face Datasets Docs**:
[https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)

📘 **Token Classification Tutorial**:
[https://huggingface.co/docs/transformers/tasks/token_classification](https://huggingface.co/docs/transformers/tasks/token_classification)

