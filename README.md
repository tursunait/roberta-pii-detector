# **PII-Guardian: Robust Synthetic PII Detection with RoBERTa**

## **Overview**

Accidental disclosure of personally identifiable information (PII) remains a major privacy risk in human–AI interactions, especially when users transmit text to chatbots or LLMs.
This project builds an end-to-end pipeline for **detecting and masking sensitive entities** in free-form text using transformer-based Named Entity Recognition (NER).

We fine-tune a transformer model (RoBERTa-base) with a **BILOU tagging scheme** to detect **nine** PII categories:

`PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, ORG, ADDRESS, DATE, AGE`

Because real PII cannot be used, all training data is produced by a **rich synthetic generation pipeline** that simulates real-world writing, noise, obfuscation patterns, and near-miss negatives.

---

## **Key Features**

### **✔ Realistic variable-length documents**

* Short (1 sentence), medium (paragraph), long (multi-template posts)
* Automatic merging of spans across concatenated templates

### **✔ Deep formatting diversity**

Includes dozens of patterns for:

* Usernames (`firstname.lastname`, initials, numeric handles)
* International phone formats (+31 880 385 2406; dotted, spaced, mixed)
* SSNs in dashed/space/dot patterns
* Email obfuscation: `john[at]gmail[dot]com`, `(at)`, `"john @ gmail . com"`

### **✔ Noise injection (inside + outside PII spans)**

* Keyboard-neighbor typos
* Case flipping
* Character swaps
* In-span realistic corruption:

  * `gmail → gmial`,
  * spacing distortions (`555-1234 → 555 - 1234`)
  * casing variations

### **✔ Hard negatives**

Strings that *look* like PII but must be labeled **O**:

* GUIDs, MAC addresses, SHA1 hashes
* Invalid credit cards (missing checksum)
* Random numeric sequences
* Social handles (`@username`)

### **✔ Entity type: AGE**

Examples:
`23M`, `age 24`, `24-year-old`, `"I'm 25"`, `(25F)`.

### **✔ Reddit-style, email-style, CSV-style, legal/medical templates**

Over 100+ templates covering:

* casual conversations
* scams / support messages
* shipping and billing
* medical records
* logs and CSV rows
* messy, incomplete text

---

## **Repository Structure**

```
data/
  raw/                      # synthetic pii.jsonl
  processed/                # HF Arrow datasets + CoNLL exports

evaluation/                 # real-world evaluation scripts and reports
model/                      # saved model checkpoints
pii_synth/                  # full synthetic generator & processing pipeline
  generation.py             # NEW 2025 generator (PII sampling, noise, templates)
  build_datasets.py         # tokenization + label alignment
  write_conll.py            # CoNLL export
  config_and_labels.py      # hyperparameters, tag mappings
synth_data.py               # orchestration pipeline
requirements.txt

reddit_manual_annotation.json
stage3_complete_evaluation.json
LICENSE
```

---

## **Synthetic Data Generation**

The dataset is produced entirely programmatically to ensure privacy compliance.

### **What’s generated**

* Naturalistic text of varying length
* PII entities: PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, ORG, ADDRESS, DATE, AGE
* Obfuscated email/phone variants
* Hard negatives + O-only “safe” samples
* Character-level noise both inside and outside spans

### **Generate dataset**

```bash
python synth_data.py
```

This produces:

* `data/raw/pii.jsonl`
* HF tokenized dataset → `data/processed/`
* CoNLL files for debugging
  
---

# **Model Architecture & Training**

All model training for this project is performed using **RoBERTa-base** fine-tuned for token classification. The training pipeline is defined in `model.ipynb`, and all exported model artifacts are stored in:

```
trained_model/
    config.json
    vocab.json
    merges.txt
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
```

These files allow the model and tokenizer to be loaded directly through Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("trained_model/")
model = AutoModelForTokenClassification.from_pretrained("trained_model/")
```

### **Why RoBERTa-base?**

RoBERTa-base was selected because it:

* Performs strongly on token classification & NER
* Handles noisy, informal text better than rule-based approaches
* Offers a good performance–cost balance for limited GPU budgets
* Has strong contextual reasoning needed for messy/obfuscated PII formats

### **Labeling Scheme**

The model predicts **33 BILOU tags** corresponding to the following eight PII categories:

`EMAIL, PHONE, SSN, CREDIT_CARD, PERSON, ORG, ADDRESS, DATE`

### **Training Configuration**

| Hyperparameter        | Value  | Rationale                                                      |
| --------------------- | ------ | -------------------------------------------------------------- |
| Learning rate         | `2e-5` | Standard for transformer fine-tuning, stable on small datasets |
| Epochs                | 2      | Avoid overfitting on 6,000 training samples                    |
| Batch size            | 8      | GPU memory constraints                                         |
| Gradient accumulation | 4      | Simulates batch size of 32                                     |
| Max sequence length   | 256    | Matches typical user-generated text length                     |

Training was performed on a single GPU and completed in ~30 minutes.

### **Dataset Used for Training**

Due to compute limits, the model was trained on a **6,000-example subset** of the full 96k synthetic dataset originally generated.
Data generation components live in:

```
pii_synth/
    generation.py
    build_datasets.py
    config_and_labels.py
    write_conll.py
    synth_data.py
```

The synthetic dataset uses:

* Faker-generated identifiers
* Regex-constructed structured numbers
* Template-based text
* Noise injection (typos, spacing distortions, digit swaps)
* Hard negatives (UUIDs, hashes, non-PII numeric strings)

---

# **Evaluation Pipeline**

Evaluation scripts and results are located in:

```
evaluation/
    model_evaluation.py
    evaluations_results.json
```

`model_evaluation.py` runs **both synthetic** and **real-world** tests and produces:

* Entity-level precision, recall, and F1
* Over-prediction and under-prediction statistics
* Boundary-accuracy diagnostics

The final evaluation metrics (synthetic & real-world) used in the report were computed using this script.

---

## **1. Synthetic Evaluation**

The synthetic test set contains ~600 examples generated by the same pipeline as training data.

Key findings:

* Overall accuracy: **99.85%**
* Weighted average F1: **99.04%**
* Strong performance on EMAIL, DATE, CREDIT_CARD, ORG
* Weaker performance on ADDRESS, AGE, SSN due to span variability

Synthetic results confirm that the model successfully learns the patterns present in the synthetic corpus, but these numbers **do not** reflect performance in real-world environments due to domain shift.

---

## **2. Real-World Evaluation (ai4privacy Dataset)**

To measure real-world generalization, we used **600 real Reddit posts** from the
`ai4privacy/pii-masking-300k` dataset.

Because this dataset uses **BIO** tags, we mapped them to **BILOU** during evaluation.

### **Initial Performance**

* Precision: **17%**
* Recall: **21%**
* F1-score: **19%**
* Model predicted **5× too many entities**

This showed that the model was extremely over-aggressive on noisy human text.

### **Refined Results After Improving Synthetic Data**

After expanding synthetic templates and adding partial-address patterns:

* Precision: **28%**
* Recall: **45%**
* F1-score: **37%**

Despite improvements, the model still predicted **6.5× more entities** than actually existed.

### **Entity-Level F1**

| Entity  | F1   |
| ------- | ---- |
| EMAIL   | 0.89 |
| DATE    | 0.55 |
| PHONE   | 0.35 |
| PERSON  | 0.33 |
| SSN     | 0.28 |
| ADDRESS | 0.28 |

### **Observed Error Modes**

* **Over-prediction**: dominant failure mode; O-tag discrimination is weak
* **Boundary errors**: incomplete spans for PHONE, SSN, or credit cards
* **Address mismatch**: real data uses address *components*, synthetic uses *full addresses*
* **Domain shift**: real-world text contains context clues & semantics missing from synthetic data

---

## **3. Inference Latency**

Average latency (CPU):

* **24.45 ms** per sequence (max ~256 tokens)

Fast enough for **near-real-time masking** in chat interfaces or preprocessing pipelines.

---

## **Limitations**

* Training used only **6k** examples due to compute limits
* Over-prediction caused by class imbalance and lack of real-world negatives
* Synthetic data did not fully match real-world distribution
* Limited hyperparameter exploration
* Span detection remains challenging for structured identifiers

---

## **Future Directions**

* Train on full 96k–120k synthetic dataset
* Introduce class-weighted or focal loss to reduce over-prediction
* Improve synthetic generator with more fine-grained address/components
* Add adversarial obfuscations and contextual distractors
* Distill model for browser/in-device deployment
* Human-in-the-loop feedback loops for iterative correction
