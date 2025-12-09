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

## **Model Training (RoBERTa)**

A transformer model is fine-tuned for token classification with **BILOU labels**.

### Training configuration

* Learning rate: `2e-5`
* Epochs: `2–3`
* Batch size: `8` (with gradient accumulation)
* Max sequence length: `256–512`
* Total labels: **B-I-L-U-O for 9 entity classes**

### Checkpoints

```
model/my_trained_pii_model/
```

---

## **Evaluation Pipeline**

Evaluation includes both **quantitative** and **qualitative** components.

### **1. Synthetic evaluation (large test set)**

* Exact-span and entity-level metrics
* Measures robustness against noise and obfuscations

### **2. Real-world Reddit evaluation**

* Posts from various subreddits: relationships, jobs, personalfinance, etc.
* Model used in inference mode
* Manually annotated spans provided to allow the computation of real-world precision/recall

### **Run evaluation**

```bash
python evaluation/model_evaluation.py
```

Outputs saved as:

* `stage3_complete_evaluation.json`
* `reddit_manual_annotation.json`

---

## **Results Snapshot**

*(Example numbers—replace with final results after retraining)*

### **Synthetic evaluation**

Models generally perform well on synthetic data but reveal:

* Boundary sensitivity under heavy in-span corruption
* Strong performance on EMAIL, PHONE, CREDIT_CARD
* More difficulty with ADDRESS and ORG (high variety)

### **Reddit qualitative evaluation**

The model:

* Identifies obfuscated and noisy PII in a subset of posts
* Exhibits lower recall due to novel templates and multi-sentence context
* Shows promising generalization despite training purely on synthetic text

### **Latency**

~20–30 ms per text on CPU
(<5 ms on GPU)

---

## **Known Limitations**

* Synthetic distribution may not fully match real-world logs
* Exact-span scoring is harsh for privacy tasks
* No multi-sentence coreference resolution
* Very extreme obfuscations may still be missed

---

## **Next Steps**

* Expand template library with adversarial examples
* Add IoU / token-overlap metrics to measure “practical privacy protection”
* Distill the model into an on-device browser-ready version
* Integrate with a Chrome/Firefox extension for real-time masking
* Explore multilingual synthetic generation

---

## **License**

MIT License.
See `LICENSE`.

