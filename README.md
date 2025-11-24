# RoBERTa PII Detector 

## Overview
Accidental disclosure of personally identifiable information (PII) is a growing privacy risk in human–AI text interactions. This project builds an end-to-end pipeline to **detect and mask sensitive entities** in free-form text using transformer-based Named Entity Recognition (NER).

We fine-tuned **RoBERTa-base** for token classification using a **BILOU tagging scheme** to detect eight PII types:

`EMAIL, PHONE, SSN, CREDIT_CARD, PERSON, ORG, ADDRESS, DATE`

Because real PII cannot be used for training, we generate a large **synthetic, privacy-safe dataset** using Faker + regex templates + noise injection. We evaluate the model on both synthetic ground-truth sentences and real Reddit posts.

---

## Repository Structure
```

data/                           # (optional) data storage directory
evaluation/                     # real-world evaluation pipeline (synthetic + Reddit)
model/                          # saved fine-tuned model artifacts
pii_synth/                      # synthetic data generation utilities

synth_data.py                   # main synthetic data generation script
synth_checks.ipynb              # notebook for sanity checks / exploration
requirements.txt                # dependencies

reddit_manual_annotation.json   # manual annotation template for Reddit samples
stage3_complete_evaluation.json # saved evaluation report (outputs)
LICENSE

````

**Notes**
- Large model files are stored under `model/`.
- Evaluation outputs are saved as JSON for reproducibility.

---

## Synthetic Data Generation
We generate training data programmatically to ensure privacy compliance.

**What’s generated**
- Faker-based entities: names, emails, phone numbers, dates, orgs, addresses  
- Regex-based structured IDs: SSNs, credit cards, etc.  
- Optional noise: typos, spacing errors, character swaps  

**Run generation**
```bash
python synth_data.py
````

This produces a synthetic corpus (used in training) and stores clean spans for labeling.

---

## Model Training (RoBERTa-base)

RoBERTa-base is fine-tuned for token classification with BILOU labels.

Key training settings (as used in report):

* Learning rate: `2e-5`
* Epochs: `2`
* Batch size: `8` (gradient accumulation `4`)
* Max sequence length: `256`
* Labels: 33 BILOU tags across 8 PII types

Saved model checkpoints are in:

```
model/my_trained_pii_model/
```

---

## Evaluation Pipeline

Evaluation is two-stage:

1. **Synthetic ground-truth test set (quantitative)**

   * 5 hand-crafted sentences with known spans
   * Strict exact-span precision/recall/F1

2. **Reddit samples (real-world qualitative)**

   * Scraped from: `relationships`, `personalfinance`, `jobs`, `askreddit`
   * 15 samples evaluated in inference mode
   * Manual annotation scaffold provided for future real-world metrics

**Run evaluation**

```bash
python evaluation/model_evaluation.py
```

Outputs:

* `stage3_complete_evaluation.json`
* `reddit_manual_annotation.json`

---

## Results Snapshot

From the current evaluation run:

**Synthetic (exact-span)**

* Precision: **0.000**
* Recall: **0.000**
* F1: **0.000**

This is due to **boundary truncation** (type often correct, span slightly mismatched).

**Reddit**

* 3/15 samples had detected PII (≈20%)
* Detected types: DATE (2), EMAIL (1)

**Latency**

* Avg inference time: **~24 ms per text**

---

## Known Limitations

* Synthetic test set is very small (5 examples), so aggregate metrics are unstable.
* Exact-span scoring penalizes partial but privacy-useful detections.
* Reddit samples are not fully manually labeled yet.

---

## Next Steps

* Complete manual annotation of Reddit samples to compute real-world F1.
* Add more noisy/obfuscated structured PII during training to reduce truncation.
* Explore relaxed matching metrics (IoU / overlap) aligned with privacy utility.
* Consider distillation/quantization for browser deployment.
  
--- 
## License

MIT License. See `LICENSE`.
