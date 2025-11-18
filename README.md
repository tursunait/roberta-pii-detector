
# PII NER Synthetic Dataset (BILOU, DeBERTa-ready) — Extended

**Entities:** EMAIL, PHONE, SSN, CREDIT_CARD, PERSON, ORG, ADDRESS, DATE

**Locale mix:** en_US:1.00

**Splits:**
- Train: 64
- Val:   8
- Test:  8

**Generation:** Faker (multilingual) + custom generators (email/phone/SSN/credit-card with Luhn) + multilingual templates + rigorous hard negatives.
**Augmentation (non-entity only, length-preserving):** keyboard-neighbor subs, adjacent swaps, case jitter, diacritic strip, homoglyph subs.
**Labels:** BILOU + O. See `label2id.json` & `id2label.json`.

**Tokenizer:** `microsoft/deberta-base`  
**Max length:** 256

**Files:**
- `data/raw/pii.jsonl` — source (text + character spans)
- `data/processed/{train,val,test}` — HF datasets (`input_ids`, `attention_mask`, `labels`)
- `data/processed/pii.(train|val|test).conll` — CoNLL
- `data/processed/label2id.json`, `data/processed/id2label.json`

Notes:
- Special tokens & padding use `-100` labels (ignored by loss).
- Noise is applied only outside gold spans and preserves length, keeping offsets valid.
