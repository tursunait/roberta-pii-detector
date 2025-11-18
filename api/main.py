# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

app = FastAPI()

MODEL_NAME = "models/pii-deberta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = model.config.id2label  # already aligned with your label2id.json


class PiiRequest(BaseModel):
    text: str


class PiiSpan(BaseModel):
    start: int
    end: int
    label: str


class PiiResponse(BaseModel):
    redacted_text: str
    spans: list[PiiSpan]


def redact_text(text: str, spans: list[PiiSpan], mask_style="TYPE"):
    # spans are character-based, non-overlapping
    chars = list(text)
    for span in spans:
        if mask_style == "TYPE":
            replacement = f"[{span.label}]"
        else:
            replacement = "*" * (span.end - span.start)
        chars[span.start : span.end] = list(replacement)
    return "".join(chars)


@app.post("/mask", response_model=PiiResponse)
def mask_pii(req: PiiRequest):
    enc = tokenizer(
        req.text, return_offsets_mapping=True, return_tensors="pt", truncation=True
    )
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits  # [1, seq_len, num_labels]
        preds = logits.argmax(dim=-1)[0].tolist()

    offsets = enc["offset_mapping"][0].tolist()
    spans = []
    current = None

    for idx, (label_id, (start, end)) in enumerate(zip(preds, offsets)):
        label = LABELS[str(label_id)] if isinstance(LABELS, dict) else LABELS[label_id]
        if label == "O" or start == end:
            if current:
                spans.append(current)
                current = None
            continue

        # Example: "B-EMAIL", "I-EMAIL", "L-EMAIL"
        tag, ent_type = label.split("-", 1)

        if tag in ("B", "U"):
            if current:
                spans.append(current)
            current = {"start": start, "end": end, "label": ent_type}
        elif tag in ("I", "L") and current and current["label"] == ent_type:
            current["end"] = end
            if tag == "L":
                spans.append(current)
                current = None
        else:
            if current:
                spans.append(current)
            current = None

    if current:
        spans.append(current)

    pii_spans = [PiiSpan(**s) for s in spans]
    redacted = redact_text(req.text, pii_spans)
    return PiiResponse(redacted_text=redacted, spans=pii_spans)
