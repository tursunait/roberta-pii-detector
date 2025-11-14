# pii_synth/generation.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import random
import string

from faker import Faker

from .config_and_labels import (
    O_ONLY_RATIO,
    HARDNEG_RATIO,
    NOISE_CHAR_SUB_PROB,
    NOISE_SWAP_PROB,
    NOISE_CASE_PROB,
)

# English-only Faker
FAKE = Faker("en_US")


@dataclass
class PIIExample:
    text: str
    spans: List[
        Dict[str, Any]
    ]  # each span: {"start": int, "end": int, "label": "EMAIL"|...}


#  FIELD SAMPLING (ENGLISH)


def sample_fields(fake: Faker) -> Dict[str, str]:
    """
    Generate one instance of each field type. These are then plugged into templates.
    """
    return {
        "person": fake.name(),
        "org": fake.company(),
        "address": fake.address().replace("\n", ", "),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "ssn": fake.ssn(),
        "credit_card": fake.credit_card_number(card_type=None),
        "date": fake.date(),  # YYYY-MM-DD
    }


# Templates loosely match the patterns you already have in your .conll
TEMPLATES = [
    "Contact {person} at {email} or {phone}.",
    "{person} from {org} used card {credit_card} on {date}.",
    "SSN: {ssn}; Phone: {phone}; Email: {email}.",
    "Ship to {address} for {person} from {org} by {date}.",
    "Billing card {credit_card} was charged on {date}.",
    "You can reach {person} ({org}) via {email}.",
    "Meeting on {date}. Call {phone} if late.",
    "Invoice to {org}, attention {person}, address {address}.",
]


def fill_template(template: str, fake: Faker) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """
    Replace {field} in template with Faker values and return:
      text: full string
      spans: dict field_name -> (start_char, end_char)
    """
    fields = sample_fields(fake)
    text = ""
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    i = 0
    while i < len(template):
        if template[i] == "{" and "}" in template[i:]:
            j = template.index("}", i)
            key = template[i + 1 : j]
            value = fields[key]
            start = cursor
            text += value
            end = cursor + len(value)
            spans[key] = (start, end)
            cursor = end
            i = j + 1
        else:
            text += template[i]
            cursor += 1
            i += 1

    return text, spans


#  NOISE / TYPOS (OUTSIDE SPANS)

KEYBOARD_NEIGHBORS = {
    "a": "qs",
    "s": "qweadz",
    "d": "ersfxc",
    "f": "rtdgcv",
    "g": "tyfhbv",
    "h": "yugjbn",
    "j": "uikhmn",
    "k": "ioljm",
    "l": "opk",
}


def random_neighbor(c: str) -> str:
    lower = c.lower()
    if lower in KEYBOARD_NEIGHBORS and random.random() < 0.7:
        repl = random.choice(KEYBOARD_NEIGHBORS[lower])
        return repl.upper() if c.isupper() else repl
    # fallback random letter/digit/punct
    pool = string.ascii_letters + string.digits + " .,-_"
    repl = random.choice(pool)
    return repl


def apply_noise_outside_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """
    Apply character-level noise only on characters not covered by any PII span.
    IMPORTANT: We only use *length-preserving* noise here (substitution, case flip,
    optional swap) so that character offsets for spans remain valid.

    - random substitution (keyboard neighbor or random char)
    - case flip
    - swap with next character (also outside spans)
    """
    n = len(text)
    if n == 0:
        return text

    # Mark which character indices are inside PII spans
    protected = [False] * n
    for s, e in spans:
        for i in range(max(0, s), min(n, e)):
            protected[i] = True

    chars = list(text)
    i = 0
    while i < len(chars):
        # safety: keep in sync with protected
        if i >= len(protected):
            break

        if protected[i]:
            i += 1
            continue

        # substitution (length-preserving)
        if random.random() < NOISE_CHAR_SUB_PROB:
            chars[i] = random_neighbor(chars[i])

        # case flip
        if random.random() < NOISE_CASE_PROB:
            c = chars[i]
            if c.isalpha():
                chars[i] = c.upper() if c.islower() else c.lower()

        # optional swap with next outside-span character
        if (
            random.random() < NOISE_SWAP_PROB
            and i + 1 < len(chars)
            and not protected[i + 1]
        ):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
            continue

        i += 1

    return "".join(chars)


#  HARD NEGATIVES (HASHES, GUID, INVALID CARDS, HANDLES, ETC.)


def hard_negative_strings(fake: Faker) -> List[str]:
    """
    Things that look sensitive but we want as O labels.
    """
    valid_card = fake.credit_card_number()
    invalid_card = valid_card[:-1]  # missing digit

    candidates = [
        f"GUID {fake.uuid4()}",
        f"MAC {fake.mac_address()}",
        f"SHA1 {fake.sha1()}",
        f"IPv4 {fake.ipv4()}",
        f"Card {invalid_card} (missing digit)",
        f"Handle @{fake.user_name()}",
        f"Ref #{fake.random_int(10000, 99999)}",
        f"Acct {fake.random_int(10_000_000, 99_999_999)} checksum pending",
        f"public: MAC {fake.mac_address()}",
        f"professor: SHA1 {fake.sha1()}",
    ]
    return candidates


def sample_hard_negative(fake: Faker) -> str:
    return random.choice(hard_negative_strings(fake))


#  EXAMPLE BUILDERS


def build_positive_example(fake: Faker) -> PIIExample:
    template = fake.random_element(TEMPLATES)
    text, field_spans = fill_template(template, fake)

    field2label = {
        "person": "PERSON",
        "org": "ORG",
        "address": "ADDRESS",
        "email": "EMAIL",
        "phone": "PHONE",
        "ssn": "SSN",
        "credit_card": "CREDIT_CARD",
        "date": "DATE",
    }

    spans: List[Dict[str, Any]] = []
    char_spans: List[Tuple[int, int]] = []

    for field, (s, e) in field_spans.items():
        label = field2label[field]
        spans.append({"start": s, "end": e, "label": label})
        char_spans.append((s, e))

    noisy_text = apply_noise_outside_spans(text, char_spans)
    return PIIExample(text=noisy_text, spans=spans)


def build_o_only_example(fake: Faker) -> PIIExample:
    """
    All tokens O – for calibration of precision.
    """
    # Roughly similar to your current random “Ref #” style blobs
    text = fake.paragraph(nb_sentences=2)
    if random.random() < 0.3:
        text += f" Ref #{fake.random_int(10000, 99999)}."
    text = apply_noise_outside_spans(text, [])
    return PIIExample(text=text, spans=[])


def build_hard_negative_example(fake: Faker) -> PIIExample:
    """
    Contains hashes, GUIDs, invalid cards etc. but should all be labeled O.
    """
    text = sample_hard_negative(fake)
    text = apply_noise_outside_spans(text, [])
    return PIIExample(text=text, spans=[])


#  TOP-LEVEL JSONL GENERATION


def generate_jsonl(
    out_path: str | Path,
    n_samples: int,
    o_only_ratio: float = O_ONLY_RATIO,
    hardneg_ratio: float = HARDNEG_RATIO,
    seed: int = 42,
) -> None:
    """
    Writes a pii.jsonl file where each line is:
    {"text": "...", "spans": [{"start": int, "end": int, "label": "EMAIL"}, ...]}
    """
    random.seed(seed)
    Faker.seed(seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_o = int(n_samples * o_only_ratio)
    n_h = int(n_samples * hardneg_ratio)
    n_pos = n_samples - n_o - n_h

    with out_path.open("w", encoding="utf-8") as f:
        # positive PII examples
        for _ in range(n_pos):
            ex = build_positive_example(FAKE)
            f.write(
                json.dumps({"text": ex.text, "spans": ex.spans}, ensure_ascii=False)
                + "\n"
            )

        # all-O examples
        for _ in range(n_o):
            ex = build_o_only_example(FAKE)
            f.write(
                json.dumps({"text": ex.text, "spans": []}, ensure_ascii=False) + "\n"
            )

        # hard negatives
        for _ in range(n_h):
            ex = build_hard_negative_example(FAKE)
            f.write(
                json.dumps({"text": ex.text, "spans": []}, ensure_ascii=False) + "\n"
            )
