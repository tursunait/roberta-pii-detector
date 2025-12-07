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

     # NEW REDDIT-STYLE TEMPLATES:
    "yo anyone know how to contact {person}? their email is {email}",
    "just got scammed by {org}, card ending in {credit_card}",
    "DM me at {email} if interested",
    "{person} is legit, bought from them yesterday",
    "don't share ur ssn like {ssn} online smh",
    "hmu at {email} or call {phone}",
    "does anyone have {person}'s contact info? maybe {email}?",
    "shipping to {address}, hope it arrives by {date}",
    "{org} charged my card {credit_card} without permission wtf",
    "my phone is {phone} if u need to reach me",
    "contact info: {person}, {email}, {phone}",
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

def generate_variable_length_text(fake: Faker) -> PIIExample:
    """Generate text of varying lengths"""
    length_type = random.choice(['short', 'medium', 'long'])
    
    if length_type == 'short':
        # 20-60 tokens: simple sentence
        return build_positive_example(fake)
    elif length_type == 'medium':
        # 60-150 tokens: paragraph (2-3 templates combined)
        num_templates = random.randint(2, 3)
        texts = []
        all_spans = []
        cursor = 0
        
        for _ in range(num_templates):
            ex = build_positive_example(fake)
            texts.append(ex.text)
            # Adjust span positions
            for span in ex.spans:
                span['start'] += cursor
                span['end'] += cursor
            all_spans.extend(ex.spans)
            cursor += len(ex.text) + 1  # +1 for space
        
        return PIIExample(text=' '.join(texts), spans=all_spans)
    else:
        # 150-400 tokens: long post (4-7 templates combined)
        num_templates = random.randint(4, 7)
        texts = []
        all_spans = []
        cursor = 0
        
        for _ in range(num_templates):
            ex = build_positive_example(fake)
            texts.append(ex.text)
            # Adjust span positions
            for span in ex.spans:
                span['start'] += cursor
                span['end'] += cursor
            all_spans.extend(ex.spans)
            cursor += len(ex.text) + 1  # +1 for space
        
        return PIIExample(text=' '.join(texts), spans=all_spans)

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

# Add noise INSIDE PII spans
def apply_noise_to_pii(text: str, spans: List[Tuple[int, int]], noise_prob=0.1) -> str:
    """Add realistic noise/typos to PII itself"""
    chars = list(text)
    
    for span_idx, (s, e) in enumerate(spans):
        if random.random() < noise_prob:
            # Get the PII text
            pii_text = text[s:e]
            
            # Apply different types of realistic noise
            noise_type = random.choice(['typo', 'spacing', 'case'])
            
            if noise_type == 'typo':
                # Common typos: gmail -> gmial, yahoo -> yaho
                pii_text = pii_text.replace('gmail', 'gmial')
                pii_text = pii_text.replace('yahoo', 'yaho')
                pii_text = pii_text.replace('com', 'con')
            elif noise_type == 'spacing':
                # Add extra spaces in phone/SSN: 555-1234 -> 555 - 1234
                pii_text = pii_text.replace('-', ' - ')
                pii_text = pii_text.replace('.', ' . ')
            elif noise_type == 'case':
                # Random case changes
                pii_text = ''.join([c.upper() if random.random() < 0.3 else c.lower() for c in pii_text])
            
            # Replace in the character list
            chars[s:e] = list(pii_text)
    
    return ''.join(chars)

def obfuscate_email(email: str) -> str:
    """Create obfuscated email versions"""
    variants = [
        email.replace('@', ' at ').replace('.', ' dot '),
        email.replace('@', '[at]').replace('.', '[dot]'),
        email.replace('@', ' @ ').replace('.', ' . '),
        email.replace('@', '(at)').replace('.', '(dot)'),
    ]
    return random.choice(variants)

def obfuscate_phone(phone: str) -> str:
    """Create obfuscated phone versions"""
    # Remove formatting and add spaces
    digits = ''.join(c for c in phone if c.isdigit())
    variants = [
        ' '.join(digits),  # "5 5 5 1 2 3 4"
        '-'.join([digits[i:i+3] for i in range(0, len(digits), 3)]),  # "555-123-4"
        digits[:3] + ' ' + digits[3:6] + ' ' + digits[6:],  # "555 123 4567"
    ]
    return random.choice(variants)

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


# def build_positive_example(fake: Faker) -> PIIExample:
#     template = fake.random_element(TEMPLATES)
#     text, field_spans = fill_template(template, fake)

#     field2label = {
#         "person": "PERSON",
#         "org": "ORG",
#         "address": "ADDRESS",
#         "email": "EMAIL",
#         "phone": "PHONE",
#         "ssn": "SSN",
#         "credit_card": "CREDIT_CARD",
#         "date": "DATE",
#     }

#     spans: List[Dict[str, Any]] = []
#     char_spans: List[Tuple[int, int]] = []

#     for field, (s, e) in field_spans.items():
#         label = field2label[field]
#         spans.append({"start": s, "end": e, "label": label})
#         char_spans.append((s, e))

#     noisy_text = apply_noise_outside_spans(text, char_spans)
#     return PIIExample(text=noisy_text, spans=spans)

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
        
        # Randomly obfuscate emails and phones (20% of the time)
        if label == "EMAIL" and random.random() < 0.2:
            original = text[s:e]
            obfuscated = obfuscate_email(original)
            text = text[:s] + obfuscated + text[e:]
            e = s + len(obfuscated)
        elif label == "PHONE" and random.random() < 0.2:
            original = text[s:e]
            obfuscated = obfuscate_phone(original)
            text = text[:s] + obfuscated + text[e:]
            e = s + len(obfuscated)
        
        spans.append({"start": s, "end": e, "label": label})
        char_spans.append((s, e))
    
    # Apply noise outside spans
    noisy_text = apply_noise_outside_spans(text, char_spans)
    
    # Apply noise inside PII (10% probability)
    if random.random() < 0.1:
        noisy_text = apply_noise_to_pii(noisy_text, char_spans, noise_prob=0.15)
    
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
            ex = generate_variable_length_text(FAKE)  # CHANGED: Use variable length
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

