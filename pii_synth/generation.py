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
        "person": generate_person(),
        "org": fake.company(),
        "address": generate_address(fake),
        "email": generate_email(),
        "phone": generate_phone(fake),
        "ssn": generate_ssn(),
        "credit_card": fake.credit_card_number(card_type=None),
        "date": generate_date(fake), 
        'age': generate_age(),
    }

def generate_person():
    """Generate both real names AND usernames"""
    
    # 50% real names, 50% usernames
    if random.random() < 0.5:
        return FAKE.name()  # "John Smith"
    else:
        # Generate username-style
        patterns = [
            # KEEP YOUR EXISTING ONES:
            f"{FAKE.user_name()}",
            f"{FAKE.first_name().lower()}{random.randint(100,999)}",
            f"{FAKE.word()}{random.randint(10,99)}",
            ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(8,15))) + str(random.randint(1,999)),
            
            # ===== NEW PATTERNS BELOW (ADD THESE) =====
            
            # Numbers at START (29summikota)
            f"{random.randint(10,99)}{FAKE.user_name()}",
            
            # Single letter + numbers (A141981, N23)
            f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000,999999)}",
            f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10,99)}",
            
            # Just 2 letters - initials (GR)
            ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2)),

            # Just 3 letters - initials (GRL)
            ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3)),

            # Just 4 letters - initials (GRLO)
            ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4)),
            
            # firstname.lastname format (gholamhossein.ruschke)
            f"{FAKE.first_name().lower()}.{FAKE.last_name().lower()}",
            
            # firstname.lastname + numbers (kees.gyorgy02, nilou.ucci12)
            f"{FAKE.first_name().lower()}.{FAKE.last_name().lower()}{random.randint(10,99)}",
            
            # Very long random string + numbers (lqsdrojhmrlcw54)
            ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(12,18))) + str(random.randint(10,999)),
            
            # VERY long random + long numbers (npvhxlrgvjdhzjaf439498)
            ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(15,20))) + str(random.randint(100000,999999)),
            
            # Year at START + username (1980refad.chaïb)
            f"{random.randint(1950,2010)}{FAKE.user_name()}",
            f"{random.randint(1950,2010)}{FAKE.first_name().lower()}.{FAKE.last_name().lower()}",
            
            # Random letters + 2-digit number (japeschk92)
            ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(6,10))) + str(random.randint(10,99)),
            
            # Long username patterns from CSV (pdmjrsyoz1460)
            ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(8,12))) + str(random.randint(1000,9999)),
            
            # firstname_lastname (underscore separator)
            f"{FAKE.first_name().lower()}_{FAKE.last_name().lower()}",
            
            # firstname_lastname + numbers
            f"{FAKE.first_name().lower()}_{FAKE.last_name().lower()}{random.randint(1,99)}",
            
            # Just firstname lowercase (helbert, abdi, iloweintögl)
            FAKE.first_name().lower(),
            
            # Mixed case single word usernames
            FAKE.user_name().lower(),
        ]
        return random.choice(patterns)
    
def generate_ssn():
    """Generate various ID number formats"""
    
    formats = [
        # US SSN (KEEP THIS)
        f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
        
        # European formats (KEEP THESE)
        f"{random.randint(100000000,999999999)}",
        f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10000000,99999999)}",
        f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
        
        # Driver license / ID card style (KEEP THIS)
        f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000000,9999999)}",
        
        # ===== NEW PATTERNS BELOW (ADD THESE) =====
        
        # Spaces as separators (996 076 6460)
        f"{random.randint(100,999)} {random.randint(100,999)} {random.randint(1000,9999)}",
        
        # Dots as separators (554.575.9355)
        f"{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(1000,9999)}",
        
        # 10 digits starting with 0 (0610780437)
        f"0{random.randint(100000000,999999999)}",
        
        # 9 digits starting with 0 (080065505)
        f"0{random.randint(10000000,99999999)}",
        
        # Complex format with dots and letters (27.01.06.52.N67.7)
        f"{random.randint(10,99)}.{random.randint(10,99)}.{random.randint(10,99)}.{random.randint(10,99)}.{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10,99)}.{random.randint(1,9)}",
    ]
    
    return random.choice(formats)
    
def generate_age():
    age_num = random.randint(1, 99)
    
    formats = [
        # EXISTING:
        f"{age_num}M", f"{age_num}F", f"M{age_num}", f"F{age_num}",
        f"{age_num}m", f"{age_num}f",
        f"({age_num}M)", f"({age_num}F)",
        f"{age_num} years old", f"{age_num}yo",
        str(age_num),
        
        # NEW ONES:
        f"[{age_num}M]", f"[{age_num}F]",
        f"{age_num} year old",
        f"{age_num}-year-old",
        f"age {age_num}",
        f"I'm {age_num}",
        f"i'm {age_num}",
        f"{age_num}M/{random.randint(18,65)}F",
        f"({age_num})",
        f"{age_num} M", f"{age_num} F",
    ]
    
    return random.choice(formats)

def generate_phone(fake):
    phone = fake.phone_number()
    
    variations = [
        # KEEP ALL YOUR EXISTING ONES:
        phone,
        phone.replace('-', ' '),
        phone.replace('-', '.'),
        phone.replace('-', ''),
        phone[-8:],
        phone[-4:],
        f"xxx-xxx-{phone[-4:]}",
        f"***-***-{phone[-4:]}",
        f"ends in {phone[-4:]}",
        f"call me at {phone}",
        phone.replace('(', '').replace(')', ''),
        phone.replace('-', '/'),
        f"+1 {phone}",
        f"tel: {phone}",
        f"phone: {phone}",
        
        # ===== NEW PATTERNS BELOW (ADD THESE) =====
        
        # International with country code + mixed separators (+31880-385-2406)
        f"+{random.randint(1,999)}{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
        
        # International with DOTS and DASHES mixed (+20-914.834.1296)
        f"+{random.randint(1,99)}-{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(1000,9999)}",
        
        # International with dots at end (+51-063-367.7939)
        f"+{random.randint(1,99)}-{random.randint(100,999)}-{random.randint(100,999)}.{random.randint(1000,9999)}",
        
        # DOTS and DASHES together (01881.881.151-3030)
        f"0{random.randint(1000,9999)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(1000,9999)}",
        
        # Country code + space in middle (+3380820 0420)
        f"+{random.randint(10,999)}{random.randint(1000,9999)} {random.randint(1000,9999)}",
        
        # Space then DOT (076 1352.8018)
        f"0{random.randint(10,99)} {random.randint(1000,9999)}.{random.randint(1000,9999)}",
        
        # 4-digit groups (4929-667-4889)
        f"{random.randint(1000,9999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
        
        # Country code + space + mixed (+07 69-909 8310)
        f"+{random.randint(1,99)} {random.randint(10,99)}-{random.randint(100,999)} {random.randint(1000,9999)}",
        
        # Leading zero with dashes (0070-0###)
        f"00{random.randint(10,99)}-{random.randint(1000,9999)}",
        
        # International space-separated (+31 880 385 2406)
        f"+{random.randint(1,99)} {random.randint(100,999)} {random.randint(100,999)} {random.randint(1000,9999)}",
    ]
    
    return random.choice(variations)

def generate_date(fake):
    """Generate date in various formats"""
    date_obj = fake.date_object()
    
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    month_name = date_obj.strftime('%B')
    month_short = date_obj.strftime('%b')
    
    # Helper function for ordinal suffixes
    def ordinal_suffix(d):
        if 10 <= d % 100 <= 20:
            return 'th'
        else:
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')
    
    formats = [
        # Standard formats (KEEP THESE):
        f"{year}-{month:02d}-{day:02d}",           # 1990-05-15 (ISO)
        f"{month:02d}/{day:02d}/{year}",           # 05/15/1990 (US)
        f"{day:02d}/{month:02d}/{year}",           # 15/05/1990 (European)
        f"{month_name} {day}, {year}",             # May 15, 1990
        f"{month_short} {day}, {year}",            # May 15, 1990
        
        # More variations (KEEP THESE):
        str(year),                                  # just 1990
        f"born in {year}",                         # born in 1990
        f"{month:02d}/{day:02d}/{year % 100}",     # 05/15/90 (short year)
        f"{month}/{day}/{year}",                   # 5/15/1990 (no leading zeros)
        f"{day}/{month}/{year}",                   # 15/5/1990 (European, no zeros)
        f"{month_name} {year}",                    # May 1990
        f"birthday: {month}/{day}",                # birthday: 5/15 (no year)
        
        # ===== NEW PATTERNS BELOW (ADD THESE) =====
        
        # ISO format with T00:00:00 (2076-12-08T00:00:00)
        f"{year}-{month:02d}-{day:02d}T00:00:00",
        
        # Month/Day format with slash, no year (September/54, April/32)
        f"{month_name}/{day}",
        
        # Ordinal dates (23rd June 1958, 21st December 1989)
        f"{day}{ordinal_suffix(day)} {month_name} {year}",
        
        # Month with ordinal (August 5th, 2057)
        f"{month_name} {day}{ordinal_suffix(day)}, {year}",
        
        # Short month with ordinal (Dec 18, 1969 could be Dec 18th, 1969)
        f"{month_short} {day}{ordinal_suffix(day)}, {year}",
    ]
    
    return random.choice(formats)

def generate_address(fake):
    """Generate individual address components matching real data patterns"""
    
    # 80% individual components, 20% full addresses
    if random.random() < 0.2:
        # Full address (keep some)
        return fake.address().replace("\n", ", ")
    
    # Individual components (like real data!)
    components = [
        # === VERY SHORT COMPONENTS (critical!) ===
        fake.country_code(),  # 'GB', 'US', 'FR' (2 chars)
        fake.country_code(),  # Repeat for higher probability
        
        str(random.randint(1, 9999)),  # Building numbers: '163', '742'
        str(random.randint(1, 999)),   # More building numbers
        
        fake.state_abbr(),  # 'MA', 'ENG', 'CA' (2-3 chars)
        fake.state_abbr(),  # Repeat
        
        # === STREET NAMES (common) ===
        fake.street_name(),  # 'Main Street', 'Oak Avenue'
        fake.street_name(),  # Repeat
        fake.street_name(),  # More variety
        
        # === CITIES (very common) ===
        fake.city(),  # 'Bristol', 'Boston'
        fake.city(),  # Repeat for high probability
        fake.city(),
        fake.city(),
        
        # === POSTCODES (very common) ===
        fake.postcode(),  # 'BS34 7HU'
        fake.postcode(),  # Repeat
        fake.postcode(),
        fake.postcode().split()[0] if ' ' in fake.postcode() else fake.postcode(),  # Just first part: 'CM21'
        
        # Multiple postcodes
        f"{fake.postcode()}, {fake.postcode()}",  # 'BS34 7HU, BS34 7HZ'
        
        # === FULL STREET ADDRESSES ===
        fake.street_address(),  # '123 Main St'
        fake.street_address(),
        
        # === COUNTRIES (full names) ===
        fake.country(),  # Random countries
        fake.country(),  # Repeat
        fake.country(),
        
        # === STATES (full names) ===
        fake.state(),  # 'Massachusetts', 'California'
        
        # === SECONDARY ADDRESSES ===
        f"Apartment {random.randint(1, 999)}",
        f"Suite {random.randint(1, 999)}",
        f"Floor {random.randint(1, 50)}",
    ]
    
    return random.choice(components)

def generate_email():
    """Generate email addresses with various patterns"""
    
    # 70% use Faker's built-in (it's already good)
    if random.random() < 0.7:
        return FAKE.email()
    
    # 30% generate custom patterns to match real data
    else:
        domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com', 'protonmail.com', 'icloud.com']
        
        patterns = [
            # Very short emails (ZB@yahoo.com)
            f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}@{random.choice(domains)}",
            
            # Short lowercase (ab@yahoo.com)
            f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=2))}@{random.choice(domains)}",
            
            # Random chars + numbers (xwjhgbgg009@outlook.com)
            f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5,10)))}{random.randint(1,999):03d}@{random.choice(domains)}",
            
            # Long random string + numbers (vtpkbqcutaxb799@yahoo.com)
            f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(10,15)))}{random.randint(100,999)}@{random.choice(domains)}",
            
            # Longer names (blerenbaasgara@gmail.com)
            f"{FAKE.first_name().lower()}{FAKE.last_name().lower()}@{random.choice(domains)}",
            
            # First letter + last name (bballoi@yahoo.com)
            f"{FAKE.first_name()[0].lower()}{FAKE.last_name().lower()}@{random.choice(domains)}",
            
            # Name with numbers
            f"{FAKE.last_name().lower()}{random.randint(1,99)}@{random.choice(domains)}",
            
            # Underscore separator
            f"{FAKE.first_name().lower()}_{FAKE.last_name().lower()}@{random.choice(domains)}",
            
            # Dot separator (kees.guirard@aol.com)
            f"{FAKE.first_name().lower()}.{FAKE.last_name().lower()}@{random.choice(domains)}",
        ]
        
        return random.choice(patterns)
    
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
    "My brother is {age} years old",
    "Applicant: {person}, DOB: {date}, SSN: {ssn}, Contact: {phone}",
    "Username: {person}, Email: {email}, Age: {age}, Phone: {phone}",
    "Attendees: {person}, {person}, {person} - Call-in: {phone}",
    "Name: {person} | DOB: {date} | Email: {email} | SSN: {ssn}",
    "Full Name: {person}, Address: {address}, Email: {email}, Social Number: {ssn}",
    
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
    # Add these to your templates list:
    "I {age} just broke up with my partner and need advice",
    "My boyfriend {age} won't talk to me about our problems",
    "Me {age} and my girlfriend {age} can't agree on anything",
    "I {age} never kissed anyone about to meet a {age} and terrified",
    "My partner {age} is leaving me {age} for another woman",
    "I {age} hate my wife's {age} best friend",
    "deleted my old account but you can reach me at {email} or {phone}",


    # Add these templates to your TEMPLATES list in generation.py
    # These are more realistic and match real-world text patterns

    # ========== REDDIT/FORUM STYLE (Casual, Messy) ==========
    "throwaway but {person} totally screwed me, email {email} if u want details",
    "PSA don't trust {org}, they have my card {credit_card} and won't refund",
    "anyone else get scammed by {person}? dm me at {email}",
    "TIFU by giving {org} my ssn {ssn} over the phone",
    "deleted my account but hmu at {email} or {phone}",
    "yo {person} hit me up, lost ur number, mine is {phone}",
    "can someone help me contact {person}? maybe try {email}?",
    "bruh i think {org} leaked my info ({email}, {phone}, even address {address})",
    "my ex {person} won't stop calling from {phone} wtf do i do",
    
    # ========== APPLICATION/FORM STYLE (Structured) ==========
    "Applicant: {person}\nDOB: {date}\nSSN: {ssn}\nContact: {phone}\nEmail: {email}",
    "Full Name: {person}\nAddress: {address}\nPhone: {phone}\nSocial: {ssn}",
    "ID: {person}, Born: {date}, Age: {age}, Contact: {email}/{phone}",
    "Name: {person} | Email: {email} | SSN: {ssn} | Card: {credit_card}",
    "Username: {person}\nEmail: {email}\nPhone: {phone}\nRegistered: {date}",
    "Patient: {person}, DOB {date}, Phone {phone}, Address {address}",
    "Employee #{person}, Hired {date}, SSN {ssn}, Dept: {org}",
    "Account holder: {person}, Card ending {credit_card}, Exp {date}",
    
    # ========== MEETING/CONFERENCE STYLE ==========
    "Meeting on {date} - Attendees: {person}, {person}, {person}",
    "Call scheduled for today, dial {phone} for access",
    "Attendees: {person} ({email}), {person} ({phone})",
    "Zoom link sent to {email}, meeting on {date}",
    "Please confirm attendance for {date} - reply to {email}",
    
    # ========== DATA DUMP/CSV STYLE ==========
    "{person},{date},{email},{phone},{ssn}",
    "{person} | {age} | {address} | {phone}",
    "Name: {person}, Email: {email}, Phone: {phone}, Card: {credit_card}",
    "{person};{date};{ssn};{org};{phone}",
    
    # ========== EMAIL/MESSAGE STYLE ==========
    "Hi {person}, Your order will ship to {address} by {date}. Questions? Call {phone}",
    "From: {person}\nTo: {email}\nSubject: Meeting {date}\nCall me: {phone}",
    "Thanks {person}! Your card {credit_card} was charged on {date}",
    
    # ========== SOCIAL MEDIA STYLE ==========
    "happy birthday {person}! call me later {phone}",
    "{person} tagged you in a post from {date}",
    "following {person} now, hmu at {email}",
    
    # ========== CUSTOMER SERVICE/SUPPORT ==========
    "Ticket #{person} - Customer: {person}, Email: {email}, Issue date: {date}",
    "Reference #{person}, Contact {phone}, Card ending {credit_card}",
    "Support request from {person} at {address}, callback: {phone}",
    
    # ========== MEDICAL/HEALTH STYLE ==========
    "Patient {person}, DOB {date}, Age {age}, Contact {phone}",
    "Visit scheduled on {date}, Patient: {person}, SSN: {ssn}",
    "Prescription for {person}, filled {date}, Phone: {phone}",
    
    # ========== LEGAL/CONTRACT STYLE ==========
    "Party A: {person}, SSN {ssn}, Address {address}, Date {date}",
    "Signed by {person} on {date}, Contact: {email}",
    "Witness: {person}, Phone: {phone}, Date: {date}",
    
    # ========== DELIVERY/SHIPPING STYLE ==========
    "Ship to {person} at {address}, Contact {phone}, Expected {date}",
    "Delivery for {person}, Address: {address}, Phone: {phone}",
    "Package arriving {date} at {address}, Call {phone} if issues",
    
    # ========== REGISTRATION/SIGNUP STYLE ==========
    "Welcome {person}! Your account email is {email}, registered {date}",
    "User {person} created {date}, Contact: {email}/{phone}",
    "Profile: {person}, Age {age}, Email {email}, Joined {date}",
    
    # ========== MIXED/CHAOTIC (Most realistic!) ==========
    "ok so {person} said to email {email} or call {phone} but idk if its legit",
    "my info: {person}, born {date}, email me at {email} not {phone}",
    "DON'T SHARE but here's {person}'s contact: {email} / {phone} / ssn {ssn}",
    "invoice to {org} attn {person} card {credit_card} address {address}",
    "{person} ({age}) applied on {date}, contact via {email} or {phone}",
    
    # ========== SHORT/INCOMPLETE (Test edge cases) ==========
    "{person} {email}",
    "{phone} call me",
    "email: {email} phone: {phone}",
    "{person} - {date}",
    "{ssn} {person}",

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
                pii_text = pii_text.replace('gmail', 'gmal')
                pii_text = pii_text.replace('yahoo', 'yaho')
                pii_text = pii_text.replace('yahoo', 'yahooo')
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

def obfuscate_email(email):
    """Generate obfuscated email variations"""
    try:
        local, domain = email.split('@')
    except:
        return email  # Return original if malformed
    
    domain_parts = domain.split('.')
    
    variations = [
        email,
        f"{local} at {domain}",
        f"{local}[at]{domain}",
        f"{local} (at) {domain}",
        f"{local} AT {domain}",
        f"{local} @ {domain}",
        f"{local} [AT] {domain}",
    ]
    
    # Only add variations with domain parts if domain has proper structure
    if len(domain_parts) >= 2:
        variations.extend([
            f"{local} dot {domain_parts[0]} dot {domain_parts[1]}",  # john dot gmail dot com
            f"{local}[at]{domain_parts[0]}[dot]{domain_parts[1]}",   # john[at]gmail[dot]com
            f"{local} @ {domain_parts[0]} . {domain_parts[1]}",      # john @ gmail . com
        ])
    
    # Safe replacements
    variations.append(email.replace('.', ' dot '))
    variations.append(email.replace('@', ' at ').replace('.', ' dot '))
    
    return random.choice(variations)


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
        "age": "AGE", # ADDED THIS
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

def build_realistic_o_only_example(fake: Faker) -> PIIExample:
    """Generate realistic non-PII text that looks like it could have PII but has none"""
    
    patterns = [
        # Casual conversation (no actual PII)
        "hey did you see that movie last night? was amazing",
        "just finished my homework, finally free for the weekend",
        "anyone know a good restaurant around here?",
        "the weather today is terrible, staying inside",
        
        # Forum posts (no PII)
        "UPDATE: figured out the bug, it was a typo lol",
        "PSA: new update is out, anyone tried it yet?",
        "throwaway but need advice on a situation",
        
        # Work-like text (no PII)
        "meeting went well, discussed the new features",
        "project deadline is next week, almost done",
        "presentation was good, got positive feedback",
        
        # Technical text (no PII)
        "error code 404, server not responding",
        "build version 1.2.3 deployed successfully",
        "database query returned empty results",
        
        # Random paragraph
        fake.paragraph(nb_sentences=random.randint(2, 5)),
    ]
    
    text = random.choice(patterns)
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

