"""A careful rule based extractor, the kind you would write before reaching for a language model.

It knows the section labels, the date formats, a list of drugs and body parts, and a handful
of phrases that mean a condition is denied, ruled out or belongs to a relative. It is not a
strawman. It is meant to show what the test bench reveals about a reasonable hand built reader.
"""
import re

from dates import find_dates
from textutil import collapse

DRUGS = ["ibuprofen", "naproxen", "cyclobenzaprine", "tramadol", "gabapentin", "meloxicam"]
BODY = ["lower back", "neck", "left shoulder", "right knee", "right wrist"]
NEG_DX = ("ruled out", "no evidence of", "negative for")
NEG_HISTORY = ("denies", "no history", "no prior", "no previous", "states no", "without")
RELATIVES = ("family history", "mother", "father")


def extract(text, doc_type):
    return _record(text) if doc_type == "record" else _deposition(text)


def _record(text):
    t = collapse(text)
    facts = []
    for block in re.split(r"(?=(?:Date of service|DOS):)", t)[1:]:
        m = re.match(r"((?:Date of service|DOS):\s*(.+?)\s+Provider:\s*(Dr\.\s*[A-Za-z]+))", block)
        if not m:
            continue
        ds = find_dates(m.group(2))
        if not ds:
            continue
        vd = ds[0][2]
        facts.append(dict(type="VISIT", date=vd, provider=m.group(3).lower(), quote=m.group(1)))

        for am in re.finditer(r"(?:accident|MVA) on ", block):
            rest = find_dates(block[am.end():am.end() + 30])
            if rest and rest[0][0] == 0:
                facts.append(dict(type="ACCIDENT_DATE", date=rest[0][2], quote=block[am.start():am.end() + rest[0][1]]))

        a = re.search(r"(?:Assessment|Dx):\s*(.*?)\s*(?=Plan:|Rx:|$)", block)
        if a:
            for item in a.group(1).split(";"):
                item = item.strip().rstrip(".").strip()
                if item and not any(n in item.lower() for n in NEG_DX):
                    facts.append(dict(type="DIAGNOSIS", date=vd, name=item.lower(), quote=item))

        for mm in re.finditer(r"(?:Start|Prescribed|Rx:)\s+(" + "|".join(DRUGS) + r")\b", block, re.I):
            facts.append(dict(type="MEDICATION", date=vd, drug=mm.group(1).lower(), quote=mm.group(0)))

        for s in re.split(r"(?<=\.)\s+", block):
            sl = s.lower()
            if any(r in sl for r in RELATIVES) or any(n in sl for n in NEG_HISTORY):
                continue
            pm = re.search(r"(?:history of|prior|previous)\s+(" + "|".join(BODY) + r")\s+(?:pain|injury|problems|complaints)", sl)
            if pm:
                facts.append(dict(type="PRIOR_CONDITION", body_part=pm.group(1), quote=s))
    return facts


def _deposition(text):
    t = collapse(text)
    facts = []
    pairs = re.findall(r"\d+:\d+\s+Q\.\s+(.*?)\s+\d+:\d+\s+A\.\s+(.*?)(?=\s+\d+:\d+\s+Q\.|$)", t)
    for q, a in pairs:
        ql = q.lower()
        if "date of the accident" in ql:
            ds = find_dates(a)
            if ds:
                facts.append(dict(type="TESTIFIED_ACCIDENT_DATE", date=ds[0][2], quote=a))
            continue
        m = re.search(r"problems with your (.+?)\?", ql)
        if m and m.group(1) in BODY and re.search(r"^(no\b|not that i recall)|\bnever\b", a.lower()):
            facts.append(dict(type="TESTIFIED_NO_PRIOR", body_part=m.group(1), quote=a))
    return facts
