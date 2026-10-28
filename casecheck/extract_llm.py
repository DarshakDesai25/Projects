"""Extractor that asks a language model running on your own computer, through Ollama.

Nothing leaves the machine. The model is asked for facts as JSON, each with a quote copied
from the document, so every fact can be checked against the text.
"""
import json
import re
import urllib.request

from dates import parse_date
from score import FIELDS

PROMPT = """You read legal and medical documents for a law firm. Extract only facts about the patient or witness themselves.

Return JSON in exactly this form and nothing else: {{"facts": [ ... ]}}

Each fact has a "type", the fields listed for that type, and a "quote" that is copied exactly, word for word, from the document.
Types for this document.
{types}

Dates must be written YYYY-MM-DD. For DIAGNOSIS and MEDICATION the date is the date of that visit.
Rules. Do not extract anything that was ruled out, excluded, denied or negative. Do not extract conditions of relatives. Do not guess. If unsure, leave it out.

Document:
{chunk}
"""

TYPES = {
    "record": """- ACCIDENT_DATE: date
- VISIT: date, provider (as written, for example "Dr. Patel")
- DIAGNOSIS: date, name (lowercase)
- MEDICATION: date, drug (lowercase)
- PRIOR_CONDITION: body_part (one of lower back, neck, left shoulder, right knee, right wrist), only when the patient had an earlier problem with it""",
    "deposition": """- TESTIFIED_ACCIDENT_DATE: date (the date the witness gives for the accident)
- TESTIFIED_NO_PRIOR: body_part (one of lower back, neck, left shoulder, right knee, right wrist), only when the witness says they never had problems with it before""",
}
DATE_FIELDS = {"date"}


def chunk_text(text, doc_type, limit=3000):
    """Split at visit or line boundaries so each piece fits in the model's attention."""
    sep = "\n\n" if doc_type == "record" else "\n"
    pieces, cur = [], ""
    for part in text.split(sep):
        if cur and len(cur) + len(part) > limit:
            pieces.append(cur)
            cur = ""
        cur += part + sep
    if cur.strip():
        pieces.append(cur)
    return pieces


def parse_llm_output(raw):
    """Turn the model's reply into fact dictionaries. Anything unreadable is dropped."""
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
    except ValueError:
        i, j = raw.find("{"), raw.rfind("}")
        try:
            data = json.loads(raw[i:j + 1])
        except ValueError:
            return []
    items = data.get("facts", []) if isinstance(data, dict) else data
    facts = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", "")).upper().strip()
        if t not in FIELDS:
            continue
        fact = {"type": t, "quote": str(it.get("quote", ""))}
        for f in FIELDS[t]:
            v = it.get(f)
            if v is None:
                continue
            fact[f] = parse_date(str(v)) if f in DATE_FIELDS else str(v).lower().strip()
        facts.append(fact)
    return facts


def ollama_generate(model, host="http://localhost:11434", timeout=600):
    def generate(prompt):
        body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(host + "/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"]
    return generate


def make_extractor(generate, limit=3000):
    def extract(text, doc_type):
        facts = []
        for chunk in chunk_text(text, doc_type, limit):
            facts += parse_llm_output(generate(PROMPT.format(types=TYPES[doc_type], chunk=chunk)))
        return facts
    return extract
