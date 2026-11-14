"""Check every citation in code before an answer is shown.

  UNGROUNDED         the quote is not on the cited page
  NO_SUCH_PAGE       the file or page does not exist
  WRONG_PERSON_FILE  the page belongs to a different patient, by date of birth and record number
  UNSUPPORTED        the quote does not state the fact, or states it with a negation or about a relative

A claim survives if at least one citation survives. A contradiction needs a surviving citation on both
sides. The verifier never sees the answer key.

Honest caveat. The negation and relative words below were written knowing how the generator phrases
things, so this is close to the best a support check could do on this data. A real support check
would be a separate model that sees only the claim and the quote.
"""
import re
from collections import Counter

from dates import find_dates

NEG = ("ruled out", "no evidence of", "negative for", "was excluded", "is not seen", "not seen", "no acute",
       "denies", "no history", "no prior", "no previous", "states no")
RELATIVE = ("family history", "mother", "father", "sister", "brother", "son", "daughter", "aunt", "uncle")
NEG_ANSWER = re.compile(r"\b(no|never|not that i recall|nothing like that)\b")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def segment(quote, value):
    """The clause of the quote that holds the value."""
    for part in re.split(r"[;.]", quote):
        if value in part:
            return part
    return quote


def supports(fact, quote):
    t, q = fact[0], norm(quote)
    if t in ("ACCIDENT_DATE", "TESTIFIED_ACCIDENT_DATE"):
        return any(d[2] == fact[1] for d in find_dates(quote))
    if t == "TESTIFIED_NO_PRIOR":
        parts = q.split(" a. ", 1)
        return fact[1] in parts[0] and len(parts) == 2 and bool(NEG_ANSWER.search(parts[1]))
    value = fact[-1]
    if value not in q:
        return False
    seg = segment(q, value)
    return not any(n in seg for n in NEG) and not any(r in seg for r in RELATIVE)


def check_citation(env, fact, cit):
    text = env.page_text(cit.get("file"), cit.get("page"))
    if text is None:
        return "NO_SUCH_PAGE"
    if not cit.get("quote") or norm(cit["quote"]) not in norm(text):
        return "UNGROUNDED"
    doc_type, patient = env.file_patient(cit["file"])
    me = env.case["plaintiff"]
    same = patient["name"] == me["name"] if doc_type == "deposition" else (patient["dob"] == me["dob"] and patient["mrn"] == me["mrn"])
    if not same:
        return "WRONG_PERSON_FILE"
    if not supports(cit.get("fact", fact), cit["quote"]):
        return "UNSUPPORTED"
    return None


def verify(env, claims):
    """Returns (kept claims, Counter of removed citations by reason, number of claims dropped)."""
    kept, removed, dropped = [], Counter(), 0
    for c in claims:
        good = []
        for cit in c.get("citations", []):
            why = check_citation(env, c["fact"], cit)
            if why:
                removed[why] += 1
            else:
                good.append(cit)
        if c["fact"][0] in ("PRIOR_DENIED", "DATE_MISMATCH"):
            sides = {x["fact"][0] for x in good}
            ok = len(sides) == 2
        else:
            ok = bool(good)
        if ok:
            kept.append({**c, "citations": good})
        else:
            dropped += 1
    return kept, removed, dropped
