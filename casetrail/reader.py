"""A careful rule based page reader, the kind you would write before reaching for a language model.

It reads one page at a time and returns claims, each with the exact line it came from. It also
notices an addendum saying a diagnosis was entered in error, as a RETRACTION claim, so an agent
that reads that page can drop the diagnosis. Like the
CaseCheck reader, it knows the common phrasings, not every phrasing. It does not know "was excluded",
"is not seen", a sister as a relative, "Hx of" or "Nothing like that". It also does not check whose
record a page belongs to. Those gaps are what the verifier and the scorer are there to catch.
"""
import re

from dates import find_dates

DRUGS = ["ibuprofen", "naproxen", "cyclobenzaprine", "tramadol", "gabapentin", "meloxicam"]
BODY = ["lower back", "neck", "left shoulder", "right knee", "right wrist"]
NEG_DX = ("ruled out", "no evidence of", "negative for")
NEG_HISTORY = ("denies", "no history", "no prior", "no previous", "states no")
RELATIVES = ("family history", "mother", "father")
NEG_MRI = ("no acute", "negative for")


def claim(fact, file, page, quote):
    return {"fact": list(fact), "citations": [{"file": file, "page": page, "quote": quote}]}


def read_page(file, page, text):
    lines = text.split("\n")[1:]                  # skip the page header
    out, visit = [], None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if ln.startswith("Date of service:"):
            ds = find_dates(ln)
            visit = ds[0][2] if ds else None
        m = re.search(r"after a motor vehicle accident on (.+)$", ln)
        if m:
            ds = find_dates(m.group(1))
            if ds:
                out.append(claim(("ACCIDENT_DATE", ds[0][2]), file, page, ln))
        if not any(r in low for r in RELATIVES) and not any(n in low for n in NEG_HISTORY):
            pm = re.search(r"(?:history of|prior|previous)\s+(" + "|".join(BODY) + r")\s+(?:pain|injury|problems|complaints)", low)
            if pm:
                out.append(claim(("PRIOR_CONDITION", pm.group(1)), file, page, ln))
        if ln.startswith("Assessment:") and visit:
            for item in ln[len("Assessment:"):].split(";"):
                item = item.strip().rstrip(".").strip()
                if item and not any(n in item.lower() for n in NEG_DX):
                    out.append(claim(("DIAGNOSIS", visit, item.lower()), file, page, item))
        if ln.startswith("Plan:") and visit:
            for mm in re.finditer(r"(?:Start|Prescribed|Rx:)\s+(" + "|".join(DRUGS) + r")\b", ln, re.I):
                out.append(claim(("MEDICATION", visit, mm.group(1).lower()), file, page, ln))
        pm = re.match(r"(\d{2}/\d{2}/\d{4})\s+Dispensed:\s+(" + "|".join(DRUGS) + r")\b", ln)
        if pm:
            ds = find_dates(pm.group(1))
            if ds:
                out.append(claim(("MEDICATION", ds[0][2], pm.group(2)), file, page, ln))
        am_ = re.match(r"Addendum: (.+?) recorded on (.+?) was entered in error\.", ln)
        if am_:
            ds = find_dates(am_.group(2))
            if ds:
                out.append(claim(("RETRACTION", ds[0][2], am_.group(1).lower()), file, page, ln))
        if ln.startswith(("FINDINGS:", "IMPRESSION:")):
            for item in ln.split(":", 1)[1].split(";"):
                item = item.strip().rstrip(".").strip()
                if item and not any(n in item.lower() for n in NEG_MRI):
                    out.append(claim(("MRI_FINDING", item.lower()), file, page, item))
        qa = re.match(r"\d+:\d+ Q\. (.*)$", ln)
        if qa and i + 1 < len(lines):
            am = re.match(r"\d+:\d+ A\. (.*)$", lines[i + 1])
            if am:
                q, a, quote = qa.group(1).lower(), am.group(1), ln + "\n" + lines[i + 1]
                if "date of the accident" in q:
                    ds = find_dates(a)
                    if ds:
                        out.append(claim(("TESTIFIED_ACCIDENT_DATE", ds[0][2]), file, page, quote))
                bm = re.search(r"problems with your (.+?)\?", q)
                if bm and bm.group(1) in BODY and re.search(r"^(no\b|not that i recall)|\bnever\b", a.lower()):
                    out.append(claim(("TESTIFIED_NO_PRIOR", bm.group(1)), file, page, quote))
    return out
