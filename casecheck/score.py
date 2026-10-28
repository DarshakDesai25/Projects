"""Score extracted facts against the answer key and say what kind of mistakes were made.

Mistake types
  MISSED           a true fact was not found
  WRONG_VALUE      the right kind of fact was found but with a wrong value, such as a wrong date
  NEGATION_ERROR   something denied or ruled out was extracted as if it were true
  WRONG_PERSON     something about a relative was extracted as the patient's own
  UNGROUNDED       the supporting quote does not appear in the document
  EXTRA            a real quote, but the fact is not in the answer key
"""
from collections import Counter

from textutil import collapse

ERRORS = ["MISSED", "WRONG_VALUE", "NEGATION_ERROR", "WRONG_PERSON", "UNGROUNDED", "EXTRA"]
FIELDS = {"ACCIDENT_DATE": ("date",), "VISIT": ("date", "provider"), "DIAGNOSIS": ("date", "name"),
          "MEDICATION": ("date", "drug"), "PRIOR_CONDITION": ("body_part",),
          "TESTIFIED_ACCIDENT_DATE": ("date",), "TESTIFIED_NO_PRIOR": ("body_part",)}


def key(fact):
    f = FIELDS.get(fact.get("type"))
    if not f or any(not fact.get(x) for x in f):
        return None
    return (fact["type"],) + tuple(str(fact[x]).lower().strip() for x in f)


SINGLE = {"ACCIDENT_DATE", "TESTIFIED_ACCIDENT_DATE"}   # exactly one per document


def related(a, b):
    """Could these two facts be the same fact with one value wrong?"""
    return a[0] == b[0] and (a[0] in SINGLE or any(x == y for x, y in zip(a[1:], b[1:])))


def score_doc(facts, truth, traps, doc_text):
    """Returns (counts by type, mistake counter, grounded count, extracted count, details).

    details is a list of (mistake type, fact key) so a single document can be inspected.
    """
    doc = collapse(doc_text)
    extracted, grounded = {}, 0
    for f in facts:
        k = key(f)
        if k is None:
            continue
        ok = bool(f.get("quote")) and collapse(f["quote"]) in doc
        if k not in extracted:
            extracted[k] = ok
            grounded += ok
    tp = {k for k in extracted if k in truth}
    fp = [k for k in extracted if k not in truth]
    fn = [k for k in truth if k not in extracted]
    mistakes, unpaired, details = Counter(), list(fn), []
    for k in fp:
        label = None
        if not extracted[k]:
            label = "UNGROUNDED"
        elif k in traps:
            label = traps[k]
        elif k[0] == "DIAGNOSIS" and any(k[1] == t[1] and t[2] in k[2] for t, lab in traps.items() if t[0] == "DIAGNOSIS"):
            label = "NEGATION_ERROR"
        else:
            match = next((u for u in unpaired if related(u, k)), None)
            if match is not None:
                unpaired.remove(match)
                label = "WRONG_VALUE"
            else:
                label = "EXTRA"
        mistakes[label] += 1
        details.append((label, k))
    if unpaired:
        mistakes["MISSED"] += len(unpaired)
    details += [("MISSED", u) for u in unpaired]
    by_type = {t: Counter() for t in FIELDS}
    for k in tp:
        by_type[k[0]]["tp"] += 1
    for k in fp:
        by_type[k[0]]["fp"] += 1
    for k in fn:
        by_type[k[0]]["fn"] += 1
    return by_type, mistakes, grounded, len(extracted), details


def find_inconsistencies(record_facts, depo_facts):
    """Flag places where testimony contradicts the record, using only extracted facts."""
    found = set()
    prior = {f["body_part"] for f in record_facts if f.get("type") == "PRIOR_CONDITION" and f.get("body_part")}
    for f in depo_facts:
        if f.get("type") == "TESTIFIED_NO_PRIOR" and f.get("body_part") in prior:
            found.add(("PRIOR_DENIED", f["body_part"]))
    acc = {f["date"] for f in record_facts if f.get("type") == "ACCIDENT_DATE" and f.get("date")}
    said = {f["date"] for f in depo_facts if f.get("type") == "TESTIFIED_ACCIDENT_DATE" and f.get("date")}
    if acc and said and not (acc & said):
        found.add(("DATE_MISMATCH",))
    return found
