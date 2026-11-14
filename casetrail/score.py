"""Score answers against the answer key, citation by citation, and name every mistake.

  found           a true fact, with at least one citation on a page where that fact really appears
  WRONG_CITATION  a true fact, but none of its citations point to a page where it appears
  MISSED          a true fact that no claim gave
  NEGATION_ERROR  something denied or ruled out, given as true
  WRONG_PERSON    something about a relative, given as the plaintiff's
  WRONG_PERSON_FILE  something from a different patient's record with the same name
  RETRACTED       a diagnosis that a later page says was entered in error
  WRONG_VALUE     the right kind of fact with a wrong value, such as a wrong accident date
  EXTRA           anything else that is not in the answer key
"""
from collections import Counter

from generate import key

KINDS = ["NEGATION_ERROR", "WRONG_PERSON", "WRONG_PERSON_FILE", "RETRACTED", "WRONG_VALUE", "EXTRA"]


def classify(fact, case):
    k = key(fact)
    if k in case["traps"]:
        return case["traps"][k]
    for tk, kind in case["traps"].items():     # "fracture was excluded" is the ruled out fracture, said differently
        tf = tk.split("|")
        if tf[0] == fact[0] and tf[:-1] == list(fact[:-1]) and tf[-1] in str(fact[-1]):
            return kind
    if fact[0] in ("ACCIDENT_DATE",) or (fact[0] == "DIAGNOSIS" and any(g[1] == fact[1] for g in case["gold"]["diagnoses"])):
        return "WRONG_VALUE"
    return "EXTRA"


def valid_citation(fact, citations, case):
    if fact[0] == "PRIOR_DENIED":
        sides = [key(("TESTIFIED_NO_PRIOR", fact[1])), key(("PRIOR_CONDITION", fact[1]))]
    elif fact[0] == "DATE_MISMATCH":
        said = [k for k in case["locations"] if k.startswith("TESTIFIED_ACCIDENT_DATE|")]
        sides = said[:1] + [key(("ACCIDENT_DATE", case["gold"]["accident"][0][1]))]
    else:
        locs = {tuple(x) for x in case["locations"].get(key(fact), [])}
        return any((c["file"], c["page"]) in locs for c in citations)
    return all(any((c["file"], c["page"]) in {tuple(x) for x in case["locations"].get(s, [])} for c in citations) for s in sides)


def score_question(qid, claims, case):
    gold = {key(g) for g in case["gold"][qid]}
    found, mistakes = set(), Counter()
    for c in claims:
        k = key(c["fact"])
        if k in gold:
            if valid_citation(c["fact"], c["citations"], case):
                found.add(k)
            else:
                mistakes["WRONG_CITATION"] += 1
        else:
            mistakes[classify(c["fact"], case)] += 1
    mistakes["MISSED"] = len(gold - found)
    return {"gold": len(gold), "claims": len(claims), "found": len(found), "mistakes": mistakes,
            "empty_and_right": not gold and not claims}
