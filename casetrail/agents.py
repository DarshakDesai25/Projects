"""Three ways to answer the questions, all returning claims with citations.

  FixedWorkflow  for each question, run set searches on the navigation map, read the top pages up
                 to a page budget, and pull out facts with the rule based reader. Predictable and cheap.
  FactsTable     read every page once, keep a table of facts with their pages, and answer every
                 question from the table. Expensive up front, nothing missed for lack of reading.
  LLMAgent       a language model decides which tool to call next, one step at a time, with a step
                 limit, and finishes with claims that quote their sources. See llm_agent.py.

A claim is {"question", "fact", "citations": [{"file", "page", "quote", "fact"?}]}. A contradiction
claim cites both sides, and each of its citations names the fact it supports.
"""
import math

from generate import key
from reader import read_page

PLANS = {
    "accident": {"queries": ["motor vehicle accident"], "doc_types": ["medical_record"], "types": ["ACCIDENT_DATE"]},
    "diagnoses": {"queries": ["assessment"], "doc_types": ["medical_record"], "types": ["DIAGNOSIS"]},
    "medications": {"queries": ["dispensed", "plan start prescribed rx"], "doc_types": ["pharmacy", "medical_record"], "types": ["MEDICATION"]},
    "prior": {"queries": ["history prior previous"], "doc_types": ["medical_record"], "types": ["PRIOR_CONDITION"]},
    "mri": {"queries": ["mri findings impression"], "doc_types": ["radiology"], "types": ["MRI_FINDING"]},
}


def merge(claims):
    """One claim per fact, with every citation that supports it."""
    out = {}
    for c in claims:
        k = key(c["fact"])
        if k in out:
            seen = {(x["file"], x["page"], x["quote"]) for x in out[k]["citations"]}
            out[k]["citations"] += [x for x in c["citations"] if (x["file"], x["page"], x["quote"]) not in seen]
        else:
            out[k] = {"fact": list(c["fact"]), "citations": list(c["citations"])}
    return list(out.values())


def apply_retractions(claims):
    """Drop diagnoses that a page the agent read says were entered in error, and the retraction notes."""
    gone = {("DIAGNOSIS", c["fact"][1], c["fact"][2]) for c in claims if c["fact"][0] == "RETRACTION"}
    return [c for c in claims if c["fact"][0] != "RETRACTION" and tuple(c["fact"]) not in gone]


def contradictions(claims):
    """Compare testimony with the records, using only claims that were extracted."""
    by_type = {}
    for c in claims:
        by_type.setdefault(c["fact"][0], []).append(c)
    out = []
    prior = {c["fact"][1]: c for c in by_type.get("PRIOR_CONDITION", [])}
    for t in by_type.get("TESTIFIED_NO_PRIOR", []):
        bp = t["fact"][1]
        if bp in prior:
            out.append({"fact": ["PRIOR_DENIED", bp],
                        "citations": [{**x, "fact": t["fact"]} for x in t["citations"][:1]] +
                                     [{**x, "fact": prior[bp]["fact"]} for x in prior[bp]["citations"][:1]]})
    acc, said = by_type.get("ACCIDENT_DATE", []), by_type.get("TESTIFIED_ACCIDENT_DATE", [])
    if acc and said and acc[0]["fact"][1] != said[0]["fact"][1]:
        out.append({"fact": ["DATE_MISMATCH"], "citations": [{**said[0]["citations"][0], "fact": said[0]["fact"]},
                                                             {**acc[0]["citations"][0], "fact": acc[0]["fact"]}]})
    return out


def ranked_pages(env, queries, doc_types):
    """Pages from the map searches, best first, without repeats."""
    pages = []
    for q in queries:
        for h in env.search_map(q, doc_types, top=1000):
            if (h["file"], h["page"]) not in pages:
                pages.append((h["file"], h["page"]))
    return pages


def question_pages(env, qid, budget):
    """The pages the fixed workflow reads for a question. A contradiction splits the budget between
    the medical records and the deposition."""
    if qid == "contradictions":
        half = None if budget is None else max(1, math.ceil(budget / 2))
        record = ranked_pages(env, ["motor vehicle accident history prior previous"], ["medical_record"])
        testimony = ranked_pages(env, ["accident problems"], ["deposition"])
        return (record if half is None else record[:half]) + (testimony if half is None else testimony[:half])
    plan = PLANS[qid]
    pages = ranked_pages(env, plan["queries"], plan["doc_types"])
    return pages if budget is None else pages[:budget]


class FixedWorkflow:
    """Search, read the best pages up to a budget per question, extract with rules."""

    name = "fixed_workflow"

    def __init__(self, budget=None):
        self.budget = budget                      # pages per question, None means every matching page

    def answer(self, env, question):
        qid = question["id"]
        claims = []
        for fid, p in question_pages(env, qid, self.budget):
            r = env.open_page(fid, p)
            if "text" in r:
                claims += read_page(fid, p, r["text"])
        if qid == "contradictions":
            return contradictions(merge(claims))
        claims = apply_retractions(claims)
        return merge([c for c in claims if c["fact"][0] in PLANS[qid]["types"]])


class FactsTable:
    """Read every page once, then answer every question from the table."""

    name = "facts_table"

    def __init__(self):
        self.table = None

    def build(self, env):
        claims = []
        for f in env.case["files"]:
            for p in range(1, len(f["pages"]) + 1):
                r = env.open_page(f["file_id"], p)
                claims += read_page(f["file_id"], p, r["text"])
        self.table = merge(apply_retractions(claims))

    def answer(self, env, question):
        if self.table is None:
            self.build(env)
        qid = question["id"]
        if qid == "contradictions":
            return contradictions(self.table)
        types = PLANS[qid]["types"]
        return [c for c in self.table if c["fact"][0] in types]
