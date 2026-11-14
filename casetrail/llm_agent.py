"""A language model running on your own computer through Ollama, in two modes.

  agent    The model decides which tool to call next, one step at a time, with a step limit and a
           page budget, and finishes with claims that cite their sources.
  guided   The fixed workflow chooses the pages, the same pages the rule based agent reads. The model
           only reads them and extracts claims in one call per question. Comparisons, such as
           contradictions between testimony and records, are done by code from the extracted facts.

Comparing the two separates "did it find the right pages" from "did it read them correctly".

Two options for the agent, so each change can be measured on its own.
  citations="lines"   pages are shown with numbered lines and the model cites line numbers. The code
                      fills in the exact text, so a citation can be wrong but never misquoted.
  citations="quotes"  the model copies the words itself, as in the first version.
  nudges=1            if the model finishes while pages from its own searches are still unopened and
                      budget remains, the code says so once and asks it to continue. 0 turns it off.

  ollama pull qwen2.5:7b
  python run.py --agent ollama --mode agent  --model qwen2.5:7b --size large --cases 5 --budget 8
  python run.py --agent ollama --mode guided --model qwen2.5:7b --size large --cases 5 --budget 8

The claims are checked by the same verifier and scored by the same scorer as the rule based agents.
"""
import json
import re
import urllib.request

from agents import contradictions, merge, question_pages
from dates import parse_date
from reader import BODY, DRUGS

TYPES = {"accident": ["ACCIDENT_DATE"], "diagnoses": ["DIAGNOSIS"], "medications": ["MEDICATION"],
         "prior": ["PRIOR_CONDITION"], "mri": ["MRI_FINDING"], "contradictions": ["PRIOR_DENIED", "DATE_MISMATCH"]}
GUIDED_CONTRADICTION_TYPES = ["PRIOR_CONDITION", "TESTIFIED_NO_PRIOR", "ACCIDENT_DATE", "TESTIFIED_ACCIDENT_DATE"]

TYPE_HELP = {
    "ACCIDENT_DATE": "ACCIDENT_DATE: date is the accident date stated in a medical record. No value needed.",
    "DIAGNOSIS": "DIAGNOSIS: date is the date of service of that visit, value is one diagnosis.",
    "MEDICATION": "MEDICATION: date is the date prescribed or dispensed, value is the drug name.",
    "PRIOR_CONDITION": "PRIOR_CONDITION: value is a body part the plaintiff had problems with before the accident, per a medical record.",
    "MRI_FINDING": "MRI_FINDING: value is one finding.",
    "PRIOR_DENIED": "PRIOR_DENIED: value is a body part the plaintiff denied problems with under oath, while a medical record shows a prior problem. Cite the deposition and the record.",
    "DATE_MISMATCH": "DATE_MISMATCH: the deposition gives a different accident date from the records. Cite both.",
    "TESTIFIED_NO_PRIOR": "TESTIFIED_NO_PRIOR: value is a body part the plaintiff said under oath they had no problems with before the accident.",
    "TESTIFIED_ACCIDENT_DATE": "TESTIFIED_ACCIDENT_DATE: date is the accident date the plaintiff gave in the deposition.",
}

CITE = {
    "lines": ('{{"file": "F03", "page": 2, "lines": [7]}}',
              "Cite the L numbers shown on the page. For a deposition answer, cite the question line and the answer line."),
    "quotes": ('{{"file": "F03", "page": 2, "quote": "words copied exactly from that page"}}',
               "The quote must be copied exactly from a page you opened."),
}

RULES = """Rules
1. Every claim needs a citation. {cite_rule}
2. Only the plaintiff counts: {name}, date of birth {dob}, record number {mrn}. Ignore any page about someone else.
3. Leave out anything denied, ruled out, excluded, negative, not seen, entered in error, or about a relative.
4. If nothing answers the question, return an empty list of claims."""

AGENT_SYSTEM = """You answer one question about a legal case folder of medical records and a deposition.
You cannot see the files. Call one tool per reply, by replying with a single JSON object and nothing else.

Tools
{{"tool": "list_files"}}
{{"tool": "search_map", "query": "a few words", "doc_types": ["medical_record"]}}   doc_types is optional. Types: medical_record, therapy_notes, radiology, pharmacy, billing, deposition
{{"tool": "open_page", "file": "F03", "page": 2}}
{{"tool": "finish", "claims": [{{"type": "...", "date": "YYYY-MM-DD", "value": "...", "citations": [{cite_example}]}}]}}

""" + RULES + """
5. You may open at most {budget} pages and take at most {steps} steps. If the question asks for every item, keep reading until you have seen every relevant page.

Claim types for this question: {types}
{type_help}"""

GUIDED_SYSTEM = """You read pages from a legal case folder and extract facts. Reply with one JSON object and nothing else:
{{"claims": [{{"type": "...", "date": "YYYY-MM-DD", "value": "...", "citations": [{cite_example}]}}]}}

""" + RULES + """

Claim types to extract: {types}
{type_help}"""


class OllamaClient:
    """Ollama's chat API. num_ctx sets the context window. If a conversation grows past it, Ollama
    drops the oldest messages without saying so, including the instructions, so keep it large."""

    def __init__(self, model, host="http://localhost:11434", timeout=600, num_ctx=16384):
        self.model, self.host, self.timeout, self.num_ctx = model, host.rstrip("/"), timeout, num_ctx

    def chat(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "stream": False, "format": "json",
                           "options": {"temperature": 0, "num_ctx": self.num_ctx}}).encode()
        req = urllib.request.Request(self.host + "/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]


def parse_json(text):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def parse_action(text):
    obj = parse_json(text)
    return obj if obj and isinstance(obj.get("tool"), str) else None


def number_lines(text):
    return "\n".join(f"L{i} | {ln}" for i, ln in enumerate(text.split("\n"), 1))


def _find(options, text):
    t = (text or "").lower()
    return next((o for o in options if o in t), None)


def resolve_citation(c, env):
    """A citation as file, page and quote. Line numbers are turned into the exact text of those lines;
    a line that does not exist gives an empty quote, which the verifier rejects."""
    out = {"file": str(c.get("file", "")), "page": c.get("page"), "quote": str(c.get("quote", "") or "")}
    nums = c.get("lines", c.get("line"))
    if nums is None:
        return out
    nums = nums if isinstance(nums, list) else [nums]
    text = env.page_text(out["file"], out["page"])
    try:
        nums = sorted({int(n) for n in nums})
    except (TypeError, ValueError):
        return {**out, "quote": ""}
    if text is None:
        return {**out, "quote": ""}
    lines = text.split("\n")
    if not nums or any(n < 1 or n > len(lines) for n in nums):
        return {**out, "quote": ""}
    return {**out, "quote": "\n".join(lines[n - 1] for n in nums)}


def to_internal(raw, env):
    """A model's claim as the internal fact and citations, or None if it cannot be read."""
    if not isinstance(raw, dict):
        return None
    t = str(raw.get("type", "")).upper()
    date = parse_date(str(raw.get("date") or "")) or parse_date(str(raw.get("value") or ""))
    value = str(raw.get("value") or "").strip().rstrip(".").lower()
    cits = [resolve_citation(c, env) for c in raw.get("citations", []) if isinstance(c, dict)]
    if t in ("ACCIDENT_DATE", "TESTIFIED_ACCIDENT_DATE") and date:
        fact = [t, date]
    elif t == "DIAGNOSIS" and date and value:
        fact = ["DIAGNOSIS", date, value]
    elif t == "MEDICATION" and date and _find(DRUGS, value):
        fact = ["MEDICATION", date, _find(DRUGS, value)]
    elif t in ("PRIOR_CONDITION", "TESTIFIED_NO_PRIOR") and _find(BODY, value):
        fact = [t, _find(BODY, value)]
    elif t == "MRI_FINDING" and value:
        fact = ["MRI_FINDING", value]
    elif t in ("PRIOR_DENIED", "DATE_MISMATCH"):
        fact = ["PRIOR_DENIED", _find(BODY, value)] if t == "PRIOR_DENIED" else ["DATE_MISMATCH"]
        if fact[-1] is None:
            return None
        for c in cits:                          # each side of a contradiction supports its own fact
            doc_type, _ = env.file_patient(c["file"])
            d = parse_date(c["quote"])
            if t == "PRIOR_DENIED":
                c["fact"] = ["TESTIFIED_NO_PRIOR", fact[1]] if doc_type == "deposition" else ["PRIOR_CONDITION", fact[1]]
            else:
                c["fact"] = ["TESTIFIED_ACCIDENT_DATE" if doc_type == "deposition" else "ACCIDENT_DATE", d or ""]
    else:
        return None
    return {"fact": fact, "citations": cits}


def _prompt_args(env, types, citations, **extra):
    me = env.case["plaintiff"]
    example, rule = CITE[citations]
    return dict(name=me["name"], dob=me["dob"], mrn=me["mrn"], types=", ".join(types),
                type_help="\n".join(TYPE_HELP[t] for t in types), cite_example=example.format(), cite_rule=rule, **extra)


class LLMAgent:
    name = "llm_agent"

    def __init__(self, client, budget=8, max_steps=20, citations="lines", nudges=1):
        if citations not in CITE:
            raise ValueError("citations must be lines or quotes")
        self.client, self.budget, self.max_steps, self.citations, self.nudges = client, budget, max_steps, citations, nudges
        self.log = []

    def answer(self, env, question):
        types = TYPES[question["id"]]
        system = AGENT_SYSTEM.format(**_prompt_args(env, types, self.citations, budget=self.budget, steps=self.max_steps))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "Question: " + question["text"]}]
        opened, hits, pending, nudges_left = [], [], None, self.nudges
        entry = {"question": question["id"], "steps": 0, "bad_replies": 0, "finished": False, "unreadable_claims": 0,
                 "nudges": 0, "pages": 0}
        self.log.append(entry)
        for _ in range(self.max_steps):
            entry["steps"] += 1
            reply = self.client.chat(messages)
            action = parse_action(reply)
            messages.append({"role": "assistant", "content": reply if isinstance(reply, str) else json.dumps(reply)})
            if action is None:
                entry["bad_replies"] += 1
                messages.append({"role": "user", "content": "Reply with one JSON object that calls a tool."})
                continue
            tool = action["tool"]
            if tool == "finish":
                claims = []
                for raw in action.get("claims") or []:
                    c = to_internal(raw, env)
                    if c is None:
                        entry["unreadable_claims"] += 1
                    else:
                        claims.append(c)
                unopened = [h for h in hits if h not in opened]
                room = self.budget - len(opened)
                if nudges_left > 0 and unopened and room > 0:
                    nudges_left -= 1
                    entry["nudges"] += 1
                    pending = claims
                    listed = ", ".join(f"{f} page {p}" for f, p in unopened[:room])
                    messages.append({"role": "user", "content":
                                     f"Before you finish: you opened {len(opened)} of your {self.budget} pages. These pages from "
                                     f"your searches are still unopened: {listed}. If the question asks for every item, open "
                                     f"them first. Then call finish again with all of your claims."})
                    continue
                entry["finished"] = True
                entry["pages"] = len(opened)
                return claims
            if tool == "list_files":
                result = env.list_files()
            elif tool == "search_map":
                dt = action.get("doc_types")
                result = env.search_map(str(action.get("query", "")), dt if isinstance(dt, list) else None, top=8)
                for h in result:
                    if (h["file"], h["page"]) not in hits:
                        hits.append((h["file"], h["page"]))
            elif tool == "open_page":
                try:
                    loc = (str(action.get("file")), int(action.get("page")))
                except (TypeError, ValueError):
                    loc = (str(action.get("file")), action.get("page"))
                if loc not in opened and len(opened) >= self.budget:
                    result = {"error": f"page budget of {self.budget} reached, finish now"}
                else:
                    result = env.open_page(*loc)
                    if "text" in result:
                        if loc not in opened:
                            opened.append(loc)
                        if self.citations == "lines":
                            result = {**result, "text": number_lines(result["text"])}
            else:
                result = {"error": f"unknown tool {tool}"}
            messages.append({"role": "user", "content": "Tool result: " + json.dumps(result)})
        entry["pages"] = len(opened)
        return pending or []                     # step limit: keep the last claims it offered, if any


class GuidedReader:
    """The fixed workflow picks the pages; the model reads them and extracts claims in one call."""

    name = "guided"

    def __init__(self, client, budget=8, citations="lines", retries=1):
        self.client, self.budget, self.citations, self.retries = client, budget, citations, retries
        self.log = []

    def answer(self, env, question):
        qid = question["id"]
        types = GUIDED_CONTRADICTION_TYPES if qid == "contradictions" else TYPES[qid]
        pages = []
        for fid, p in question_pages(env, qid, self.budget):
            r = env.open_page(fid, p)
            if "text" in r:
                doc_type, who = env.file_patient(fid)
                title = next(f["title"] for f in env.case["files"] if f["file_id"] == fid)
                text = number_lines(r["text"]) if self.citations == "lines" else r["text"]
                pages.append(f"=== File {fid}, page {p}, {title}, type {doc_type} ===\n{text}")
        entry = {"question": qid, "pages": len(pages), "calls": 0, "bad_replies": 0, "unreadable_claims": 0}
        self.log.append(entry)
        if not pages:
            return []
        system = GUIDED_SYSTEM.format(**_prompt_args(env, types, self.citations))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Question: {question['text']}\n\n" + "\n\n".join(pages)}]
        obj = None
        for _ in range(1 + self.retries):
            entry["calls"] += 1
            obj = parse_json(self.client.chat(messages))
            if obj is not None and isinstance(obj.get("claims"), list):
                break
            entry["bad_replies"] += 1
            obj = None
            messages.append({"role": "user", "content": 'Reply with one JSON object of the form {"claims": [...]}.'})
        claims = []
        for raw in (obj or {}).get("claims", []):
            c = to_internal(raw, env)
            if c is None or c["fact"][0] not in types:
                entry["unreadable_claims"] += 1
            else:
                claims.append(c)
        if qid == "contradictions":
            return contradictions(merge(claims))           # the comparison is done by code
        return merge(claims)
