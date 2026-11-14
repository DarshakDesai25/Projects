import json

from agents import FactsTable, FixedWorkflow, apply_retractions, contradictions, merge
from dates import find_dates
from generate import key, make_case, make_cases
from llm_agent import GuidedReader, LLMAgent, OllamaClient, parse_action, resolve_citation, to_internal
from navmap import NavMap
from reader import read_page
from score import classify, score_question
from tools import CaseEnv
from verify import verify


def page_of(case, fid, p):
    f = next(f for f in case["files"] if f["file_id"] == fid)
    return "\n".join(f["pages"][p - 1])


# the generator and its answer key

def test_cases_are_reproducible():
    assert make_case(1, "large", 4) == make_case(1, "large", 4)
    assert make_case(1, "large", 4) != make_case(2, "large", 4)


def test_every_true_fact_is_on_every_page_the_key_says():
    for c in make_cases(0, "large", 20) + make_cases(0, "small", 20):
        for q, facts in c["gold"].items():
            if q == "contradictions":
                continue
            for fact in facts:
                locs = c["locations"][key(fact)]
                assert locs
                for fid, p in locs:
                    text = page_of(c, fid, p)
                    if fact[0] == "ACCIDENT_DATE":
                        assert any(d[2] == fact[1] for d in find_dates(text))
                    else:
                        assert fact[-1] in text.lower()


def test_pages_never_split_a_visit():
    for c in make_cases(0, "large", 20):
        for f in c["files"]:
            for page in f["pages"]:
                body = page[1:]
                for i, ln in enumerate(body):
                    if ln.startswith(("Assessment:", "Plan: Start", "Plan: Prescribed", "Plan: Rx:")):
                        assert any(x.startswith("Date of service:") for x in body[:i])


def test_traps_are_never_true_facts_and_large_cases_have_every_kind():
    kinds = set()
    for c in make_cases(0, "large", 30):
        gold = {key(f) for g in c["gold"].values() for f in g}
        assert not gold & set(c["traps"])
        kinds |= set(c["traps"].values())
    assert {"NEGATION_ERROR", "WRONG_PERSON", "WRONG_PERSON_FILE", "RETRACTED"} <= kinds


def test_the_other_person_shares_the_name_but_not_the_identity():
    c = make_case(0, "large", 2)
    others = [f for f in c["files"] if f["doc_type"] != "deposition" and f["patient"]["dob"] != c["plaintiff"]["dob"]]
    assert others and all(f["patient"]["name"] == c["plaintiff"]["name"] for f in others)


# the map and the tools

def test_the_map_finds_pages_and_never_returns_text():
    c = next(c for c in make_cases(0, "large", 10) if c["gold"]["mri"])
    hits = NavMap(c).search("mri findings impression", ["radiology"])
    assert hits and all(h["doc_type"] == "radiology" for h in hits)
    assert all("text" not in h for h in hits)


def test_pages_are_counted_once_and_the_budget_is_enforced():
    c = make_case(0, "large", 0)
    env = CaseEnv(c, page_budget=2)
    f = c["files"][0]["file_id"]
    assert "text" in env.open_page(f, 1)
    assert "text" in env.open_page(f, 1) and len(env.opened) == 1
    second = c["files"][1]["file_id"]
    assert "text" in env.open_page(second, 1)
    assert env.open_page(c["files"][2]["file_id"], 1) == {"error": "page budget reached"}
    assert "error" in env.open_page("F99", 1) and "error" in env.open_page(f, 99)


# the agents

def test_reading_everything_finds_every_fact_in_clean_small_cases():
    for c in make_cases(0, "small", 20):
        env, agent = CaseEnv(c), FactsTable()
        for q in c["questions"]:
            s = score_question(q["id"], verify(env, agent.answer(env, q))[0], c)
            assert s["found"] == s["gold"] and sum(v for k, v in s["mistakes"].items() if k != "MISSED") == 0


def test_a_smaller_budget_reads_fewer_pages_and_misses_more():
    cases = make_cases(0, "large", 10)
    found, pages = {}, {}
    for b in (1, 16):
        f = p = 0
        for c in cases:
            env, agent = CaseEnv(c), FixedWorkflow(b)
            for q in c["questions"]:
                f += score_question(q["id"], verify(env, agent.answer(env, q))[0], c)["found"]
            p += len(env.opened)
        found[b], pages[b] = f, p
    assert found[1] < found[16] and pages[1] < pages[16]


def test_a_retraction_that_was_read_removes_the_diagnosis():
    claims = [{"fact": ["DIAGNOSIS", "2024-01-02", "whiplash"], "citations": []},
              {"fact": ["DIAGNOSIS", "2024-01-02", "cervical sprain"], "citations": []},
              {"fact": ["RETRACTION", "2024-01-02", "whiplash"], "citations": []}]
    assert [c["fact"][2] for c in apply_retractions(claims)] == ["cervical sprain"]


def test_contradictions_need_both_sides():
    dep = {"fact": ["TESTIFIED_NO_PRIOR", "neck"], "citations": [{"file": "F01", "page": 1, "quote": "q"}]}
    rec = {"fact": ["PRIOR_CONDITION", "neck"], "citations": [{"file": "F02", "page": 1, "quote": "r"}]}
    out = contradictions(merge([dep, rec]))
    assert out[0]["fact"] == ["PRIOR_DENIED", "neck"] and len(out[0]["citations"]) == 2
    assert contradictions([dep]) == []


# the verifier and the scorer

def sample_case():
    return next(c for c in make_cases(0, "large", 30) if c["gold"]["diagnoses"] and c["gold"]["prior"])


def test_verifier_rejects_made_up_quotes_wrong_patients_and_negations():
    c = sample_case()
    env = CaseEnv(c)
    good = c["gold"]["diagnoses"][0]
    fid, p = c["locations"][key(good)][0]
    ok = {"fact": good, "citations": [{"file": fid, "page": p, "quote": good[-1]}]}
    fake = {"fact": good, "citations": [{"file": fid, "page": p, "quote": "a sentence that is not there"}]}
    other = next(f for f in c["files"] if f["doc_type"] == "medical_record" and f["patient"]["dob"] != c["plaintiff"]["dob"])
    ln = next(x for x in other["pages"][0] if x.startswith("Assessment:"))
    wrong = {"fact": ["DIAGNOSIS", "2024-01-01", "x"], "citations": [{"file": other["file_id"], "page": 1, "quote": ln}]}
    neg_line = next(x for f in c["files"] if f["patient"]["dob"] == c["plaintiff"]["dob"] for pg in f["pages"] for x in pg
                    if "ruled out" in x or "was excluded" in x or "no evidence of" in x or "negative for" in x)
    nf = next(f for f in c["files"] for pg in f["pages"] if neg_line in pg)
    npg = next(i for i, pg in enumerate(nf["pages"], 1) if neg_line in pg)
    item = next(s.strip().rstrip(".") for s in neg_line.split(":", 1)[1].split(";") if any(w in s for w in ("ruled out", "excluded", "no evidence", "negative")))
    negated = {"fact": ["DIAGNOSIS", "2024-01-01", item.split(" ")[0]], "citations": [{"file": nf["file_id"], "page": npg, "quote": item}]}
    kept, removed, dropped = verify(env, [ok, fake, wrong, negated])
    assert kept == [ok] and dropped == 3
    assert removed == {"UNGROUNDED": 1, "WRONG_PERSON_FILE": 1, "UNSUPPORTED": 1}


def test_scorer_names_wrong_citations_and_reworded_negations():
    c = sample_case()
    good = c["gold"]["diagnoses"][0]
    elsewhere = next((f["file_id"], 1) for f in c["files"] if (f["file_id"], 1) not in {tuple(x) for x in c["locations"][key(good)]})
    s = score_question("diagnoses", [{"fact": good, "citations": [{"file": elsewhere[0], "page": 1, "quote": "x"}]}], c)
    assert s["found"] == 0 and s["mistakes"]["WRONG_CITATION"] == 1 and s["mistakes"]["MISSED"] == s["gold"]
    trap = next(k for k, v in c["traps"].items() if v == "NEGATION_ERROR" and k.startswith("DIAGNOSIS"))
    t, d, name = trap.split("|")
    assert classify([t, d, f"{name} was excluded"], c) == "NEGATION_ERROR"


def test_an_empty_answer_is_right_when_there_is_nothing_to_find():
    c = next(c for c in make_cases(0, "large", 30) if not c["gold"]["mri"])
    assert score_question("mri", [], c)["empty_and_right"]


# the language model agent, with a scripted stand in for the model

class Scripted:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages):
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else json.dumps({"tool": "list_files"})


def test_llm_agent_runs_tools_and_returns_checkable_claims():
    c = sample_case()
    env = CaseEnv(c)
    good = c["gold"]["diagnoses"][0]
    fid, p = c["locations"][key(good)][0]
    script = Scripted([json.dumps({"tool": "search_map", "query": "assessment", "doc_types": ["medical_record"]}),
                       "not json at all",
                       json.dumps({"tool": "open_page", "file": fid, "page": p}),
                       json.dumps({"tool": "finish", "claims": [
                           {"type": "DIAGNOSIS", "date": good[1], "value": good[2], "citations": [{"file": fid, "page": p, "quote": good[2]}]},
                           {"type": "NOT_A_TYPE", "value": "x"}]})])
    agent = LLMAgent(script, budget=3, nudges=0)
    claims = agent.answer(env, next(q for q in c["questions"] if q["id"] == "diagnoses"))
    assert [x["fact"] for x in claims] == [good]
    assert agent.log[0]["bad_replies"] == 1 and agent.log[0]["unreadable_claims"] == 1 and agent.log[0]["finished"]
    assert len(env.opened) == 1 and verify(env, claims)[0] == claims


def test_llm_agent_respects_the_page_budget_and_the_step_limit():
    c = make_case(0, "large", 1)
    env = CaseEnv(c)
    opens = [json.dumps({"tool": "open_page", "file": f["file_id"], "page": 1}) for f in c["files"][:3]]
    agent = LLMAgent(Scripted(opens), budget=2, max_steps=5)
    assert agent.answer(env, c["questions"][0]) == []
    assert len(env.opened) == 2 and not agent.log[0]["finished"]


def test_model_claims_are_normalized_and_contradiction_sides_are_labeled():
    c = sample_case()
    env = CaseEnv(c)
    dep = next(f["file_id"] for f in c["files"] if f["doc_type"] == "deposition")
    rec = next(f["file_id"] for f in c["files"] if f["doc_type"] == "medical_record")
    m = to_internal({"type": "medication", "date": "March 3, 2024", "value": "Ibuprofen 600 mg", "citations": []}, env)
    assert m["fact"] == ["MEDICATION", "2024-03-03", "ibuprofen"]
    x = to_internal({"type": "PRIOR_DENIED", "value": "the neck", "citations": [{"file": dep, "page": 1, "quote": "a"}, {"file": rec, "page": 1, "quote": "b"}]}, env)
    assert [k["fact"][0] for k in x["citations"]] == ["TESTIFIED_NO_PRIOR", "PRIOR_CONDITION"]
    assert parse_action('Sure. {"tool": "list_files"} done') == {"tool": "list_files"} and parse_action("{}") is None


def test_ollama_client_sends_a_json_chat_request(monkeypatch):
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"message": {"content": "{\"tool\": \"list_files\"}"}}).encode()

    def fake_urlopen(req, timeout):
        sent.update(url=req.full_url, body=json.loads(req.data))
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = OllamaClient("qwen2.5:7b").chat([{"role": "user", "content": "hi"}])
    assert out == "{\"tool\": \"list_files\"}" and sent["url"].endswith("/api/chat")
    assert sent["body"]["format"] == "json" and sent["body"]["options"]["temperature"] == 0 and sent["body"]["stream"] is False
    assert sent["body"]["options"]["num_ctx"] == 16384


def test_the_model_run_answers_once_and_scores_before_and_after_the_checks(tmp_path=None, monkeypatch=None):
    import argparse
    import tempfile
    from pathlib import Path

    import llm_agent
    import run as R
    calls = {"n": 0}

    class Fake:
        def __init__(self, *a, **k):
            pass

        def chat(self, messages):
            calls["n"] += 1
            return json.dumps({"tool": "finish", "claims": []})
    real = llm_agent.OllamaClient
    llm_agent.OllamaClient = Fake
    try:
        out = Path(tempfile.mkdtemp())
        args = argparse.Namespace(seed=0, size="small", cases=2, model="fake", host="x", budget=4, steps=5, num_ctx=1024, out=out,
                                  mode="agent", citations="lines", nudges=0)
        R.llm(args, out)
    finally:
        llm_agent.OllamaClient = real
    assert calls["n"] == 2 * 6                    # one finish per question, not two
    res = json.loads(next(out.glob("llm_*.json")).read_text())
    assert {"unverified", "verified", "log", "minutes"} <= set(res)


def test_line_citations_become_the_exact_page_text_and_bad_lines_are_rejected():
    c = sample_case()
    env = CaseEnv(c)
    good = c["gold"]["diagnoses"][0]
    fid, p = c["locations"][key(good)][0]
    lines = env.page_text(fid, p).split("\n")
    n = next(i for i, ln in enumerate(lines, 1) if ln.startswith("Assessment:") and good[2] in ln.lower())
    cit = resolve_citation({"file": fid, "page": p, "lines": [n]}, env)
    assert cit["quote"] == lines[n - 1]
    assert resolve_citation({"file": fid, "page": p, "lines": [999]}, env)["quote"] == ""
    claim = to_internal({"type": "DIAGNOSIS", "date": good[1], "value": good[2], "citations": [{"file": fid, "page": p, "lines": [n]}]}, env)
    assert verify(env, [claim])[0] == [claim]
    bad = to_internal({"type": "DIAGNOSIS", "date": good[1], "value": good[2], "citations": [{"file": fid, "page": p, "line": 999}]}, env)
    assert verify(env, [bad])[1] == {"UNGROUNDED": 1}


def test_the_nudge_sends_the_model_back_once_when_its_own_search_hits_are_unread():
    c = sample_case()
    env = CaseEnv(c)
    hits = env.map.search("assessment", ["medical_record"], top=8)
    first, second = (hits[0]["file"], hits[0]["page"]), (hits[1]["file"], hits[1]["page"])
    finish = json.dumps({"tool": "finish", "claims": []})
    replies = [json.dumps({"tool": "search_map", "query": "assessment", "doc_types": ["medical_record"]}),
               json.dumps({"tool": "open_page", "file": first[0], "page": first[1]}), finish,
               json.dumps({"tool": "open_page", "file": second[0], "page": second[1]}), finish]
    script = Scripted(replies)
    agent = LLMAgent(script, budget=4, nudges=1)
    agent.answer(env, next(q for q in c["questions"] if q["id"] == "diagnoses"))
    assert agent.log[0]["nudges"] == 1 and agent.log[0]["finished"] and agent.log[0]["pages"] == 2
    assert any("still unopened" in m for m in script.seen)
    env2 = CaseEnv(c)
    agent2 = LLMAgent(Scripted(replies[:3]), budget=4, nudges=0)
    agent2.answer(env2, next(q for q in c["questions"] if q["id"] == "diagnoses"))
    assert agent2.log[0]["nudges"] == 0 and agent2.log[0]["pages"] == 1


def test_guided_mode_reads_the_workflow_pages_and_code_finds_the_contradiction():
    c = next(c for c in make_cases(0, "large", 60) if any(g[0] == "PRIOR_DENIED" for g in c["gold"]["contradictions"]))
    bp = next(g[1] for g in c["gold"]["contradictions"] if g[0] == "PRIOR_DENIED")
    env = CaseEnv(c)

    def loc_line(fact_key, test):
        fid, p = c["locations"][fact_key][0]
        lines = env.page_text(fid, p).split("\n")
        return fid, p, [i for i, ln in enumerate(lines, 1) if test(ln, lines, i)]
    rf, rp, rl = loc_line(key(("PRIOR_CONDITION", bp)), lambda ln, L, i: bp in ln.lower() and "history" in ln.lower() or f"prior {bp}" in ln.lower())
    df, dp, dl = loc_line(key(("TESTIFIED_NO_PRIOR", bp)), lambda ln, L, i: f"problems with your {bp}" in ln or (i > 1 and f"problems with your {bp}" in L[i - 2]))

    class Reader:
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return json.dumps({"claims": [
                {"type": "PRIOR_CONDITION", "value": bp, "citations": [{"file": rf, "page": rp, "lines": rl[:1]}]},
                {"type": "TESTIFIED_NO_PRIOR", "value": bp, "citations": [{"file": df, "page": dp, "lines": dl[:2]}]}]})
    model = Reader()
    agent = GuidedReader(model, budget=8)
    q = next(q for q in c["questions"] if q["id"] == "contradictions")
    claims = verify(env, agent.answer(env, q))[0]
    assert model.calls == 1 and agent.log[0]["pages"] == len(env.opened) > 0
    s = score_question("contradictions", claims, c)
    assert ["PRIOR_DENIED", bp] in [x["fact"] for x in claims] and s["found"] >= 1
