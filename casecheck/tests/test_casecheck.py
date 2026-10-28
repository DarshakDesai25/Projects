import extract_llm
import extract_rules
import generate as G
import run
import score as S
from dates import find_dates, parse_date


def test_date_formats():
    for text in ["March 3, 2023", "03/03/2023", "3 Mar 2023", "2023-03-03", "03/03/23", "It was on 3 March 2023."]:
        assert parse_date(text) == "2023-03-03", text
    assert parse_date("no date here") is None
    assert parse_date("13/45/2023") is None            # not a real date


def test_find_dates_in_order_without_overlap():
    hits = find_dates("seen 2023-03-03 and again March 9, 2023")
    assert [h[2] for h in hits] == ["2023-03-03", "2023-03-09"]


def test_generator_is_deterministic():
    a, b = G.make_case(1, 3, 2), G.make_case(1, 3, 2)
    assert a["record"] == b["record"] and a["truth_record"] == b["truth_record"]


def test_answer_key_and_traps():
    c = G.make_case(0, 1, 0)
    kinds = {k[0] for k in c["truth_record"]}
    assert {"ACCIDENT_DATE", "VISIT", "DIAGNOSIS"} <= kinds
    assert set(c["traps"].values()) >= {"NEGATION_ERROR", "WRONG_PERSON"}
    assert not (set(c["traps"]) & c["truth_record"])      # a trap is never also a true fact


def test_messier_levels_are_longer_and_scanned_text_differs():
    assert len(G.make_case(0, 3, 0)["record"]) > len(G.make_case(0, 0, 0)["record"])
    assert G.make_case(0, 4, 0)["record"] != G.make_case(0, 3, 0)["record"]


def test_rules_reader_is_perfect_on_clean_documents():
    for level in (0, 1):
        for c in G.make_cases(5, level, 20):
            r = run.run_case(extract_rules.extract, c)
            assert sum(r["mist"].values()) == 0, (level, c["id"], r["details"])


def test_inconsistencies_found_from_extracted_facts():
    rec = [dict(type="PRIOR_CONDITION", body_part="neck"), dict(type="ACCIDENT_DATE", date="2023-03-03")]
    dep = [dict(type="TESTIFIED_NO_PRIOR", body_part="neck"), dict(type="TESTIFIED_ACCIDENT_DATE", date="2023-03-10")]
    assert S.find_inconsistencies(rec, dep) == {("PRIOR_DENIED", "neck"), ("DATE_MISMATCH",)}
    assert S.find_inconsistencies(rec, [dict(type="TESTIFIED_ACCIDENT_DATE", date="2023-03-03")]) == set()


# scoring

DOC = "Date of service: 2023-03-04 Provider: Dr. Patel Assessment: lumbar strain; fracture ruled out."
TRUTH = {("VISIT", "2023-03-04", "dr. patel"), ("DIAGNOSIS", "2023-03-04", "lumbar strain")}
TRAPS = {("DIAGNOSIS", "2023-03-04", "fracture"): "NEGATION_ERROR", ("PRIOR_CONDITION", "neck"): "WRONG_PERSON"}


def mistakes(facts):
    return dict(S.score_doc(facts, TRUTH, TRAPS, DOC)[1])


def fact(t, quote, **kw):
    return dict(type=t, quote=quote, **kw)


def test_perfect_reading_has_no_mistakes():
    f = [fact("VISIT", "Provider: Dr. Patel", date="2023-03-04", provider="Dr. Patel"),
         fact("DIAGNOSIS", "lumbar strain", date="2023-03-04", name="lumbar strain")]
    assert sum(mistakes(f).values()) == 0


def test_missed_fact():
    f = [fact("VISIT", "Provider: Dr. Patel", date="2023-03-04", provider="dr. patel")]
    assert mistakes(f) == {"MISSED": 1}


def test_wrong_value_is_one_mistake_not_two():
    f = [fact("VISIT", "Provider: Dr. Patel", date="2023-03-04", provider="dr. patel"),
         fact("DIAGNOSIS", "lumbar strain", date="2023-03-05", name="lumbar strain")]
    assert mistakes(f) == {"WRONG_VALUE": 1}


def test_negation_and_wrong_person_traps():
    base = [fact("VISIT", "Provider: Dr. Patel", date="2023-03-04", provider="dr. patel"),
            fact("DIAGNOSIS", "lumbar strain", date="2023-03-04", name="lumbar strain")]
    m = mistakes(base + [fact("DIAGNOSIS", "fracture ruled out", date="2023-03-04", name="fracture")])
    assert m["NEGATION_ERROR"] == 1
    m = mistakes(base + [fact("DIAGNOSIS", "fracture ruled out", date="2023-03-04", name="fracture ruled out")])
    assert m["NEGATION_ERROR"] == 1                      # a differently worded version of the same trap
    m = mistakes(base + [fact("PRIOR_CONDITION", "Provider", body_part="neck")])
    assert m["WRONG_PERSON"] == 1


def test_invented_fact_with_invented_quote_is_flagged():
    f = [fact("VISIT", "Provider: Dr. Patel", date="2023-03-04", provider="dr. patel"),
         fact("DIAGNOSIS", "lumbar strain", date="2023-03-04", name="lumbar strain"),
         fact("MEDICATION", "this sentence is not in the document", date="2023-03-04", drug="ibuprofen")]
    assert mistakes(f) == {"UNGROUNDED": 1}


def test_correct_fact_with_invented_quote_counts_as_ungrounded_evidence():
    f = [fact("VISIT", "this sentence is not in the document", date="2023-03-04", provider="dr. patel"),
         fact("DIAGNOSIS", "lumbar strain", date="2023-03-04", name="lumbar strain")]
    _, mist, grounded, n, _ = S.score_doc(f, TRUTH, TRAPS, DOC)
    assert sum(mist.values()) == 0 and (grounded, n) == (1, 2)


def test_extra_fact_with_real_quote():
    f = [fact("VISIT", "Provider: Dr. Patel", date="2023-03-04", provider="dr. patel"),
         fact("DIAGNOSIS", "lumbar strain", date="2023-03-04", name="lumbar strain"),
         fact("MEDICATION", "Provider", date="2023-03-04", drug="ibuprofen")]
    assert mistakes(f) == {"EXTRA": 1}


# language model reader

def test_llm_output_parsing_is_forgiving():
    raw = '```json\n{"facts":[{"type":"ACCIDENT_DATE","date":"March 3, 2023","quote":"q"},{"type":"BOGUS"}]}\n```'
    assert extract_llm.parse_llm_output(raw) == [{"type": "ACCIDENT_DATE", "quote": "q", "date": "2023-03-03"}]
    assert extract_llm.parse_llm_output("not json") == []
    assert extract_llm.parse_llm_output('Sure! {"facts":[{"type":"PRIOR_CONDITION","body_part":"Neck"}]}')[0]["body_part"] == "neck"


def test_llm_extractor_plumbing_with_a_fake_model():
    c = G.make_case(0, 3, 0)
    seen = []

    def fake(prompt):
        seen.append(prompt)
        return '{"facts": [{"type": "ACCIDENT_DATE", "date": "2023-03-03", "quote": "MEDICAL RECORD SUMMARY"}]}'

    facts = extract_llm.make_extractor(fake, limit=800)(c["record"], "record")
    assert len(seen) >= 2 and all(f["type"] == "ACCIDENT_DATE" for f in facts)


def test_chunks_cover_the_whole_document():
    c = G.make_case(0, 3, 1)
    chunks = extract_llm.chunk_text(c["record"], "record", 800)
    assert len(chunks) > 1
    joined = "".join(chunks)
    assert all(p in joined for p in c["record"].split("\n\n") if p.strip())


def test_summary_has_every_level():
    levels = {lv: [run.run_case(extract_rules.extract, c) for c in G.make_cases(0, lv, 4)] for lv in (0, 2)}
    rows = run.summarize(levels)
    assert set(rows) == {0, 2} and rows[0]["f1"] == 1.0 and rows[2]["n_cases"] == 4
