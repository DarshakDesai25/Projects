"""Run the test bench on an extractor and write tables and figures.

  python run.py                                                    # rule based reader, all five levels
  python run.py --extractor ollama --model mistral --cases 20 --out results/mistral
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import extract_llm
import extract_rules
import generate as G
import score as S

LEVEL_TEXT = {0: "0 clean", 1: "1 date styles", 2: "2 wording and wrapping", 3: "3 long and cluttered", 4: "4 badly scanned"}
COLORS = {"MISSED": "C0", "WRONG_VALUE": "C1", "NEGATION_ERROR": "C3", "WRONG_PERSON": "C4", "UNGROUNDED": "C5", "EXTRA": "C7"}


def run_case(extract, case):
    out = dict(types={t: Counter() for t in S.FIELDS}, mist=Counter(), grounded=0, extracted=0, truth=0, details=[])
    facts = {}
    for doc, truth, name in [("record", case["truth_record"], "record"), ("deposition", case["truth_depo"], "deposition")]:
        facts[doc] = extract(case[doc], name)
        by_type, mist, grounded, n, details = S.score_doc(facts[doc], truth, case["traps"], case[doc])
        for t, c in by_type.items():
            out["types"][t].update(c)
        out["mist"].update(mist)
        out["grounded"] += grounded
        out["extracted"] += n
        out["truth"] += len(truth)
        out["details"] += details
    found = S.find_inconsistencies(facts["record"], facts["deposition"])
    want = case["inconsistencies"]
    out["inc"] = (len(found & want), len(found - want), len(want - found))
    out["facts"] = facts
    return out


def totals(cases):
    tp = sum(sum(c["types"][t]["tp"] for t in S.FIELDS) for c in cases)
    fp = sum(sum(c["types"][t]["fp"] for t in S.FIELDS) for c in cases)
    fn = sum(sum(c["types"][t]["fn"] for t in S.FIELDS) for c in cases)
    return tp, fp, fn


def f1_interval(cases, rng, boot=1000):
    per = np.array([[sum(c["types"][t][k] for t in S.FIELDS) for k in ("tp", "fp", "fn")] for c in cases], float)
    vals = []
    for _ in range(boot):
        s = per[rng.integers(0, len(per), len(per))].sum(axis=0)
        vals.append(2 * s[0] / max(1, 2 * s[0] + s[1] + s[2]))
    return np.percentile(vals, [2.5, 97.5])


def summarize(levels):
    rng = np.random.default_rng(0)
    rows = {}
    for lv, cases in levels.items():
        tp, fp, fn = totals(cases)
        truth = sum(c["truth"] for c in cases)
        itp, ifp, ifn = (sum(c["inc"][i] for c in cases) for i in range(3))
        lo, hi = f1_interval(cases, rng)
        rows[lv] = dict(
            precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn), f1_ci=(lo, hi),
            grounded=sum(c["grounded"] for c in cases) / max(1, sum(c["extracted"] for c in cases)),
            inc_recall=itp / max(1, itp + ifn), inc_precision=itp / max(1, itp + ifp),
            mistakes={e: 100 * sum(c["mist"][e] for c in cases) / truth for e in S.ERRORS},
            by_type={t: dict(recall=sum(c["types"][t]["tp"] for c in cases) / max(1, sum(c["types"][t]["tp"] + c["types"][t]["fn"] for c in cases)),
                             precision=sum(c["types"][t]["tp"] for c in cases) / max(1, sum(c["types"][t]["tp"] + c["types"][t]["fp"] for c in cases)))
                     for t in S.FIELDS},
            n_cases=len(cases), n_truth=truth)
    return rows


def tables(rows, path):
    lines = ["| Messiness level | Precision | Recall | F1 (95% range) | Quotes found in document | Contradictions caught | False contradictions |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for lv, r in rows.items():
        lines.append(f"| {LEVEL_TEXT[lv]} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} ({r['f1_ci'][0]:.3f} to {r['f1_ci'][1]:.3f}) | "
                     f"{r['grounded']:.1%} | {r['inc_recall']:.0%} | {1 - r['inc_precision']:.0%} of flags |")
    lines += ["", "Mistakes per 100 true facts.", "",
              "| Messiness level | " + " | ".join(S.ERRORS) + " |", "| --- |" + " --- |" * len(S.ERRORS)]
    for lv, r in rows.items():
        lines.append(f"| {LEVEL_TEXT[lv]} | " + " | ".join(f"{r['mistakes'][e]:.1f}" for e in S.ERRORS) + " |")
    lines += ["", "Recall by kind of fact.", "", "| Messiness level | " + " | ".join(S.FIELDS) + " |", "| --- |" + " --- |" * len(S.FIELDS)]
    for lv, r in rows.items():
        lines.append(f"| {LEVEL_TEXT[lv]} | " + " | ".join(f"{r['by_type'][t]['recall']:.2f}" for t in S.FIELDS) + " |")
    text = "\n".join(lines)
    open(path, "w").write(text + "\n")
    return text


def figures(rows, out):
    lv = list(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(lv, [rows[l]["precision"] for l in lv], "o-", color="C3", label="precision (of what it extracted, how much was true)")
    ax.plot(lv, [rows[l]["recall"] for l in lv], "o-", color="C0", label="recall (of the true facts, how many it found)")
    ax.plot(lv, [rows[l]["inc_recall"] for l in lv], "s--", color="C2", label="contradictions caught")
    ax.fill_between(lv, [rows[l]["f1_ci"][0] for l in lv], [rows[l]["f1_ci"][1] for l in lv], color="gray", alpha=0.2, label="F1 95% range")
    ax.set_xticks(lv)
    ax.set_xticklabels([LEVEL_TEXT[l].replace(" ", "\n", 1) for l in lv], fontsize=8)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("How the reader holds up as the documents get messier", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "quality_by_level.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bottom = np.zeros(len(lv))
    for e in S.ERRORS:
        v = np.array([rows[l]["mistakes"][e] for l in lv])
        ax.bar(range(len(lv)), v, bottom=bottom, color=COLORS[e], label=e)
        bottom += v
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels([LEVEL_TEXT[l].replace(" ", "\n", 1) for l in lv], fontsize=8)
    ax.set_ylabel("mistakes per 100 true facts")
    ax.legend(fontsize=7)
    ax.set_title("What kind of mistakes it makes", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mistakes_by_level.png"), dpi=150)


def example(extract, path, level=2):
    cases = G.make_cases(0, level, 30)
    runs = [(c, run_case(extract, c)) for c in cases]
    case, run = max(runs, key=lambda cr: len(set(lab for lab, _ in cr[1]["details"])))
    lines = [f"# One level {level} case, read by the extractor", "", "## Medical record", "```", case["record"], "```", "",
             "## Deposition", "```", case["deposition"], "```", "", "## Mistakes", ""]
    quotes = {S.key(f): f.get("quote", "") for fs in run["facts"].values() for f in fs}
    for label, k in sorted(run["details"]):
        q = quotes.get(k)
        lines.append(f"- {label}: {' | '.join(k)}" + (f"  (it quoted: {q!r})" if q else ""))
    lines += ["", "Contradictions in the case: " + (", ".join(" ".join(i) for i in sorted(case["inconsistencies"])) or "none"),
              "Contradictions flagged: " + (", ".join(" ".join(i) for i in sorted(S.find_inconsistencies(*run["facts"].values()))) or "none")]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["rules", "ollama"], default="rules")
    ap.add_argument("--model", help="name of the Ollama model, for example mistral")
    ap.add_argument("--cases", type=int, default=100, help="cases per messiness level")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.extractor == "ollama":
        if not a.model:
            raise SystemExit("--model is required with --extractor ollama")
        extract = extract_llm.make_extractor(extract_llm.ollama_generate(a.model))
    else:
        extract = extract_rules.extract

    levels = {}
    for lv in a.levels:
        levels[lv] = [run_case(extract, c) for c in G.make_cases(a.seed, lv, a.cases)]
        print(f"level {lv} done", flush=True)
    rows = summarize(levels)
    print("\n" + tables(rows, os.path.join(a.out, "summary.md")))
    figures(rows, a.out)
    example(extract, os.path.join(a.out, "example.md"), level=2 if 2 in levels else a.levels[0])
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({str(k): {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in rows.items()}, f, indent=2)
    print("\nsaved to", a.out)


if __name__ == "__main__":
    main()
