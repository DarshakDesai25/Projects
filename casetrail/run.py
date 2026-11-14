"""Run the experiments and write the results.

  python run.py                          rule based agents on 100 small and 100 large cases
  python run.py --agent ollama --model qwen2.5:7b --size large --cases 5 --budget 8

Writes results/results.json, results/summary.md and three charts.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from agents import FactsTable, FixedWorkflow
from generate import make_cases
from score import KINDS, score_question
from tools import CaseEnv
from verify import verify

QIDS = ["accident", "diagnoses", "medications", "prior", "mri", "contradictions"]
BUDGETS = [1, 2, 4, 8, 16]


def run_agent(cases, make_agent, verified=True):
    """Totals by question, mistakes, pages read per case, and citations the verifier removed."""
    by_q = defaultdict(Counter)
    mistakes, removed, per_case = Counter(), Counter(), []
    for c in cases:
        env, agent = CaseEnv(c), make_agent()
        case_found = case_gold = 0
        for q in c["questions"]:
            claims = agent.answer(env, q)
            if verified:
                claims, rem, _ = verify(env, claims)
                removed += rem
            s = score_question(q["id"], claims, c)
            by_q[q["id"]].update(gold=s["gold"], found=s["found"], claims=s["claims"], empty_right=int(s["empty_and_right"]),
                                 empty_gold=int(s["gold"] == 0))
            mistakes += s["mistakes"]
            case_found += s["found"]
            case_gold += s["gold"]
        per_case.append({"found": case_found, "gold": case_gold, "pages": len(env.opened)})
    return {"by_question": {q: dict(v) for q, v in by_q.items()}, "mistakes": dict(mistakes), "removed": dict(removed),
            "pages_per_case": sum(x["pages"] for x in per_case) / len(per_case), "per_case": per_case}


def recall_ci(per_case, n=1000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [rng.choice(per_case) for _ in per_case]
        vals.append(sum(x["found"] for x in s) / max(1, sum(x["gold"] for x in s)))
    vals.sort()
    return round(vals[int(0.025 * n)], 3), round(vals[int(0.975 * n)], 3)


def totals(r):
    g = sum(v["gold"] for v in r["by_question"].values())
    f = sum(v["found"] for v in r["by_question"].values())
    c = sum(v["claims"] for v in r["by_question"].values())
    return {"recall": round(f / g, 3) if g else None, "precision": round(f / c, 3) if c else None,
            "recall_95": recall_ci(r["per_case"]), "pages_per_case": round(r["pages_per_case"], 1)}


def rules(args, out):
    results = {}
    for size in ["small", "large"]:
        cases = make_cases(args.seed, size, args.cases)
        pages = [sum(len(f["pages"]) for f in c["files"]) for c in cases]
        results[size] = {"cases": len(cases), "files_per_case": round(sum(len(c["files"]) for c in cases) / len(cases), 1),
                         "pages_per_case": round(sum(pages) / len(pages), 1), "runs": {}}
        runs = [(f"workflow_{b}", lambda b=b: FixedWorkflow(b)) for b in BUDGETS] + [("workflow_all", lambda: FixedWorkflow(None)),
                                                                                    ("facts_table", FactsTable)]
        for name, make in runs:
            for verified in (False, True):
                r = run_agent(cases, make, verified)
                label = f"{name}{'_verified' if verified else ''}"
                results[size]["runs"][label] = {**totals(r), "by_question": r["by_question"], "mistakes": r["mistakes"], "removed": r["removed"]}
                print(f"{size:5} {label:26} recall {results[size]['runs'][label]['recall']} precision {results[size]['runs'][label]['precision']} "
                      f"pages/case {results[size]['runs'][label]['pages_per_case']}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    charts(results, out)
    summary(results, out)


def qrecall(run, q):
    v = run["by_question"][q]
    return v["found"] / v["gold"] if v["gold"] else None


def charts(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = results["large"]["runs"]
    xs = BUDGETS + [32]
    labels = [str(b) for b in BUDGETS] + ["all"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    names = {"accident": "accident date", "diagnoses": "diagnoses", "medications": "medications", "prior": "prior conditions",
             "mri": "MRI findings", "contradictions": "contradictions"}
    for q in QIDS:
        ys = [qrecall(L[f"workflow_{b}_verified"], q) for b in BUDGETS] + [qrecall(L["workflow_all_verified"], q)]
        ax.plot(xs, ys, marker="o", label=names[q])
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("pages the agent may read for each question")
    ax.set_ylabel("share of true facts found")
    ax.set_ylim(0, 1.05)
    ax.set_title("Large cases: what a reading budget misses, after verification")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "recall_by_budget.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    groups = [("facts_table", "read everything"), ("workflow_4", "4 pages per question")]
    kinds = ["NEGATION_ERROR", "WRONG_PERSON", "WRONG_PERSON_FILE", "RETRACTED", "WRONG_VALUE", "EXTRA"]
    xpos, ticks = 0, []
    colors = dict(zip(kinds, ["#C8553D", "#F2A541", "#7A9E9F", "#3D5A80", "#8AB17D", "#B8B8B8"]))
    for g, glabel in groups:
        for suffix, slabel in [("", "before"), ("_verified", "after")]:
            bottom = 0
            m = L[g + suffix]["mistakes"]
            for k in kinds:
                v = m.get(k, 0)
                ax.bar(xpos, v, bottom=bottom, color=colors[k], label=k if xpos == 0 else None)
                bottom += v
            ax.text(xpos, bottom + 5, str(bottom), ha="center", fontsize=8)
            ticks.append((xpos, f"{glabel}\n{slabel} checks"))
            xpos += 1
        xpos += 0.5
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=8)
    ax.set_ylabel("wrong claims, 100 large cases")
    ax.set_title("Wrong claims before and after the citation checks")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "errors_before_after.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    for size, color in [("small", "#3A6EA5"), ("large", "#C8553D")]:
        R = results[size]["runs"]
        pts = [(R[f"workflow_{b}_verified"]["pages_per_case"], R[f"workflow_{b}_verified"]["recall"], str(b)) for b in BUDGETS]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=color, label=f"{size} cases, page budget per question")
        shown = set()
        for x, y, t in pts:
            if (round(x, 1), round(y, 3)) not in shown:          # label each distinct point once
                ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=color)
                shown.add((round(x, 1), round(y, 3)))
        ft = R["facts_table_verified"]
        ax.scatter([ft["pages_per_case"]], [ft["recall"]], marker="*", s=140, color=color, label=f"{size} cases, read everything once")
    ax.set_xlabel("pages read per case, all six questions")
    ax.set_ylabel("share of true facts found")
    ax.set_title("Cost and coverage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_recall.png")
    plt.close(fig)


def summary(results, out):
    lines = ["# Results", ""]
    for size in ["small", "large"]:
        R = results[size]
        lines += [f"## {size.capitalize()} cases", "",
                  f"{R['cases']} cases, about {R['files_per_case']} files and {R['pages_per_case']} pages each.", "",
                  "| Agent | Checks | Recall (95% range) | Precision | Pages read per case |", "| --- | --- | --- | --- | --- |"]
        for name, run in R["runs"].items():
            lo, hi = run["recall_95"]
            lines.append(f"| {name.replace('_verified', '')} | {'after' if name.endswith('_verified') else 'before'} | "
                         f"{run['recall']} ({lo} to {hi}) | {run['precision']} | {run['pages_per_case']} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def llm(args, out):
    """Each case is answered once by the model. The same claims are then scored before and after the checks."""
    import time
    from llm_agent import GuidedReader, LLMAgent, OllamaClient
    cases = make_cases(args.seed, args.size, args.cases)
    client = OllamaClient(args.model, args.host, num_ctx=args.num_ctx)
    by = {v: defaultdict(Counter) for v in ("unverified", "verified")}
    mistakes = {v: Counter() for v in by}
    removed, per_case, log = Counter(), {v: [] for v in by}, []
    pages = []
    start = time.time()
    for ci, c in enumerate(cases, 1):
        env = CaseEnv(c)
        if args.mode == "agent":
            agent = LLMAgent(client, budget=args.budget, max_steps=args.steps, citations=args.citations, nudges=args.nudges)
        else:
            agent = GuidedReader(client, budget=args.budget, citations=args.citations)
        found = {v: 0 for v in by}
        gold = 0
        for q in c["questions"]:
            t0 = time.time()
            claims = agent.answer(env, q)
            checked, rem, _ = verify(env, claims)
            removed += rem
            for v, cl in (("unverified", claims), ("verified", checked)):
                s = score_question(q["id"], cl, c)
                by[v][q["id"]].update(gold=s["gold"], found=s["found"], claims=s["claims"],
                                      empty_right=int(s["empty_and_right"]), empty_gold=int(s["gold"] == 0))
                mistakes[v] += s["mistakes"]
                found[v] += s["found"]
            gold += len(c["gold"][q["id"]])
            e = agent.log[-1]
            calls = f"{e['steps']} steps" if "steps" in e else f"{e['calls']} calls"
            print(f"case {ci}/{len(cases)} {q['id']:14} {len(claims)} claims, {len(checked)} after checks, {e['pages']} pages, "
                  f"{calls}, {time.time() - t0:.0f}s", flush=True)
        for v in by:
            per_case[v].append({"found": found[v], "gold": gold, "pages": len(env.opened)})
        pages.append(len(env.opened))
        log += [{**e, "case": c["id"]} for e in agent.log]
    res = {"settings": {k: str(v) for k, v in vars(args).items()}, "minutes": round((time.time() - start) / 60, 1)}
    for v in by:
        r = {"by_question": {q: dict(x) for q, x in by[v].items()}, "per_case": per_case[v], "pages_per_case": sum(pages) / len(pages)}
        res[v] = {**totals(r), "by_question": r["by_question"], "mistakes": dict(mistakes[v])}
        print(f"{v:10}", totals(r), dict(mistakes[v]))
    res["removed_citations"] = dict(removed)
    res["log"] = log
    tag = f"{args.mode}_{args.citations}" + (f"_nudges{args.nudges}" if args.mode == "agent" else "")
    for v in by:
        print(f"{v:10} by question:", {q: f"{x['found']}/{x['gold']}" for q, x in res[v]["by_question"].items()})
    print("citations removed by the checks:", dict(removed))
    print("bad replies:", sum(e.get("bad_replies", 0) for e in log), "| nudges:", sum(e.get("nudges", 0) for e in log),
          "| minutes:", res["minutes"])
    path = out / f"llm_{args.model.replace(':', '_').replace('/', '_')}_{args.size}_{args.cases}cases_{tag}.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=["rules", "ollama"], default="rules")
    p.add_argument("--cases", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", choices=["small", "large"], default="large", help="for --agent ollama")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--budget", type=int, default=8, help="pages per question, for --agent ollama")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window; lower it if you run out of memory")
    p.add_argument("--mode", choices=["agent", "guided"], default="agent", help="agent: the model explores; guided: the workflow picks pages")
    p.add_argument("--citations", choices=["lines", "quotes"], default="lines", help="cite line numbers, or copy quotes")
    p.add_argument("--nudges", type=int, default=1, help="agent mode: times the code may point out unopened pages")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (rules if args.agent == "rules" else llm)(args, args.out)


if __name__ == "__main__":
    main()
