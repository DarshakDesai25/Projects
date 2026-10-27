"""Run every experiment and write tables and figures to results/.

  python run.py                 # 10 simulated runs, about a minute
  python run.py --seeds 30      # tighter error bars
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate as E
import simulate as S

BUDGETS = [0.25, 1.0]
STEALTH = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
SCALED = ["BURST", "EXFIL", "SLOW_EXFIL", "CRED_EXTERNAL"]
COLORS = {"Learned baseline": "C1", "Fixed rules, tuned": "C0", "Fixed rules, default": "gray"}


def methods(o, budget):
    """The result of each method for one run, with settings chosen on the calibration days."""
    return {"Learned baseline": o["baseline"][E.pick_threshold(o["val_fp"], budget)],
            "Fixed rules, tuned": o["rules"][E.pick_factor(o["rules_val_fp"], budget)],
            "Fixed rules, default": o["rules"][1.0]}


def pooled(outs, name, budget, kinds=None):
    caught = total = 0
    for o in outs:
        c, n = E.recall(methods(o, budget)[name], o["incidents"], kinds)
        caught, total = caught + c, total + n
    return caught, total


def fmt_recall(c, n):
    lo, hi = E.wilson(c, n)
    return f"{c / n:.0%} ({lo:.0%} to {hi:.0%})"


def summary_table(outs, budget):
    lines = [f"Settings chosen so that false alarms on the calibration days stay at or below {budget} per agent per day.\n",
             "| Method | " + " | ".join(S.TYPES) + " | All incidents | False alarms per agent per day | Share from benign look-alikes | Median delay |",
             "| --- |" + " --- |" * (len(S.TYPES) + 4)]
    for name in COLORS:
        row = [fmt_recall(*pooled(outs, name, budget, [k])) for k in S.TYPES]
        row.append(fmt_recall(*pooled(outs, name, budget)))
        res = [methods(o, budget)[name] for o in outs]
        fp = np.array([r["fp_per_agent_day"] for r in res])
        tot, look = sum(r["fp"] for r in res), sum(r["fp_lookalike"] for r in res)
        delays = [d for r in res for d in r["delay"].values()]
        row += [f"{fp.mean():.2f} (sd {fp.std():.2f})", f"{look / tot:.0%}" if tot else "n/a",
                f"{np.median(delays) / 60:.1f} min" if delays else "n/a"]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def fig_tradeoff(outs, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    thr = sorted(outs[0]["baseline"])
    x = [np.mean([o["baseline"][t]["fp_per_agent_day"] for o in outs]) for t in thr]
    y = [sum(E.recall(o["baseline"][t], o["incidents"])[0] for o in outs) / sum(len(o["incidents"]) for o in outs) for t in thr]
    ax.plot(np.maximum(x, 0.01), y, "-", color="C1", label="Learned baseline, every threshold")
    fs = sorted(outs[0]["rules"])
    x = [np.mean([o["rules"][f]["fp_per_agent_day"] for o in outs]) for f in fs]
    y = [sum(E.recall(o["rules"][f], o["incidents"])[0] for o in outs) / sum(len(o["incidents"]) for o in outs) for f in fs]
    ax.plot(np.maximum(x, 0.01), y, "o-", color="C0", label="Fixed rules, every strictness")
    for f, xx, yy in zip(fs, x, y):
        if f in (0.1, 0.2, 0.35):
            ax.annotate(f"strictness x{f}", (max(xx, 0.01), yy), fontsize=6.5, xytext=(3, -9), textcoords="offset points")
    for name, mk in [("Learned baseline", "s"), ("Fixed rules, tuned", "D"), ("Fixed rules, default", "^")]:
        for i, b in enumerate(BUDGETS):
            xs = np.mean([methods(o, b)[name]["fp_per_agent_day"] for o in outs])
            c, n = pooled(outs, name, b)
            ax.plot(max(xs, 0.01), c / n, mk, color=COLORS[name], ms=8, mec="k",
                    label=f"{name}, settings picked on calibration days" if i == 0 and name != "Fixed rules, default" else
                    ("Fixed rules, default settings" if name == "Fixed rules, default" and i == 0 else None))
    ax.set_xscale("log")
    ax.set_xlabel("false alarms per agent per day on the test week")
    ax.set_ylabel("share of incidents caught")
    ax.set_ylim(0, 1.03)
    ax.set_title("Catching more means more false alarms\nlarge markers are settings picked on the calibration days", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_by_type(outs, path, budget=1.0):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    kinds = S.TYPES
    for j, name in enumerate(COLORS):
        vals, err = [], []
        for k in kinds:
            c, n = pooled(outs, name, budget, [k])
            lo, hi = E.wilson(c, n)
            vals.append(c / n)
            err.append([max(0.0, c / n - lo), max(0.0, hi - c / n)])
        ax.bar(np.arange(len(kinds)) + (j - 1) * 0.27, vals, 0.27, color=COLORS[name], label=name, yerr=np.array(err).T, error_kw={"lw": 1})
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of incidents caught")
    ax.set_title(f"Which incidents each method catches (false alarm budget {budget} per agent per day)", fontsize=9)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def fig_stealth(sweep, path, budget=1.0):
    fig, axes = plt.subplots(1, len(SCALED), figsize=(13, 3.6), sharey=True)
    for ax, kind in zip(axes, SCALED):
        for name in COLORS:
            y, lo, hi = [], [], []
            for s in STEALTH:
                c, n = pooled(sweep[s], name, budget, [kind])
                a, b = E.wilson(c, n)
                y.append(c / n); lo.append(a); hi.append(b)
            ax.plot(STEALTH, y, "o-", color=COLORS[name], label=name)
            ax.fill_between(STEALTH, lo, hi, color=COLORS[name], alpha=0.12)
        ax.set_title(kind, fontsize=9)
        ax.set_xlabel("attack strength (1.0 is the standard attack)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("share of incidents caught")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"What happens when attackers are quieter (false alarm budget {budget} per agent per day)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_timeline(extras, o, path, budget=1.0):
    log, scores, _ = extras
    inc = next(i for i in o["incidents"] if i["type"] == "SLOW_EXFIL" and S.AGENTS[i["agent"]] != "report-bot")
    thr = E.pick_threshold(o["val_fp"], budget)
    lo, hi = inc["start"] - 2 * 3600, inc["end"] + 2 * 3600
    m = (log.agent == inc["agent"]) & (log.t >= lo) & (log.t <= hi)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.scatter((log.t[m] - inc["start"]) / 3600, scores[m], s=5, color="C1")
    ax.axvspan(0, (inc["end"] - inc["start"]) / 3600, color="red", alpha=0.1, label="slow data theft (incident)")
    ax.axhline(thr, color="k", ls="--", lw=1, label=f"alert threshold {thr}")
    ax.set_xlabel("hours since the incident started")
    ax.set_ylabel("surprise score of each request")
    ax.set_title(f"{S.AGENTS[inc['agent']]}: slow data theft is a steady trickle with no burst, but every response is far larger than normal", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sweep = {s: [] for s in STEALTH}
    extras0 = None
    for seed in range(a.seeds):
        normal = S.make_normal(seed)
        for s in STEALTH:
            o, ex = E.run_seed(seed, s, normal)
            sweep[s].append(o)
            if seed == 0 and s == 1.0:
                extras0 = ex
        print(f"seed {seed + 1}/{a.seeds} done", flush=True)
    outs = sweep[1.0]

    text = "\n\n".join(summary_table(outs, b) for b in BUDGETS)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)

    lines = ["| Attack strength | " + " | ".join(f"{k}: baseline" for k in SCALED[:3]) + " | " + " | ".join(f"{k}: rules" for k in SCALED[:3]) + " |",
             "| --- |" + " --- |" * 6]
    for s in STEALTH:
        cells = [fmt_recall(*pooled(sweep[s], "Learned baseline", 1.0, [k])) for k in SCALED[:3]]
        cells += [fmt_recall(*pooled(sweep[s], "Fixed rules, default", 1.0, [k])) for k in SCALED[:3]]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "stealth.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig_tradeoff(outs, os.path.join(a.out, "tradeoff.png"))
    fig_by_type(outs, os.path.join(a.out, "by_incident_type.png"))
    fig_stealth(sweep, os.path.join(a.out, "stealth.png"))
    fig_timeline(extras0, outs[0], os.path.join(a.out, "timeline.png"))

    raw = {"seeds": a.seeds, "incidents_per_run": len(outs[0]["incidents"]),
           "summary": {str(b): {name: {"caught": pooled(outs, name, b)[0], "total": pooled(outs, name, b)[1]} for name in COLORS} for b in BUDGETS}}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(raw, f, indent=2)
    S.to_jsonl(extras0[0], os.path.join(a.out, "sample_logs.jsonl"), S.VAL_END + 1 * S.DAY, S.VAL_END + 1 * S.DAY + 3600)
    print("\nsaved tables and figures to", a.out)


if __name__ == "__main__":
    main()
