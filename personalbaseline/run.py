"""Run every experiment and write tables and figures to results/.

  python run.py            # about a minute
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as E
import methods as M
import sim

COLORS = dict(zip(M.NAMES, ["gray", "C0", "C1"]))
SWEEPS = [
    ("Differences between people (sigma_b)", "sigma_b", [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8], "level"),
    ("Day to day swings within a person (sigma_w)", "sigma_w", [0.05, 0.1, 0.2, 0.3, 0.4], "level"),
    ("Readings per day from the wearable (k)", "k", [1, 2, 4, 8, 24], "level"),
    ("Size of the shift in level (delta)", "delta", [0.1, 0.2, 0.3, 0.5, 0.7, 1.0], "level"),
    ("Days used to learn a personal baseline", "base", [7, 14, 28, 56], "level"),
    ("How much the day to day swings grow (vol)", "vol", [1.5, 2, 3, 4], "volatility"),
]
DEFAULT_OF = dict(sim.DEFAULTS, base=sim.BASE)


def headline(path):
    lines = []
    for change, title in [("level", "A shift in the level, +0.5 on the log scale (about 65 percent higher)"),
                          ("volatility", "The day to day swings double")]:
        r = E.run(change)
        lines += [f"{title}", "", "| Method | Changes caught within 60 days | Median days to catch | Never changed, flagged by mistake |",
                  "| --- | --- | --- | --- |"]
        for name, v in r.items():
            lines.append(f"| {name} | {v['caught']:.0%} ({v['lo']:.0%} to {v['hi']:.0%}) | {v['delay']:.0f} | {v['false_alarm']:.1%} |")
        lines.append("")
    open(path, "w").write("\n".join(lines))
    return "\n".join(lines)


def sweeps(figpath, jsonpath, tablepath):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    table, store = [], {}
    for ax, (title, key, values, change) in zip(axes.ravel(), SWEEPS):
        res = [E.run(change, **{key: v}) for v in values]
        store[key] = dict(values=values, change=change, results=res)
        for name in M.NAMES:
            y = np.array([r[name]["caught"] for r in res])
            lo = np.array([r[name]["lo"] for r in res])
            hi = np.array([r[name]["hi"] for r in res])
            ax.plot(values, y, "o-", color=COLORS[name], label=name)
            ax.fill_between(values, lo, hi, color=COLORS[name], alpha=0.15)
        ax.axvline(DEFAULT_OF[key], color="k", ls=":", lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.3)
        if key in ("k",):
            ax.set_xscale("log")
            ax.set_xticks(values)
            ax.set_xticklabels(values)
        table += [f"{title}", "", "| Value | " + " | ".join(M.NAMES) + " |", "| --- |" + " --- |" * 3]
        for v, r in zip(values, res):
            table.append(f"| {v} | " + " | ".join(f"{r[n]['caught']:.0%}" for n in M.NAMES) + " |")
        table.append("")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of changes caught")
    axes[0, 0].legend(fontsize=7, loc="center left")
    fig.suptitle("Share of real changes caught, with every method flagging about 5 percent of people who never change\n(dotted line is the setting used in the headline table)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    open(tablepath, "w").write("\n".join(table))
    json.dump(store, open(jsonpath, "w"), indent=1)


def example(path):
    rng = np.random.default_rng(5)
    ref, thr, _ = E.calibrate("level", 4000, rng)
    pop = sim.people(2000, rng, "level")
    days = sim.lab_days(2000, rng)
    scores = [M.valid(s) for s in M.level_scores(pop, ref, days)]
    t = np.arange(sim.DAYS)[None, :]
    tau = pop["tau"][:, None]
    hit = [(s > th) & (t >= tau) & (t <= tau + sim.WINDOW) for s, th in zip(scores, thr)]
    i = int(np.where(hit[2].any(axis=1) & ~hit[0].any(axis=1) & ~hit[1].any(axis=1))[0][0])
    m7 = M.rolling(pop["daily"], 7, np.mean)[i]
    b, s = M.own_baseline(pop["daily"])
    alarm = int(hit[2][i].argmax())
    x = np.arange(sim.DAYS)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axhspan(ref.mu7 - thr[1] * ref.sd7, ref.mu7 + thr[1] * ref.sd7, color="C0", alpha=0.12, label="what a healthy population looks like")
    ax.axhspan(b[i, 0] - thr[2] * s[i, 0], b[i, 0] + thr[2] * s[i, 0], color="C1", alpha=0.2, label="this person's own normal")
    ax.plot(x, pop["daily"][i], color="gray", lw=0.7, alpha=0.6, label="daily wearable reading")
    ax.plot(x, m7, color="k", lw=1.6, label="7 day average")
    ax.plot(days[i], pop["single"][i, days[i]], "D", color="C3", ms=7, label="monthly lab test")
    ax.axvline(pop["tau"][i], color="C3", ls="--", lw=1)
    ax.text(pop["tau"][i] + 1, ax.get_ylim()[1] * 0.95, "real change begins", color="C3", fontsize=8, va="top")
    ax.axvline(alarm, color="C1", lw=1.5)
    ax.text(alarm + 1, ax.get_ylim()[0] * 0.95, "personal baseline raises the alarm", color="C1", fontsize=8, va="bottom")
    ax.set_xlabel("day")
    ax.set_ylabel("signal (log scale, invented units)")
    ax.set_title("One person whose change is invisible against the population but clear against their own history", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    os.makedirs("results", exist_ok=True)
    print(headline("results/summary.md"))
    sweeps("results/sweeps.png", "results/sweeps.json", "results/sweeps.md")
    example("results/example_person.png")
    print("saved to results/")


if __name__ == "__main__":
    main()
