"""Run a scenario: set thresholds on people who never change, then count who each method catches."""
import numpy as np

import methods as M
import sim


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def calibrate(change, n, rng, false_alarm=0.05, base=sim.BASE, **kw):
    """Learn the healthy population and set each method's threshold on people who never change."""
    score = M.level_scores if change == "level" else M.volatility_scores
    calib = sim.people(n, rng, None, **kw)
    ref = M.Reference(calib, base)
    cal = [M.valid(s, base) for s in score(calib, ref, sim.lab_days(n, rng, base), base)]
    return ref, [np.quantile(s.max(axis=1), 1 - false_alarm) for s in cal], score


def run(change, n=4000, seed=0, false_alarm=0.05, base=sim.BASE, **kw):
    """For each method, the share of changes caught within 60 days, its 95 percent range, the median
    delay in days and the share of people who never changed that it flagged on fresh data."""
    rng = np.random.default_rng(seed)
    ref, thr, score = calibrate(change, n, rng, false_alarm, base, **kw)
    fresh = sim.people(n, rng, None, **kw)
    fresh_scores = [M.valid(s, base) for s in score(fresh, ref, sim.lab_days(n, rng, base), base)]
    changed = sim.people(n, rng, change, **kw)
    ch_scores = [M.valid(s, base) for s in score(changed, ref, sim.lab_days(n, rng, base), base)]

    t = np.arange(sim.DAYS)[None, :]
    tau = changed["tau"][:, None]
    out = {}
    for name, s_fresh, s_ch, th in zip(M.NAMES, fresh_scores, ch_scores, thr):
        hit = (s_ch > th) & (t >= tau) & (t <= tau + sim.WINDOW)
        caught = hit.any(axis=1)
        delay = hit.argmax(axis=1)[caught] - changed["tau"][caught]
        lo, hi = wilson(int(caught.sum()), n)
        out[name] = dict(caught=float(caught.mean()), lo=lo, hi=hi,
                         delay=float(np.median(delay)) if caught.any() else float("nan"),
                         false_alarm=float((s_fresh > th).any(axis=1).mean()))
    return out
