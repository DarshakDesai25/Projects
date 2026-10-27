"""Score alerts against the known incidents."""
import numpy as np

import baseline
import rules
import simulate as S

COOLDOWN = 600   # after an alert for an agent, ignore its alerts for 10 minutes
GRACE = 300      # an alert up to 5 minutes after an incident ends still counts
THRESHOLDS = np.round(np.arange(1.0, 10.01, 0.1), 2)
RULE_FACTORS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]


def dedupe(alerts, cooldown=COOLDOWN):
    last, out = {}, []
    for t, a in alerts:
        if a not in last or t >= last[a] + cooldown:
            out.append((int(t), int(a)))
            last[a] = t
    return out


def baseline_alerts(log, scores, thr):
    idx = np.where(scores >= thr)[0]
    return dedupe(list(zip(log.t[idx], log.agent[idx])))


def rule_alerts(log, factor):
    raw = rules.alerts(log, rate_limit=max(3, int(round(20 * factor))), exfil_bytes=50e6 * factor)
    return dedupe([(t, a) for t, a, _ in raw])


def score_alerts(alerts, incidents, benign, t_lo, t_hi):
    """Which incidents were caught, how fast, and how many alerts were false alarms."""
    alerts = [(t, a) for t, a in alerts if t_lo <= t < t_hi]
    delay = {}
    fp = lookalike = 0
    for t, a in alerts:
        hit = False
        for k, inc in enumerate(incidents):
            if inc["agent"] == a and inc["start"] <= t <= inc["end"] + GRACE:
                hit = True
                delay.setdefault(k, t - inc["start"])
        if not hit:
            fp += 1
            if any(b["agent"] == a and b["start"] <= t <= b["end"] + GRACE for b in benign):
                lookalike += 1
    days = (t_hi - t_lo) / S.DAY
    return dict(delay=delay, fp=fp, fp_lookalike=lookalike, fp_per_agent_day=fp / (len(S.AGENTS) * days))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_seed(seed, stealth=1.0, normal=None, thresholds=THRESHOLDS, factors=RULE_FACTORS):
    """One simulated run, every detector, every setting. Returns plain dicts of results."""
    log, incidents, benign = S.build(seed, stealth, normal)
    bl = baseline.Baseline().fit(log, S.TRAIN_END)
    scores = bl.score(log)
    out = dict(seed=seed, stealth=stealth, incidents=incidents, baseline={}, rules={}, val_fp={}, rules_val_fp={})
    for thr in thresholds:
        al = baseline_alerts(log, scores, thr)
        out["baseline"][float(thr)] = score_alerts(al, incidents, benign, S.VAL_END, N_END)
        out["val_fp"][float(thr)] = score_alerts(al, [], [], S.TRAIN_END, S.VAL_END)["fp_per_agent_day"]
    for f in factors:
        al = rule_alerts(log, f)
        out["rules"][f] = score_alerts(al, incidents, benign, S.VAL_END, N_END)
        out["rules_val_fp"][f] = score_alerts(al, [], [], S.TRAIN_END, S.VAL_END)["fp_per_agent_day"]
    return out, (log, scores, benign)


N_END = S.N_DAYS * S.DAY


def pick_threshold(val_fp, budget):
    """Lowest threshold whose false alarms on the calibration days stay within the budget."""
    for thr in sorted(val_fp):
        if val_fp[thr] <= budget:
            return thr
    return max(val_fp)


def recall(result, incidents, kinds=None):
    """(caught, total) for the chosen incident types."""
    ks = [k for k, inc in enumerate(incidents) if kinds is None or inc["type"] in kinds]
    return sum(k in result["delay"] for k in ks), len(ks)


def pick_factor(rules_val_fp, budget):
    """Most sensitive rule setting whose false alarms on the calibration days stay within the budget."""
    for f in sorted(rules_val_fp):
        if rules_val_fp[f] <= budget:
            return f
    return max(rules_val_fp)
