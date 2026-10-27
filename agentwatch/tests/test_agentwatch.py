import numpy as np

import baseline
import evaluate as E
import rules
import simulate as S
from windows import window_counts, window_sums

A = S.AID["invoice-bot"]


def make_log(events):
    """events is a list of (time, family name, status, bytes) for invoice-bot."""
    t, fam, st, b = zip(*events)
    return S.Log(t, [A] * len(t), [S.FAM[f] for f in fam], b, st)


def ev(t, fam="/invoices", status=200, nbytes=1000):
    return (t, fam, status, nbytes)


# windows

def test_window_counts_and_sums():
    t = np.array([0, 10, 59, 60, 61])
    assert list(window_counts(t, 60)) == [1, 2, 3, 3, 4]      # window is (t - 60, t], so the event at 0 drops out at t = 60
    assert list(window_sums(t, np.ones(5), 60)) == [1, 2, 3, 3, 4]


# simulator

def test_simulation_is_deterministic():
    a, _, _ = S.build(3)
    b, _, _ = S.build(3)
    assert np.array_equal(a.t, b.t) and np.array_equal(a.nbytes, b.nbytes)


def test_incidents_only_in_test_period():
    _, incidents, _ = S.build(1)
    assert len(incidents) == sum(p[1] for p in S.INCIDENT_PLAN)
    assert all(i["start"] >= S.VAL_END for i in incidents)


def test_stealth_changes_strength_not_schedule():
    _, weak, _ = S.build(2, 0.25)
    _, strong, _ = S.build(2, 1.5)
    assert [(i["type"], i["agent"], i["start"]) for i in weak] == [(i["type"], i["agent"], i["start"]) for i in strong]


def test_events_sorted():
    log, _, _ = S.build(0)
    assert np.all(np.diff(log.t) >= 0)


# rules

def kinds(log, **kw):
    return [k for _, _, k in rules.alerts(log, **kw)]


def test_rate_spike_needs_more_than_the_limit():
    assert "RATE_SPIKE" not in kinds(make_log([ev(i) for i in range(20)]))
    assert "RATE_SPIKE" in kinds(make_log([ev(i) for i in range(21)]))


def test_rate_window_is_open_on_the_left():
    assert "RATE_SPIKE" not in kinds(make_log([ev(i) for i in range(20)] + [ev(60)]))


def test_sensitive_access_first_time_only():
    log = make_log([ev(0, "/secrets/db"), ev(10, "/secrets/db"), ev(20, "/admin/status")])
    assert kinds(log).count("SENSITIVE_ACCESS") == 2


def test_allowlist():
    log = make_log([ev(0, "/admin/status")])
    assert "SENSITIVE_ACCESS" in kinds(log)
    assert "SENSITIVE_ACCESS" not in kinds(log, allowlist={"invoice-bot": ["/admin"]})


def test_cred_then_external_window():
    assert "CRED_THEN_EXTERNAL" in kinds(make_log([ev(0, "/credentials/aws"), ev(120, "evil:paste")]))
    assert "CRED_THEN_EXTERNAL" not in kinds(make_log([ev(0, "/credentials/aws"), ev(121, "evil:paste")]))
    assert "CRED_THEN_EXTERNAL" not in kinds(make_log([ev(0, "/credentials/aws", status=403), ev(10, "evil:paste")]))


def test_exfil_sum_must_exceed():
    log = make_log([ev(i * 30, nbytes=12e6) for i in range(5)])
    assert "DATA_EXFIL" in kinds(log)
    assert "DATA_EXFIL" not in kinds(make_log([ev(0, nbytes=30e6), ev(400, nbytes=30e6)]))


# baseline

def fitted():
    log, _, _ = S.build(0)
    return log, baseline.Baseline().fit(log, S.TRAIN_END)


def test_baseline_scores_unseen_endpoint_high():
    log, bl = fitted()
    t0 = S.VAL_END + 10 * 3600
    extra = make_log([ev(t0, "/credentials/aws")])
    both = S.Log(np.concatenate([log.t, extra.t]), np.concatenate([log.agent, extra.agent]),
                 np.concatenate([log.fam, extra.fam]), np.concatenate([log.nbytes, extra.nbytes]),
                 np.concatenate([log.status, extra.status]))
    scores = bl.score(both)
    i = np.where((both.t == t0) & (both.fam == S.FAM["/credentials/aws"]))[0][0]
    assert scores[i] > 3.5


def test_baseline_scores_off_hours_high():
    log, bl = fitted()
    t0 = S.VAL_END + 3600 * 3          # 3 AM, when invoice-bot never works
    extra = make_log([ev(t0, "/invoices")])
    both = S.Log(np.concatenate([log.t, extra.t]), np.concatenate([log.agent, extra.agent]),
                 np.concatenate([log.fam, extra.fam]), np.concatenate([log.nbytes, extra.nbytes]),
                 np.concatenate([log.status, extra.status]))
    scores = bl.score(both)
    i = np.where((both.t == t0) & (both.agent == A) & (both.fam == S.FAM["/invoices"]))[0][0]
    assert scores[i] > 3.0


def test_baseline_normal_traffic_mostly_quiet():
    log, bl = fitted()
    s = bl.score(log)
    held_out = (log.t >= S.TRAIN_END) & (log.t < S.VAL_END)
    assert (s[held_out] >= 4).mean() < 0.01


# evaluation

def test_dedupe_quiet_period():
    assert E.dedupe([(0, 1), (100, 1), (700, 1), (50, 2)]) == [(0, 1), (700, 1), (50, 2)]


def test_score_alerts_hits_and_false_alarms():
    inc = [dict(type="BURST", agent=1, start=1000, end=1060)]
    benign = [dict(agent=1, start=5000, end=5100)]
    r = E.score_alerts([(1010, 1), (5050, 1), (9000, 1), (1020, 2)], inc, benign, 0, 86400)
    assert r["delay"] == {0: 10}
    assert r["fp"] == 3 and r["fp_lookalike"] == 1


def test_wilson_interval():
    lo, hi = E.wilson(10, 10)
    assert 0.6 < lo < 1.0 and hi == 1.0
    assert E.wilson(0, 0)[0] != E.wilson(0, 0)[0]   # nan
