import numpy as np

import experiments as E
import methods as M
import sim

LAB, POP, PERSONAL = M.NAMES


def test_simulation_is_reproducible():
    a = sim.people(50, np.random.default_rng(1), "level")
    b = sim.people(50, np.random.default_rng(1), "level")
    assert np.array_equal(a["daily"], b["daily"]) and np.array_equal(a["tau"], b["tau"])


def test_people_differ_and_wander_by_the_amounts_asked_for():
    p = sim.people(3000, np.random.default_rng(0), sigma_b=0.4, sigma_w=0.2, sigma_m=0.25, k=4)
    assert abs(p["daily"].mean(axis=1).std() - 0.4) < 0.03
    assert abs(p["daily"][:, 28:].std(axis=1).mean() - np.sqrt(0.2 ** 2 + 0.25 ** 2 / 4)) < 0.02


def test_level_shift_has_the_requested_size():
    p = sim.people(1500, np.random.default_rng(2), "level", delta=0.5)
    tau = p["tau"]
    pre = np.array([p["daily"][i, 30:tau[i] - 1].mean() for i in range(300)])
    post = np.array([p["daily"][i, tau[i] + 20:tau[i] + 60].mean() for i in range(300)])
    assert abs((post - pre).mean() - 0.5) < 0.05


def test_volatility_change_makes_the_signal_jumpier():
    p = sim.people(1500, np.random.default_rng(3), "volatility", vol=3.0)
    tau = p["tau"]
    pre = np.array([np.diff(p["daily"][i, 30:tau[i] - 1]).std() for i in range(300)])
    post = np.array([np.diff(p["daily"][i, tau[i] + 5:tau[i] + 60]).std() for i in range(300)])
    assert (post / pre).mean() > 1.8


def test_people_who_never_change_have_no_change_day():
    assert (sim.people(20, np.random.default_rng(0), None)["tau"] == sim.NO_CHANGE).all()


def test_lab_days_stay_inside_follow_up_for_every_baseline_length():
    rng = np.random.default_rng(0)
    for base in (7, 14, 28, 56):
        d = sim.lab_days(200, rng, base)
        assert d.min() >= base and d.max() < sim.DAYS


def test_days_before_the_baseline_never_alarm():
    s = M.valid(np.ones((2, sim.DAYS)))
    assert (s[:, :sim.BASE] == -np.inf).all() and (s[:, sim.BASE:] == 1).all()


def test_rolling_mean_matches_a_hand_calculation():
    x = np.arange(10, dtype=float)[None, :]
    r = M.rolling(x, 3, np.mean)
    assert np.isnan(r[0, :2]).all() and r[0, 2] == 1.0 and r[0, 9] == 8.0


def test_every_method_flags_about_the_chosen_share_of_unchanged_people():
    r = E.run("level", n=2500, seed=4)
    for name in M.NAMES:
        assert 0.02 < r[name]["false_alarm"] < 0.09, (name, r[name]["false_alarm"])


def test_personal_baseline_wins_when_people_differ_a_lot():
    r = E.run("level", n=2500, seed=5, sigma_b=0.6)
    assert r[PERSONAL]["caught"] > r[POP]["caught"] + 0.3 > r[LAB]["caught"] + 0.3


def test_population_range_wins_when_people_are_nearly_identical():
    r = E.run("level", n=2500, seed=6, sigma_b=0.05)
    assert r[POP]["caught"] > r[PERSONAL]["caught"]


def test_bigger_shifts_are_easier_to_catch_and_longer_baselines_help():
    small = E.run("level", n=2000, seed=7, delta=0.2)[PERSONAL]["caught"]
    large = E.run("level", n=2000, seed=7, delta=1.0)[PERSONAL]["caught"]
    assert large > small + 0.5
    short = E.run("level", n=2000, seed=8, base=7)[PERSONAL]["caught"]
    long = E.run("level", n=2000, seed=8, base=56)[PERSONAL]["caught"]
    assert long > short + 0.3


def test_wilson_interval_brackets_the_estimate():
    lo, hi = E.wilson(80, 100)
    assert lo < 0.8 < hi and 0 <= lo and hi <= 1
