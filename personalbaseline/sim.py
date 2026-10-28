"""A made up signal that moves around, to ask when a personal baseline beats a population range.

Nothing here is real physiology. The signal stands for any measurement that varies a lot from
person to person and from day to day, such as a hormone level. It is on a log scale, so adding
0.3 means a 35 percent increase. All numbers are invented.

Each person has
  a personal level          people differ from each other by sigma_b
  day to day swings         the signal wanders around the personal level by sigma_w, and today
                            resembles yesterday (correlation phi)
  measurement noise         every reading is off by sigma_m, and a wearable that reads k times a
                            day averages the noise down

A change can then happen on a day tau, in one of two ways.
  level      the personal level rises by delta over 14 days
  volatility the day to day swings get vol times bigger
"""
import numpy as np

DAYS = 210            # length of follow up
BASE = 28             # days used to learn a person's own baseline
CHANGE_RANGE = (60, 120)
WINDOW = 60           # a change counts as caught if an alarm comes within this many days
NO_CHANGE = 10 ** 6

DEFAULTS = dict(sigma_b=0.4, sigma_w=0.2, sigma_m=0.25, k=4, phi=0.6, delta=0.5, vol=2.0)


def people(n, rng, change=None, **kw):
    """Simulate n people for DAYS days. change is None, "level" or "volatility".

    Returns a dict with
      daily   (n, DAYS) what a wearable reports each day, the average of k noisy readings
      single  (n, DAYS) what one lab reading would show on that day
      tau     (n,)      the day the change starts, or a huge number if there is none
    """
    p = {**DEFAULTS, **kw}
    tau = rng.integers(CHANGE_RANGE[0], CHANGE_RANGE[1] + 1, n) if change else np.full(n, NO_CHANGE)
    t = np.arange(DAYS)[None, :]
    swing = np.where((t >= tau[:, None]) & (change == "volatility"), p["sigma_w"] * p["vol"], p["sigma_w"])

    a = np.empty((n, DAYS))
    a[:, 0] = swing[:, 0] * rng.standard_normal(n)
    for d in range(1, DAYS):
        a[:, d] = p["phi"] * a[:, d - 1] + np.sqrt(1 - p["phi"] ** 2) * swing[:, d] * rng.standard_normal(n)

    level = p["sigma_b"] * rng.standard_normal(n)[:, None] + a
    if change == "level":
        level = level + p["delta"] * np.clip((t - tau[:, None]) / 14.0, 0, 1)
    return dict(daily=level + p["sigma_m"] / np.sqrt(p["k"]) * rng.standard_normal((n, DAYS)),
                single=level + p["sigma_m"] * rng.standard_normal((n, DAYS)), tau=tau, params=p)


def lab_days(n, rng, base=BASE):
    """Up to six lab tests for each person, one every 30 days, starting on a day of their own."""
    count = min(6, (DAYS - 1 - base) // 30 + 1)
    spread = min(29, DAYS - 1 - base - 30 * (count - 1))
    return base + rng.integers(0, spread + 1, n)[:, None] + 30 * np.arange(count)[None, :]
