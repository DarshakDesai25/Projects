"""Three ways to notice that something has changed, scored on the same people.

  lab test             one reading a month, compared with the range of a healthy population
  wearable, average    daily readings, compared with the same population range
  wearable, personal   daily readings, compared with the person's own first four weeks

Every method turns the data into an alarm score for each day. A person is flagged when the
score passes a threshold. Thresholds are set on people who never change, so that every method
flags the same share of them by mistake. That makes the methods comparable.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

import sim

NAMES = ["Lab test, monthly", "Wearable, population range", "Wearable, personal baseline"]


def rolling(x, w, fn):
    out = np.full(x.shape, np.nan)
    out[:, w - 1:] = fn(windows(x, w, axis=1), axis=-1)
    return out


class Reference:
    """What a healthy population looks like, learned once from people who do not change."""

    def __init__(self, pop, base=sim.BASE):
        d = pop["daily"][:, base:]
        self.mu = d.mean()
        self.sd_daily = d.std()
        self.sd_single = pop["single"][:, base:].std()
        m7 = rolling(pop["daily"], 7, np.mean)[:, base:]
        self.mu7, self.sd7 = np.nanmean(m7), np.nanstd(m7)


def own_baseline(daily, base=sim.BASE):
    b = daily[:, :base]
    return b.mean(axis=1, keepdims=True), b.std(axis=1, ddof=1, keepdims=True)


def lab_scores(pop, ref, days):
    """Distance from the population mean, only on lab test days."""
    n = len(days)
    out = np.full(pop["single"].shape, -np.inf)
    rows = np.arange(n)[:, None]
    out[rows, days] = np.abs(pop["single"][rows, days] - ref.mu) / ref.sd_single
    return out


def level_scores(pop, ref, days, base=sim.BASE):
    """Alarm score for every day, for each method, when the thing to catch is a shift in level."""
    m7 = rolling(pop["daily"], 7, np.mean)
    b, s = own_baseline(pop["daily"], base)
    return [lab_scores(pop, ref, days), np.abs(m7 - ref.mu7) / ref.sd7, np.abs(m7 - b) / s]


def volatility_scores(pop, ref, days, base=sim.BASE):
    """Alarm score for every day when the thing to catch is a rise in day to day swings."""
    _, s = own_baseline(pop["daily"], base)
    extreme = rolling(np.abs(pop["daily"] - ref.mu) / ref.sd_daily, 14, np.max)
    return [lab_scores(pop, ref, days), extreme, rolling(pop["daily"], 14, np.std) / s]


def valid(scores, base=sim.BASE):
    """Only days after the baseline period count, and early days with no score never alarm."""
    out = np.where(np.isnan(scores), -np.inf, scores)
    out[:, :base] = -np.inf
    return out
