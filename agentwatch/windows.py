"""Sliding window helpers. Events must be sorted by time."""
import numpy as np


def window_counts(t, width):
    """For each event, how many events (itself included) fall in (t - width, t]."""
    left = np.searchsorted(t, t - width, side="right")
    return np.arange(len(t)) + 1 - left


def window_sums(t, x, width):
    """For each event, the sum of x over the events that fall in (t - width, t]."""
    left = np.searchsorted(t, t - width, side="right")
    cs = np.concatenate(([0.0], np.cumsum(x)))
    return cs[np.arange(len(t)) + 1] - cs[left]
