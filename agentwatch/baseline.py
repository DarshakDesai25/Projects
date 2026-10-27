"""A monitor that learns what normal looks like for each agent.

For every agent it learns, from clean history, five things.
  rate        how many requests it usually makes in a minute while it is working,
              including how much that count varies from day to day
  volume      how many bytes it usually moves in a five minute stretch
  hours       which hours of the day it is usually active
  endpoints   which endpoint families it usually calls
  order       which endpoint it usually calls right after which other one

Each new request then gets five surprise scores, one per thing above. A surprise score
is the negative base 10 log of how likely this was under the agent's own history, so 3
means "about 1 in 1,000". The request is scored by its biggest surprise. This is a
simple way to combine the five, not a calibrated probability.

An alert fires when the score passes a threshold. The threshold is the knob that trades
missed incidents against false alarms.
"""
import numpy as np
from scipy.stats import nbinom, norm, poisson

import simulate as S
from windows import window_counts, window_sums

CLIP = 10.0


def surprise(p):
    return np.clip(-np.log10(np.maximum(p, 1e-12)), 0, CLIP)


class Baseline:
    def fit(self, log, train_end):
        k = len(S.FAMILIES)
        self.models = {}
        for a in range(len(S.AGENTS)):
            idx = np.where((log.agent == a) & (log.t < train_end))[0]
            t, fam, b = log.t[idx], log.fam[idx], log.nbytes[idx]
            hours = (t % S.DAY) // 3600
            hour_counts = np.bincount(hours, minlength=24)
            trans = np.zeros((k, k))
            np.add.at(trans, (fam[:-1], fam[1:]), 1)
            counts = window_counts(t, 60) - 1                      # other requests in the last minute
            mean, var = counts.mean(), counts.var()
            logv = np.log10(window_sums(t, b, 300) + 1)
            self.models[a] = dict(
                mean=mean, var=var,
                logv_mean=logv.mean(), logv_std=max(logv.std(), 1e-3),
                hour_counts=hour_counts,
                fam_counts=np.bincount(fam, minlength=k).astype(float),
                n=len(t), trans=trans, k=k)
        return self

    def components(self, log, a):
        """The five surprise scores for every event of agent a, and their indices."""
        m = self.models[a]
        idx = np.where(log.agent == a)[0]
        t, fam, b = log.t[idx], log.fam[idx], log.nbytes[idx]
        others = window_counts(t, 60) - 1
        if m["var"] > m["mean"] * 1.05:      # counts vary more than pure chance: negative binomial
            r = m["mean"] ** 2 / (m["var"] - m["mean"])
            rate = surprise(nbinom.sf(others - 1, r, r / (r + m["mean"])))
        else:
            rate = surprise(poisson.sf(others - 1, max(m["mean"], 1e-9)))
        z = (np.log10(window_sums(t, b, 300) + 1) - m["logv_mean"]) / m["logv_std"]
        volume = surprise(norm.sf(z))
        hour = surprise((m["hour_counts"][(t % S.DAY) // 3600] + 0.5) / (m["n"] + 12))
        endpoint = surprise((m["fam_counts"][fam] + 0.5) / (m["n"] + 0.5 * m["k"]))
        prev = np.concatenate(([-1], fam[:-1]))
        row = np.where(prev >= 0, m["trans"][np.maximum(prev, 0)].sum(axis=1), 0.0)
        cnt = np.where(prev >= 0, m["trans"][np.maximum(prev, 0), fam], 0.0)
        order = np.where(prev >= 0, surprise((cnt + 0.5) / (row + 0.5 * m["k"])), 0.0)
        return idx, dict(rate=rate, volume=volume, hours=hour, endpoints=endpoint, order=order)

    def score(self, log):
        """One score per event, aligned with the log."""
        out = np.zeros(len(log))
        for a in self.models:
            idx, comp = self.components(log, a)
            out[idx] = np.max(np.vstack(list(comp.values())), axis=0)
        return out
