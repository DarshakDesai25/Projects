"""Fixed rules, the way a simple monitor is usually first built.

Four rules, each with a hand picked threshold.
  RATE_SPIKE          more than rate_limit requests in 60 seconds
  DATA_EXFIL          more than exfil_bytes downloaded in 300 seconds
  SENSITIVE_ACCESS    first time an agent touches /admin, /secrets or /credentials
                      (agents on the allowlist may use the prefixes listed for them)
  CRED_THEN_EXTERNAL  a successful /credentials call followed by a call to an
                      outside address within 120 seconds

Spike and exfil rules stay quiet for a while after they fire, 60 and 300 seconds.
"""
from collections import deque

import numpy as np

import simulate as S
from windows import window_counts, window_sums


def _cooldown(times, width):
    kept, last = [], None
    for x in times:
        if last is None or x >= last + width:
            kept.append(int(x))
            last = x
    return kept


def alerts(log, rate_limit=20, exfil_bytes=50_000_000, allowlist=None):
    """Return a list of (time, agent, kind) sorted by time."""
    allowlist = S.ALLOWLIST if allowlist is None else allowlist
    out = []
    for a in range(len(S.AGENTS)):
        idx = np.where(log.agent == a)[0]
        t, fam, b, st = log.t[idx], log.fam[idx], log.nbytes[idx], log.status[idx]

        for x in _cooldown(t[window_counts(t, 60) > rate_limit], 60):
            out.append((x, a, "RATE_SPIKE"))
        for x in _cooldown(t[window_sums(t, b, 300) > exfil_bytes], 300):
            out.append((x, a, "DATA_EXFIL"))

        seen = set()
        allowed = allowlist.get(S.AGENTS[a], [])
        for i in np.where(S.IS_SENSITIVE[fam])[0]:
            if fam[i] in seen:
                continue
            seen.add(fam[i])
            if not any(S.FAMILIES[fam[i]].startswith(p) for p in allowed):
                out.append((int(t[i]), a, "SENSITIVE_ACCESS"))

        creds = deque()
        for i in np.where(S.IS_CRED[fam] | S.IS_EXTERNAL[fam])[0]:
            while creds and t[i] - creds[0] > 120:
                creds.popleft()
            if S.IS_EXTERNAL[fam[i]] and creds:
                creds.popleft()
                out.append((int(t[i]), a, "CRED_THEN_EXTERNAL"))
            if S.IS_CRED[fam[i]] and 200 <= st[i] < 300:
                creds.append(t[i])
    return sorted(out)
