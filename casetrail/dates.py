"""Find dates written in the formats that show up in case files and turn them into YYYY-MM-DD."""
import re
from datetime import date

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
_M = {m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)}
_NAMES = "|".join(MONTHS)
_ABBR = "|".join(m[:3] for m in MONTHS)

_PATTERNS = [
    (re.compile(rf"\b({_NAMES})\s+(\d{{1,2}}),\s*(\d{{4}})\b"), lambda g: (int(g[2]), _M[g[0][:3].lower()], int(g[1]))),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda g: (int(g[0]), int(g[1]), int(g[2]))),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), lambda g: (int(g[2]), int(g[0]), int(g[1]))),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2})(?!\d)"), lambda g: (2000 + int(g[2]), int(g[0]), int(g[1]))),
    (re.compile(rf"\b(\d{{1,2}})\s+({_ABBR})[a-z]*\.?\s+(\d{{4}})\b"), lambda g: (int(g[2]), _M[g[1].lower()], int(g[0]))),
]


def find_dates(text):
    """Every date in the text as (start, end, iso, matched_text), in order, without overlaps."""
    found = []
    for pat, build in _PATTERNS:
        for m in pat.finditer(text):
            try:
                y, mo, d = build(m.groups())
                iso = date(y, mo, d).isoformat()
            except ValueError:
                continue
            if not any(m.start() < e and s < m.end() for s, e, _, _ in found):
                found.append((m.start(), m.end(), iso, m.group(0)))
    return sorted(found)


def parse_date(text):
    """The first date in the text as YYYY-MM-DD, or None."""
    hits = find_dates(text or "")
    return hits[0][2] if hits else None
