"""Made up case files with a known answer key.

Each case has two documents about one invented plaintiff.
  record      a medical record summary with several visits
  deposition  a short transcript where the plaintiff answers questions about the accident

Because the cases are generated from facts, the answer key is exact. Documents also
contain traps, which are statements a careless reader mistakes for facts.
  ruled out     "fracture ruled out" is not a diagnosis
  denied        "denies prior neck problems" is not a prior condition
  wrong person  "family history of lower back pain (mother)" is not the plaintiff's

Messiness is controlled by a level from 0 to 4.
  0  clean, one date style
  1  four date styles
  2  abbreviations, varied wording for traps, hard wrapped lines, five visits
  3  eight visits, page headers and billing clutter, two digit years
  4  the text is scanned badly, so some characters are misread
"""
import random
import textwrap
from datetime import date, timedelta

from dates import MONTHS

FIRST = ["Maria", "James", "Priya", "Daniel", "Aisha", "Robert", "Chen", "Olivia", "Marcus", "Elena"]
LAST = ["Alvarez", "Thompson", "Patel", "Nguyen", "Okafor", "Schmidt", "Rossi", "Kim", "Harris", "Moreau"]
PROVIDERS = ["Dr. Patel", "Dr. Lindqvist", "Dr. Okoro", "Dr. Ramirez", "Dr. Whitfield", "Dr. Sato", "Dr. Brennan"]
BODY_PARTS = ["lower back", "neck", "left shoulder", "right knee", "right wrist"]
DIAGNOSES = {"lower back": ["lumbar strain", "lumbar disc bulge"], "neck": ["cervical sprain", "whiplash"],
             "left shoulder": ["rotator cuff strain", "shoulder contusion"],
             "right knee": ["knee sprain", "meniscus tear"], "right wrist": ["wrist sprain", "scaphoid contusion"]}
RULED_OUT = ["fracture", "spinal cord injury", "concussion", "dislocation"]
DRUGS = ["ibuprofen", "naproxen", "cyclobenzaprine", "tramadol", "gabapentin", "meloxicam"]
DOSES = {"ibuprofen": "600 mg", "naproxen": "500 mg", "cyclobenzaprine": "10 mg", "tramadol": "50 mg",
         "gabapentin": "300 mg", "meloxicam": "15 mg"}

LEVELS = {
    0: dict(styles=[0], abbr=False, varied=False, wrap=None, visits=3, clutter=False, ocr=0.0),
    1: dict(styles=[0, 1, 2, 3], abbr=False, varied=False, wrap=None, visits=3, clutter=False, ocr=0.0),
    2: dict(styles=[0, 1, 2, 3], abbr=True, varied=True, wrap=70, visits=5, clutter=False, ocr=0.0),
    3: dict(styles=[0, 1, 2, 3, 4], abbr=True, varied=True, wrap=70, visits=8, clutter=True, ocr=0.0),
    4: dict(styles=[0, 1, 2, 3, 4], abbr=True, varied=True, wrap=70, visits=8, clutter=True, ocr=0.01),
}

OCR_MAP = {"0": "O", "1": "l", "5": "S", "l": "1", "o": "0", "e": "c"}


def fmt_date(d, style):
    if style == 0:
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == 1:
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == 2:
        return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"
    if style == 3:
        return d.isoformat()
    return f"{d.month:02d}/{d.day:02d}/{d.year % 100:02d}"


def _wrap(text, width):
    if not width:
        return text
    return "\n".join(textwrap.fill(line, width) if line else "" for line in text.split("\n"))


def _ocr(rng, text, rate):
    if rate <= 0:
        return text
    return "".join(OCR_MAP[c] if c in OCR_MAP and rng.random() < rate else c for c in text)


def make_case(seed, level, cid=0):
    """One case. Returns a dict with the texts, the answer key and the traps."""
    cfg = LEVELS[level]
    rng = random.Random(f"{seed}-{level}-{cid}")
    D = lambda d: fmt_date(d, rng.choice(cfg["styles"]))
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    accident = date(2023, 1, 1) + timedelta(days=rng.randint(0, 600))
    parts = rng.sample(BODY_PARTS, rng.choice([1, 2]))
    others = [b for b in BODY_PARTS if b not in parts]

    prior = [p for p in parts if rng.random() < 0.4]                       # real earlier problems, in the record
    denied_bps = rng.sample([b for b in others], 1)                          # record says the patient denies these
    family_bp = rng.choice([b for b in others if b not in denied_bps])       # a relative had this, not the patient
    ruled_out = rng.choice(RULED_OUT)

    truth, traps = set(), {}
    visits, d = [], accident + timedelta(days=rng.randint(1, 3))
    for i in range(cfg["visits"]):
        v = dict(date=d, provider=rng.choice(PROVIDERS), dx=[], meds=[])
        for bp in parts:
            v["dx"].append(rng.choice(DIAGNOSES[bp]) if i == 0 else DIAGNOSES[bp][0] if rng.random() < 0.7 else rng.choice(DIAGNOSES[bp]))
        v["dx"] = list(dict.fromkeys(v["dx"]))
        if rng.random() < 0.6:
            v["meds"] = rng.sample(DRUGS, rng.choice([1, 2]))
        visits.append(v)
        d = d + timedelta(days=rng.randint(7, 21))

    # medical record
    lines = ["MEDICAL RECORD SUMMARY", f"Patient: {name}", ""]
    truth.add(("ACCIDENT_DATE", accident.isoformat()))
    for i, v in enumerate(visits):
        vd = v["date"].isoformat()
        truth.add(("VISIT", vd, v["provider"].lower()))
        lines.append(f"{'DOS' if cfg['abbr'] else 'Date of service'}: {D(v['date'])}")
        lines.append(f"Provider: {v['provider']}")
        if i == 0:
            when = D(accident)
            lines.append((f"Pt c/o {parts[0]} pain s/p MVA on {when}." if cfg["abbr"] else
                          f"Patient reports {parts[0]} pain after a motor vehicle accident on {when}."))
            hist = []
            for bp in prior:
                truth.add(("PRIOR_CONDITION", bp))
                yr = rng.randint(2015, 2021)
                hist.append(rng.choice([f"History of {bp} pain since {yr}.", f"Prior {bp} injury in {yr}.",
                                        f"Hx of {bp} pain ({yr})."] if cfg["varied"] else [f"History of {bp} pain since {yr}."]))
            for bp in denied_bps:
                traps[("PRIOR_CONDITION", bp)] = "NEGATION_ERROR"
                hist.append(rng.choice([f"Patient denies prior {bp} problems.", f"No history of {bp} injury.",
                                        f"Patient states no previous {bp} complaints."] if cfg["varied"] else [f"Patient denies prior {bp} problems."]))
            traps[("PRIOR_CONDITION", family_bp)] = "WRONG_PERSON"
            hist.append(rng.choice([f"Family history of {family_bp} pain (mother).", f"Her mother also had a prior {family_bp} injury.",
                                    f"Patient's sister had a prior {family_bp} injury."] if cfg["varied"] else [f"Family history of {family_bp} pain (mother)."]))
            rng.shuffle(hist)
            lines.append("History: " + " ".join(hist))
        else:
            lines.append(f"{'F/U' if cfg['abbr'] else 'Follow up for'} {parts[0]} pain.")
        items = list(v["dx"])
        for dx in v["dx"]:
            truth.add(("DIAGNOSIS", vd, dx))
        neg = rng.choice([f"{ruled_out} ruled out", f"no evidence of {ruled_out}", f"negative for {ruled_out}",
                          f"{ruled_out} was excluded"] if cfg["varied"] else [f"{ruled_out} ruled out"])
        items.insert(rng.randint(0, len(items)), neg)
        traps[("DIAGNOSIS", vd, ruled_out)] = "NEGATION_ERROR"
        lines.append(f"{'Dx' if cfg['abbr'] else 'Assessment'}: " + "; ".join(items) + ".")
        plan = []
        for j, drug in enumerate(v["meds"]):
            truth.add(("MEDICATION", vd, drug))
            verb = "Rx:" if cfg["abbr"] else ("Start" if j == 0 else "Prescribed")
            plan.append(f"{verb} {drug} {DOSES[drug]}.")
        plan.append("Follow up in 2 weeks.")
        lines.append("Plan: " + " ".join(plan))
        if cfg["clutter"]:
            lines.append(f"Billing code 99213. Insurance claim {rng.randint(10000, 99999)}. Page {i + 1} of {len(visits)}")
        lines.append("")
    record = _wrap("\n".join(lines), cfg["wrap"])

    # deposition
    depo_date = max(v["date"] for v in visits) + timedelta(days=rng.randint(30, 120))
    mismatch = rng.random() < 0.15
    said = accident + timedelta(days=rng.randint(1, 20)) if mismatch else accident
    truth_d = {("TESTIFIED_ACCIDENT_DATE", said.isoformat())}
    lines = [f"DEPOSITION OF {name.upper()}", f"Taken on {D(depo_date)}", ""]
    n = 10
    lines.append(f"{n}:1 Q. Can you tell me the date of the accident?")
    lines.append(f"{n}:2 A. " + rng.choice([f"It was {D(said)}.", f"That would be {D(said)}."]))
    n += 1
    denied_depo = []
    for bp in parts:
        lines.append(f"{n}:1 Q. Before the accident, had you ever had any problems with your {bp}?")
        if bp in prior and rng.random() < 0.5:
            ans = f"Yes, I had some {bp} pain back in {rng.randint(2015, 2021)}."
        else:
            neg_choices = ["No.", "No, never.", "Not that I recall."]
            if cfg["varied"]:
                neg_choices += ["Nothing like that.", "I've never had an issue with it."]
            ans = rng.choice(neg_choices)
            truth_d.add(("TESTIFIED_NO_PRIOR", bp))
            denied_depo.append(bp)
        lines.append(f"{n}:2 A. {ans}")
        n += 1
        if cfg["clutter"]:
            lines.append(f"{n}:1 Q. Where do you work?")
            lines.append(f"{n}:2 A. At a warehouse on Main Street.")
            n += 1
    deposition = _wrap("\n".join(lines) + "\n", cfg["wrap"])

    inconsistencies = {("PRIOR_DENIED", bp) for bp in denied_depo if bp in prior}
    if mismatch:
        inconsistencies.add(("DATE_MISMATCH",))
    return dict(id=cid, level=level, name=name,
                record=_ocr(rng, record, cfg["ocr"]), deposition=_ocr(rng, deposition, cfg["ocr"]),
                truth_record={k for k in truth}, truth_depo=truth_d, traps=traps, inconsistencies=inconsistencies)


def make_cases(seed, level, n):
    return [make_case(seed, level, i) for i in range(n)]
