"""Invented case folders with many files and an exact answer key, down to the page.

Each case is one made up plaintiff. Its folder holds several files, in a shuffled order, as
they would arrive in a legal production.
  emergency record   the first visit, with the accident date and the patient's history
  clinic records     follow up visits with diagnoses and prescriptions
  therapy notes      physical therapy visits, with no diagnoses or prescriptions
  radiology report   an MRI with findings, in most cases
  pharmacy record    the same prescriptions again, as dispensed
  billing            codes and amounts only
  deposition         the plaintiff's testimony
In the large size, a folder also has a duplicate copy of one file and a record of a different
person who has the same name, which is a common mistake in real productions.

Every page starts with a header naming the patient, date of birth and record number, and pages
never split a visit or a question and answer, so every fact sits on one page. The answer key
lists each fact and every page where it appears.

Traps are statements a careless reader mistakes for facts.
  NEGATION_ERROR      "fracture ruled out", "patient denies prior neck problems"
  WRONG_PERSON        "family history of neck pain (mother)"
  WRONG_PERSON_FILE   anything in the other person's record
  RETRACTED           in the large size, a diagnosis that a later visit says was entered in error.
                      The original line still looks like a normal diagnosis, so only a reader who
                      also saw the later addendum can know it is not a fact.
"""
import random
from datetime import date, timedelta

from dates import MONTHS

FIRST = ["Maria", "James", "Priya", "Daniel", "Aisha", "Robert", "Chen", "Olivia", "Marcus", "Elena"]
LAST = ["Alvarez", "Thompson", "Patel", "Nguyen", "Okafor", "Schmidt", "Rossi", "Kim", "Harris", "Moreau"]
PROVIDERS = ["Dr. Patel", "Dr. Lindqvist", "Dr. Okoro", "Dr. Ramirez", "Dr. Whitfield", "Dr. Sato", "Dr. Brennan"]
THERAPISTS = ["J. Moore, PT", "A. Liu, PT", "S. Grant, PT"]
BODY_PARTS = ["lower back", "neck", "left shoulder", "right knee", "right wrist"]
DIAGNOSES = {"lower back": ["lumbar strain", "lumbar disc bulge"], "neck": ["cervical sprain", "whiplash"],
             "left shoulder": ["rotator cuff strain", "shoulder contusion"],
             "right knee": ["knee sprain", "meniscus tear"], "right wrist": ["wrist sprain", "scaphoid contusion"]}
MRI_FINDINGS = {"lower back": ["l4-l5 disc bulge", "mild facet arthropathy"], "neck": ["c5-c6 disc protrusion", "straightening of the cervical lordosis"],
                "left shoulder": ["supraspinatus tendinosis", "small joint effusion"],
                "right knee": ["medial meniscus tear", "small joint effusion"], "right wrist": ["scaphoid bone bruise", "mild soft tissue swelling"]}
RULED_OUT = ["fracture", "spinal cord injury", "concussion", "dislocation"]
DRUGS = ["ibuprofen", "naproxen", "cyclobenzaprine", "tramadol", "gabapentin", "meloxicam"]
DOSES = {"ibuprofen": "600 mg", "naproxen": "500 mg", "cyclobenzaprine": "10 mg", "tramadol": "50 mg",
         "gabapentin": "300 mg", "meloxicam": "15 mg"}

SIZES = {
    "small": dict(clinic_files=1, visits_per_file=2, therapy_files=1, therapy_visits=(3, 5), billing_files=1,
                  duplicate=False, other_person=False, varied=False, retraction=0.0),
    "large": dict(clinic_files=4, visits_per_file=5, therapy_files=3, therapy_visits=(8, 12), billing_files=4,
                  duplicate=True, other_person=True, varied=True, retraction=0.6),
}
PAGE_LINES = 22

QUESTIONS = [
    {"id": "accident", "text": "On what date did the accident happen, according to the medical records?", "types": ["ACCIDENT_DATE"]},
    {"id": "diagnoses", "text": "List every diagnosis the plaintiff received, with the visit date.", "types": ["DIAGNOSIS"]},
    {"id": "medications", "text": "List every medication prescribed to the plaintiff, with the date.", "types": ["MEDICATION"]},
    {"id": "prior", "text": "Did the plaintiff have problems with any body part before the accident? List each prior condition.", "types": ["PRIOR_CONDITION"]},
    {"id": "mri", "text": "What did the plaintiff's MRI show? List each finding, or say there was no MRI.", "types": ["MRI_FINDING"]},
    {"id": "contradictions", "text": "Where does the plaintiff's deposition testimony contradict the medical records?", "types": ["PRIOR_DENIED", "DATE_MISMATCH"]},
]


def key(fact):
    """A fact as a string, for dictionaries and JSON."""
    return "|".join(str(x) for x in fact)


def fmt_date(d, style):
    if style == 0:
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == 1:
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == 2:
        return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"
    return d.isoformat()


def line(text, facts=(), traps=()):
    return {"text": text, "facts": list(facts), "traps": list(traps)}


def visit_block(rng, v, D, varied, first=None):
    """One clinic visit. `first` holds the accident and history lines for the emergency visit."""
    vd = v["date"].isoformat()
    lines = [line(f"Date of service: {D(v['date'])}"), line(f"Provider: {v['provider']}")]
    if first:
        lines += first
    else:
        lines.append(line(f"Follow up for {v['parts'][0]} pain."))
    neg = rng.choice([f"{v['ruled_out']} ruled out", f"no evidence of {v['ruled_out']}", f"negative for {v['ruled_out']}",
                      f"{v['ruled_out']} was excluded"] if varied else [f"{v['ruled_out']} ruled out"])
    items = list(v["dx"])
    items.insert(rng.randint(0, len(items)), neg)
    trap_kind = v.get("trap_kind")
    if trap_kind:      # the other person's record: everything in it is a wrong person fact
        lines.append(line("Assessment: " + "; ".join(items) + ".", traps=[(("DIAGNOSIS", vd, dx), trap_kind) for dx in v["dx"] + [v["ruled_out"]]]))
    else:
        kept = [dx for dx in v["dx"] if dx != v.get("retracted")]
        traps = [(("DIAGNOSIS", vd, v["ruled_out"]), "NEGATION_ERROR")] + ([(("DIAGNOSIS", vd, v["retracted"]), "RETRACTED")] if v.get("retracted") else [])
        lines.append(line("Assessment: " + "; ".join(items) + ".", facts=[("DIAGNOSIS", vd, dx) for dx in kept], traps=traps))
    for dx, when in v.get("addenda", []):
        lines.append(line(f"Addendum: {dx} recorded on {D(when)} was entered in error."))
    for j, drug in enumerate(v["meds"]):
        verb = rng.choice(["Start", "Prescribed", "Rx:"]) if varied else ("Start" if j == 0 else "Prescribed")
        fact = ("MEDICATION", vd, drug)
        lines.append(line(f"Plan: {verb} {drug} {DOSES[drug]}.", facts=[] if trap_kind else [fact],
                          traps=[(fact, trap_kind)] if trap_kind else []))
    lines.append(line("Plan: Follow up in 2 weeks."))
    lines.append(line(""))
    return lines


def paginate(blocks, title, header):
    """Pages of at most PAGE_LINES lines, never splitting a block."""
    pages, cur = [], []
    for b in blocks:
        if cur and len(cur) + len(b) > PAGE_LINES:
            pages.append(cur)
            cur = []
        cur = cur + b
    if cur:
        pages.append(cur)
    n = len(pages)
    return [[line(header(i + 1, n))] + p for i, p in enumerate(pages)]


def make_case(seed, size, cid=0):
    cfg = SIZES[size]
    rng = random.Random(f"{seed}-{size}-{cid}")
    D = lambda d: fmt_date(d, rng.choice([0, 1, 2, 3]) if cfg["varied"] else 0)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    dob = date(1960, 1, 1) + timedelta(days=rng.randint(0, 12000))
    mrn = f"MRN-{rng.randint(100000, 999999)}"
    accident = date(2023, 1, 1) + timedelta(days=rng.randint(0, 600))
    parts = rng.sample(BODY_PARTS, rng.choice([1, 2]))
    others = [b for b in BODY_PARTS if b not in parts]
    prior = [p for p in parts if rng.random() < 0.4]
    denied = rng.sample(others, 1)
    family_bp = rng.choice([b for b in others if b not in denied])
    ruled_out = rng.choice(RULED_OUT)
    relative = rng.choice(["mother", "sister"]) if cfg["varied"] else "mother"

    def header_for(title, who):
        return lambda i, n: f"{title} | Patient: {who['name']} | DOB: {who['dob']} | MRN: {who['mrn']} | Page {i} of {n}"

    me = {"name": name, "dob": dob.isoformat(), "mrn": mrn}
    files = []

    def add(title, doc_type, who, blocks, header=None):
        files.append({"title": title, "doc_type": doc_type, "patient": who,
                      "pages": paginate(blocks, title, header or header_for(title, who))})

    # visits
    n_clinic = cfg["clinic_files"] * cfg["visits_per_file"]
    visits, d = [], accident + timedelta(days=rng.randint(0, 2))
    for i in range(1 + n_clinic):
        dx = [rng.choice(DIAGNOSES[bp]) if i == 0 else DIAGNOSES[bp][0] if rng.random() < 0.7 else rng.choice(DIAGNOSES[bp]) for bp in parts]
        visits.append({"date": d, "provider": rng.choice(PROVIDERS), "dx": list(dict.fromkeys(dx)), "parts": parts,
                       "meds": rng.sample(DRUGS, rng.choice([1, 2])) if rng.random() < 0.6 else [], "ruled_out": ruled_out})
        d = d + timedelta(days=rng.randint(7, 21))
    step = cfg["visits_per_file"]
    if rng.random() < cfg["retraction"] and len(visits) > 1 + step:
        i = rng.randint(1, len(visits) - 1 - step)    # a clinic visit, corrected in a later visit in another file
        visits[i]["retracted"] = visits[i]["dx"][-1]
        visits[i + step].setdefault("addenda", []).append((visits[i]["dx"][-1], visits[i]["date"]))

    # emergency record, with the accident date and history
    first = [line(f"Patient reports {parts[0]} pain after a motor vehicle accident on {D(accident)}.", facts=[("ACCIDENT_DATE", accident.isoformat())])]
    for bp in prior:
        yr = rng.randint(2015, 2021)
        first.append(line(rng.choice([f"History of {bp} pain since {yr}.", f"Prior {bp} injury in {yr}.", f"Hx of {bp} pain ({yr})."]
                                     if cfg["varied"] else [f"History of {bp} pain since {yr}."]), facts=[("PRIOR_CONDITION", bp)]))
    for bp in denied:
        first.append(line(rng.choice([f"Patient denies prior {bp} problems.", f"No history of {bp} injury.", f"Patient states no previous {bp} complaints."]
                                     if cfg["varied"] else [f"Patient denies prior {bp} problems."]), traps=[(("PRIOR_CONDITION", bp), "NEGATION_ERROR")]))
    first.append(line(f"Family history of {family_bp} pain ({relative})." if relative == "mother" else f"Patient's sister had a prior {family_bp} injury.",
                      traps=[(("PRIOR_CONDITION", family_bp), "WRONG_PERSON")]))
    add("EMERGENCY DEPARTMENT RECORD", "medical_record", me, [visit_block(rng, visits[0], D, cfg["varied"], first=first)])

    clinic_titles = ["ORTHOPEDIC CLINIC NOTES", "PRIMARY CARE NOTES", "SPINE CENTER NOTES", "PAIN CLINIC NOTES"]
    for f in range(cfg["clinic_files"]):
        chunk = visits[1 + f * cfg["visits_per_file"]: 1 + (f + 1) * cfg["visits_per_file"]]
        add(clinic_titles[f], "medical_record", me, [visit_block(rng, v, D, cfg["varied"]) for v in chunk])

    for f in range(cfg["therapy_files"]):
        blocks, td = [], accident + timedelta(days=rng.randint(10, 40))
        for _ in range(rng.randint(*cfg["therapy_visits"])):
            blocks.append([line(f"Date of service: {D(td)}"), line(f"Therapist: {rng.choice(THERAPISTS)}"),
                           line(f"Pt reports {rng.choice(parts)} pain improving with exercises."),
                           line("Plan: continue home exercise program."), line("")])
            td += timedelta(days=rng.randint(3, 10))
        add(f"PHYSICAL THERAPY NOTES {f + 1}", "therapy_notes", me, blocks)

    if rng.random() < 0.7:
        bp = parts[0]
        exam = accident + timedelta(days=rng.randint(14, 45))
        found = MRI_FINDINGS[bp]
        neg = rng.choice(["no acute fracture", "fracture is not seen", "negative for fracture"] if cfg["varied"] else ["no acute fracture"])
        facts = [("MRI_FINDING", x) for x in found]
        add("RADIOLOGY REPORT", "radiology", me, [[
            line(f"Exam: MRI of the {bp}"), line(f"Date of exam: {D(exam)}"),
            line("FINDINGS: " + "; ".join(found) + ".", facts=facts),
            line("IMPRESSION: " + "; ".join(found + [neg]) + ".", facts=facts, traps=[(("MRI_FINDING", "fracture"), "NEGATION_ERROR")]),
            line("")]])

    disp = [line(f"{v['date'].month:02d}/{v['date'].day:02d}/{v['date'].year}  Dispensed: {drug} {DOSES[drug]}  Qty 30  Prescriber: {v['provider']}",
                 facts=[("MEDICATION", v["date"].isoformat(), drug)]) for v in visits for drug in v["meds"]]
    if disp:
        add("PHARMACY DISPENSING RECORD", "pharmacy", me, [[x] for x in disp])

    for f in range(cfg["billing_files"]):
        blocks = []
        for v in rng.sample(visits, min(len(visits), 8)):
            blocks.append([line(f"{v['date'].isoformat()}  CPT 99213  Office visit  ${rng.randint(120, 240)}.00"),
                           line(f"{v['date'].isoformat()}  ICD-10 code on file  Diagnosis code entered"), line("")])
        blocks.append([line(f"Balance due: ${rng.randint(200, 4000)}.00")])
        add(f"BILLING STATEMENT {f + 1}", "billing", me, blocks)

    # deposition
    mismatch = rng.random() < 0.2
    said = accident + timedelta(days=rng.randint(1, 20)) if mismatch else accident
    blocks, n = [[line("Taken on " + D(max(v["date"] for v in visits) + timedelta(days=rng.randint(30, 120)))), line("")]], 10
    blocks.append([line(f"{n}:1 Q. Can you tell me the date of the accident?"),
                   line(f"{n}:2 A. " + rng.choice([f"It was {D(said)}.", f"That would be {D(said)}."]), facts=[("TESTIFIED_ACCIDENT_DATE", said.isoformat())])])
    denied_depo = []
    for bp in parts:
        n += 1
        if bp in prior and rng.random() < 0.5:
            ans, facts = f"Yes, I had some {bp} pain back in {rng.randint(2015, 2021)}.", []
        else:
            ans = rng.choice(["No.", "No, never.", "Not that I recall."] + (["Nothing like that.", "I've never had an issue with it."] if cfg["varied"] else []))
            facts = [("TESTIFIED_NO_PRIOR", bp)]
            denied_depo.append(bp)
        blocks.append([line(f"{n}:1 Q. Before the accident, had you ever had any problems with your {bp}?"), line(f"{n}:2 A. {ans}", facts=facts)])
        if cfg["varied"]:
            n += 1
            blocks.append([line(f"{n}:1 Q. Where do you work?"), line(f"{n}:2 A. At a warehouse on Main Street.")])
    add("DEPOSITION TRANSCRIPT", "deposition", {"name": name, "dob": None, "mrn": None}, blocks,
        header=lambda i, n: f"DEPOSITION OF {name.upper()} | Page {i} of {n}")

    if cfg["other_person"]:        # same name, different person
        odob = date(1960, 1, 1) + timedelta(days=rng.randint(0, 12000))
        other = {"name": name, "dob": odob.isoformat(), "mrn": f"MRN-{rng.randint(100000, 999999)}"}
        obp = rng.choice([b for b in BODY_PARTS if b not in prior])
        ov = {"date": date(2023, 1, 1) + timedelta(days=rng.randint(0, 700)), "provider": rng.choice(PROVIDERS),
              "dx": [rng.choice(DIAGNOSES[obp])], "parts": [obp], "meds": rng.sample(DRUGS, 1), "ruled_out": rng.choice(RULED_OUT),
              "trap_kind": "WRONG_PERSON_FILE"}
        hist = [line(f"Patient reports {obp} pain."), line(f"History of {obp} pain since {rng.randint(2015, 2021)}.",
                                                            traps=[(("PRIOR_CONDITION", obp), "WRONG_PERSON_FILE")])]
        add("CLINIC RECORD", "medical_record", other, [visit_block(rng, ov, D, cfg["varied"], first=hist)])

    if cfg["duplicate"]:
        src = next(f for f in files if f["title"] == clinic_titles[0])
        files.append({**src, "title": "PRODUCED COPY " + src["title"], "pages": [list(p) for p in src["pages"]]})

    rng.shuffle(files)
    gold_facts, locations, traps = set(), {}, {}
    for i, f in enumerate(files, 1):
        f["file_id"] = f"F{i:02d}"
        for pno, page in enumerate(f["pages"], 1):
            for ln in page:
                for fact in ln["facts"]:
                    gold_facts.add(tuple(fact))
                    locations.setdefault(key(fact), set()).add((f["file_id"], pno))
                for fact, kind in ln["traps"]:
                    traps.setdefault(key(fact), kind)
                    locations.setdefault(key(fact), set()).add((f["file_id"], pno))
    inconsist = {("PRIOR_DENIED", bp) for bp in denied_depo if bp in prior} | ({("DATE_MISMATCH",)} if mismatch else set())
    gold = {q["id"]: sorted(list(x) for x in gold_facts if x[0] in q["types"]) for q in QUESTIONS if q["id"] != "contradictions"}
    gold["contradictions"] = sorted(list(x) for x in inconsist)
    for k in [k for k in traps if tuple(k.split("|")) in gold_facts]:
        del traps[k]                  # a trap that matches a true fact is not a trap
    return {"id": f"{size}-{cid}", "size": size, "plaintiff": me,
            "files": [{"file_id": f["file_id"], "title": f["title"], "doc_type": f["doc_type"], "patient": f["patient"],
                       "pages": [[ln["text"] for ln in p] for p in f["pages"]]} for f in files],
            "gold": gold, "locations": {k: sorted(v) for k, v in locations.items()}, "traps": traps,
            "questions": [{"id": q["id"], "text": q["text"]} for q in QUESTIONS]}


def make_cases(seed, size, n):
    return [make_case(seed, size, i) for i in range(n)]
