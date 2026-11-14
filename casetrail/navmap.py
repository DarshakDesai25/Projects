"""The navigation map: an index of what is in each file and where.

It is built once from the case files. An agent uses it only to decide which pages to read.
Nothing is ever cited from the map. Every citation comes from the page text itself.
"""
import math
import re
from collections import Counter, defaultdict

from dates import find_dates

STOP = {"the", "and", "for", "with", "was", "his", "her", "had", "any", "you", "your", "that", "this", "from", "has",
        "are", "were", "been", "have", "not", "but", "patient", "page", "of", "to", "in", "on", "at", "by", "or", "an", "a"}
SECTIONS = ["Date of service", "Provider", "Therapist", "Assessment", "Plan", "History", "Family history", "Exam",
            "FINDINGS", "IMPRESSION", "Dispensed", "CPT", "Q.", "A."]


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()) if len(w) > 1 and w not in STOP]


class NavMap:
    def __init__(self, case):
        self.files = []
        self.tf = defaultdict(Counter)            # term -> {(file, page): count}
        self.n_pages = 0
        for f in case["files"]:
            pages = []
            for pno, lines in enumerate(f["pages"], 1):
                body = "\n".join(lines[1:])       # the header repeats on every page, so it is kept out of the index
                for t in tokens(body):
                    self.tf[t][(f["file_id"], pno)] += 1
                ds = sorted({d[2] for d in find_dates(body)})
                pages.append({"page": pno, "sections": [s for s in SECTIONS if s in body],
                              "dates": [ds[0], ds[-1]] if ds else []})
                self.n_pages += 1
            self.files.append({"file_id": f["file_id"], "title": f["title"], "doc_type": f["doc_type"],
                               "patient": f["patient"], "n_pages": len(f["pages"]), "pages": pages})
        self.by_id = {f["file_id"]: f for f in self.files}

    def summary(self):
        """What an agent sees when it lists the files: titles, types, patients and page counts."""
        return [{k: f[k] for k in ("file_id", "title", "doc_type", "patient", "n_pages")} for f in self.files]

    def search(self, query, doc_types=None, top=5):
        """Pages ranked by how well they match the query words, with a small idf weight."""
        scores = Counter()
        for t in set(tokens(query)):
            postings = self.tf.get(t, {})
            if not postings:
                continue
            idf = math.log(1 + self.n_pages / len(postings))
            for loc, c in postings.items():
                scores[loc] += idf * c / (c + 1)
        hits = []
        for (fid, pno), s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            f = self.by_id[fid]
            if doc_types and f["doc_type"] not in doc_types:
                continue
            hits.append({"file": fid, "page": pno, "title": f["title"], "doc_type": f["doc_type"],
                         "patient_dob": f["patient"]["dob"], "sections": f["pages"][pno - 1]["sections"],
                         "dates": f["pages"][pno - 1]["dates"], "score": round(s, 3)})
            if len(hits) >= top:
                break
        return hits
