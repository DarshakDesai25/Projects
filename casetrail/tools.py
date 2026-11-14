"""The tools an agent can call, and a record of what it read.

  list_files()                  titles, document types, patients and page counts
  search_map(query, doc_types)  the best matching pages, from the navigation map
  open_page(file, page)         the full text of one page
Every page opened is counted, so the cost of an answer is measured, not guessed.
"""
from collections import Counter

from navmap import NavMap


class CaseEnv:
    def __init__(self, case, page_budget=None):
        self.case = case
        self.map = NavMap(case)
        self.page_budget = page_budget
        self.opened = []                          # unique (file, page), in the order first opened
        self.calls = Counter()
        self._files = {f["file_id"]: f for f in case["files"]}

    def list_files(self):
        self.calls["list_files"] += 1
        return self.map.summary()

    def search_map(self, query, doc_types=None, top=5):
        self.calls["search_map"] += 1
        return self.map.search(query, doc_types, top)

    def open_page(self, file, page):
        self.calls["open_page"] += 1
        text = self.page_text(file, page)
        if text is None:
            return {"error": f"no page {page} in file {file}"}
        loc = (file, int(page))
        if loc not in self.opened:
            if self.page_budget is not None and len(self.opened) >= self.page_budget:
                return {"error": "page budget reached"}
            self.opened.append(loc)
        return {"file": file, "page": int(page), "text": text}

    def page_text(self, file, page):
        """Page text without counting it as read. Used by the verifier, which checks quotes in code."""
        f = self._files.get(file)
        try:
            page = int(page)
        except (TypeError, ValueError):
            return None
        if f is None or not 1 <= page <= len(f["pages"]):
            return None
        return "\n".join(f["pages"][page - 1])

    def file_patient(self, file):
        f = self._files.get(file)
        return (f["doc_type"], f["patient"]) if f else (None, None)
