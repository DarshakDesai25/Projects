import re


def collapse(text):
    """Page headers dropped and all whitespace squeezed to single spaces."""
    text = re.sub(r"Page \d+ of \d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
