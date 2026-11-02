"""Readable names for the raw class labels of the different models (used for display and comparison only)."""
from __future__ import annotations

import re

_SPECIAL = {
    ".DS_Store": "(.DS_Store - invalid output 0 of the custom CNN)",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider mites",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellow leaf curl virus",
    "Tomato__Tomato_mosaic_virus": "Mosaic virus",
    "Spider Mite Damage": "Spider mites",
}


def readable(label: str | None) -> str:
    if not label:
        return "-"
    if label in _SPECIAL:
        return _SPECIAL[label]
    if label.startswith("(") or label == "Unknown":
        return label
    m = re.match(r"^(Pepper__bell|Potato)_+(.*)$", label)
    prefix = ""
    if m:
        prefix = ("Pepper (bell)" if m.group(1).startswith("Pepper") else "Potato") + ": "
        label = m.group(2)
    label = re.sub(r"^(tomato|mango)_+", "", label, flags=re.I)
    s = re.sub(r"\s+", " ", label.replace("_", " ")).strip()
    return prefix + (s[:1].upper() + s[1:].lower())


def score_text(result: dict) -> str:
    return "-" if result.get("score") is None else f"{result['score']:.3f}"
