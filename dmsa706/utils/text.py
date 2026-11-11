"""Chinese text helpers shared by clinical extraction and error analysis."""

from __future__ import annotations

import re
from typing import Iterable, List

NEGATION_MARKERS = ("未见", "未", "无", "否认", "排除", "不考虑")
PUNCTUATION = "，。；：、！？,.;:!? \n\t"


def normalize_text(text: str) -> str:
    """Full-width to half-width and lowercase."""
    if not text:
        return ""
    characters = []
    for char in str(text):
        code = ord(char)
        if code == 0x3000:
            characters.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            characters.append(chr(code - 0xFEE0))
        else:
            characters.append(char)
    return "".join(characters).lower().strip()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。；;！？!?])", text or "")
    return [part.strip() for part in parts if part and part.strip()]


def has_negation(text: str) -> bool:
    window = normalize_text(text)
    return any(marker in window for marker in NEGATION_MARKERS)


def keyword_present(text: str, keywords: Iterable[str], window: int = 6) -> bool:
    """True when a keyword occurs outside a nearby negation."""
    normalized = normalize_text(text)
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), normalized):
            start = max(0, match.start() - window)
            end = min(len(normalized), match.end() + 2)
            if not has_negation(normalized[start:end]):
                return True
    return False


def extract_percentages(text: str) -> List[float]:
    values = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*％", text or ""):
        values.append(float(match.group(1) or match.group(2)))
    return values


def strip_punctuation(text: str) -> str:
    return "".join(char for char in (text or "") if char not in PUNCTUATION)
