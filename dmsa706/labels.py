"""Report-derived diagnostic categories (T2).

The paper's labels were verified by clinicians. This module provides a
transparent, ordered keyword reference so the task can be reconstructed from
report conclusions. The rule is the default; if you have an annotation file,
``data_prep/build_records.py --labels`` joins a ``patient_id,diagnosis`` CSV
instead.

Rule order matters. Specific structural findings are checked before the
``normal`` category, so "right renal scar; left kidney unremarkable" is
labelled ``renal_scar`` rather than ``normal``. A conclusion that mentions a
non-functioning kidney is abnormal even when the other kidney is described as
unremarkable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

DIAGNOSIS_CATEGORIES: List[Dict[str, str]] = [
    {"key": "pyelonephritis", "zh": "肾盂肾炎", "en": "Pyelonephritis"},
    {"key": "normal", "zh": "未见明显异常", "en": "No abnormality detected"},
    {"key": "renal_scar", "zh": "肾瘢痕", "en": "Renal scar"},
    {"key": "hydronephrosis", "zh": "肾积水", "en": "Hydronephrosis"},
    {
        "key": "dysplasia_or_agenesis",
        "zh": "肾发育不良或缺如",
        "en": "Renal dysplasia or agenesis",
    },
    {
        "key": "duplex_kidney",
        "zh": "重复肾或重复集合系统",
        "en": "Duplex kidney or collecting system",
    },
    {
        "key": "abnormal_drf",
        "zh": "分肾功能异常",
        "en": "Abnormal differential renal function",
    },
]

CATEGORY_KEYS = [c["key"] for c in DIAGNOSIS_CATEGORIES]
KEY_TO_ZH = {c["key"]: c["zh"] for c in DIAGNOSIS_CATEGORIES}
ZH_TO_KEY = {c["zh"]: c["key"] for c in DIAGNOSIS_CATEGORIES}

# Ordered: more specific findings first, ``normal`` last.
_REQUIRED_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("duplex_kidney", ("重复肾", "重复肾盂", "双肾盂", "重复输尿管", "重复集合系统")),
    (
        "dysplasia_or_agenesis",
        (
            "发育不良",
            "肾缺如",
            "缺如",
            "未发育",
            "先天性缺如",
            "萎缩",
            "双肾偏小",
            "双肾小",
            "双肾外形偏小",
            "双肾体积缩小",
        ),
    ),
    ("renal_scar", ("瘢痕", "皮质缺损")),
    (
        "hydronephrosis",
        ("积水", "肾盂扩张", "输尿管扩张", "输尿管迂曲扩张", "潴留"),
    ),
    ("pyelonephritis", ("肾盂肾炎", "感染灶", "炎症")),
    (
        "abnormal_drf",
        (
            "分肾功能",
            "DRF",
            "功能受损",
            "功能减低",
            "功能下降",
            "功能降低",
            "皮质功能降低",
            "功能偏低",
            "功能略低",
            "功能差",
            "无功能",
            "几无功能",
            "未显影",
            "实质无功能",
        ),
    ),
]

_NORMAL_RULES = ("未见明显异常", "未见异常", "未见明确异常", "正常")


def derive_diagnosis(conclusion: str) -> Optional[str]:
    """Return a category key, or ``None`` when no rule matches."""
    text = (conclusion or "").strip()
    if not text:
        return None
    for key, keywords in _REQUIRED_RULES:
        if any(word in text for word in keywords):
            return key
    if any(word in text for word in _NORMAL_RULES):
        return "normal"
    return None


def derive_diagnosis_with_reason(conclusion: str) -> Tuple[Optional[str], str]:
    """Return ``(category_key, matched_keyword)`` for auditability."""
    text = (conclusion or "").strip()
    if not text:
        return None, ""
    for key, keywords in _REQUIRED_RULES:
        for word in keywords:
            if word in text:
                return key, word
    for word in _NORMAL_RULES:
        if word in text:
            return "normal", word
    return None, ""


def label_name(key: Optional[str], lang: str = "zh") -> str:
    if key is None:
        return ""
    for category in DIAGNOSIS_CATEGORIES:
        if category["key"] == key:
            return category[lang]
    return key
