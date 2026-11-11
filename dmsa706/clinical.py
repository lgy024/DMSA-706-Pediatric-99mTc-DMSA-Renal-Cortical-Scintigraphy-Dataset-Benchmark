"""Clinical attribute extraction and slot-level agreement.

The attributes mirror the five report content categories: morphology, cortical
uptake, collecting system, laterality, and the diagnostic impression.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set

from dmsa706.utils.text import extract_percentages, keyword_present, normalize_text

ATTRIBUTE_KEYWORDS: Dict[str, tuple] = {
    "morphology_small": ("外形小", "缩小", "偏小", "体积小", "萎缩"),
    "morphology_irregular": ("不规则", "欠规则", "形态失常", "形态异常"),
    "uptake_heterogeneous": ("不均匀", "不均", "稀疏", "缺损", "减低", "欠均匀"),
    "collecting_system": ("集合系统", "输尿管", "肾盂扩张", "积水", "潴留"),
    "renal_scar": ("瘢痕", "皮质缺损", "缺损灶"),
    "hydronephrosis": ("积水", "肾盂扩张", "输尿管扩张"),
    "differential_function": ("分肾功能", "分肾", "drf"),
}

LEFT_MARKERS = ("左肾", "左侧", "左")
RIGHT_MARKERS = ("右肾", "右侧", "右")
BILATERAL_MARKERS = ("双肾", "双侧", "两肾")


def extract_attributes(text: str) -> Dict[str, bool]:
    """Return a presence vector for the clinical attributes."""
    return {
        name: keyword_present(text, keywords)
        for name, keywords in ATTRIBUTE_KEYWORDS.items()
    }


def laterality(text: str) -> Set[str]:
    normalized = normalize_text(text)
    sides = set()
    if any(marker in normalized for marker in LEFT_MARKERS):
        sides.add("left")
    if any(marker in normalized for marker in RIGHT_MARKERS):
        sides.add("right")
    if any(marker in normalized for marker in BILATERAL_MARKERS):
        sides.update({"left", "right"})
    return sides


def laterality_mismatch(prediction: str, reference: str) -> bool:
    pred_sides = laterality(prediction)
    ref_sides = laterality(reference)
    if not pred_sides or not ref_sides:
        return False
    return pred_sides.isdisjoint(ref_sides)


def differential_function(text: str) -> Optional[Dict[str, float]]:
    """Parse ``左肾 45.2% / 右肾 54.8%`` style statements when present."""
    normalized = normalize_text(text)
    values = extract_percentages(normalized)
    if len(values) < 2:
        return None
    left = re.search(r"左[肾侧][^0-9]{0,12}(\d+(?:\.\d+)?)\s*[%％]", normalized)
    right = re.search(r"右[肾侧][^0-9]{0,12}(\d+(?:\.\d+)?)\s*[%％]", normalized)
    result = {}
    if left:
        result["left"] = float(left.group(1))
    if right:
        result["right"] = float(right.group(1))
    return result or None


def slot_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Per-attribute agreement for one report pair."""
    pred_attrs = extract_attributes(prediction)
    ref_attrs = extract_attributes(reference)
    scores = {}
    for name in ATTRIBUTE_KEYWORDS:
        scores[name] = 1.0 if pred_attrs[name] == ref_attrs[name] else 0.0
    scores["laterality"] = 0.0 if laterality_mismatch(prediction, reference) else 1.0
    return scores
