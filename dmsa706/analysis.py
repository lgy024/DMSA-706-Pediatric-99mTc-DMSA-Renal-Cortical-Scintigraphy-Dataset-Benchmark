"""Error analysis for generated reports.

Flags three failure modes discussed in the paper: laterality errors,
contradictions between local findings and the conclusion, and generic medical
statements that replace case-specific observations.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from dmsa706.clinical import laterality_mismatch
from dmsa706.utils.text import has_negation, normalize_text, split_sentences

GENERIC_PATTERNS = (
    "包括",
    "如",
    "等",
    "可能表现为",
    "常见于",
)


def flag_generic_answer(text: str) -> bool:
    """A long answer built from generic disease lists rather than this case."""
    sentences = split_sentences(text)
    if not sentences:
        return False
    generic = sum(
        1
        for sentence in sentences
        if any(pattern in sentence for pattern in GENERIC_PATTERNS)
    )
    return generic >= max(1, len(sentences) // 2)


def contradiction_flags(prediction: str) -> List[str]:
    """Detect an abnormal finding followed by a fully normal conclusion."""
    flags = []
    sentences = split_sentences(prediction)
    abnormal = ("稀疏", "不均匀", "缺损", "瘢痕", "积水", "未显影", "功能受损")
    normal = ("未见明显异常", "未见异常", "正常")
    has_abnormal = any(
        any(word in sentence for word in abnormal) and not has_negation(sentence)
        for sentence in sentences
    )
    has_normal_conclusion = any(
        any(word in sentence for word in normal) for sentence in sentences[-2:]
    )
    if has_abnormal and has_normal_conclusion:
        flags.append("abnormal_finding_with_normal_conclusion")
    return flags


def collect_error_cases(
    references: Sequence[dict],
    predictions: Sequence[dict],
) -> List[Dict]:
    """Return flagged patient cases for manual review."""
    ref_map = {record["patient_id"]: record for record in references}
    cases = []
    for prediction in predictions:
        reference = ref_map.get(prediction["patient_id"])
        if reference is None:
            continue
        pred_text = prediction.get("report_draft") or "".join(
            item.get("answer", "") for item in prediction.get("qa_pred", [])
        )
        ref_text = "".join(
            item.get("answer", "")
            for item in reference.get("qa", [])
            if item.get("answer", "").strip()
        )
        flags = []
        if laterality_mismatch(pred_text, ref_text):
            flags.append("laterality_mismatch")
        if flag_generic_answer(pred_text):
            flags.append("generic_answer")
        flags.extend(contradiction_flags(pred_text))
        if flags:
            cases.append(
                {
                    "patient_id": prediction["patient_id"],
                    "flags": flags,
                    "prediction": pred_text,
                    "reference": ref_text,
                }
            )
    return cases


def summarize_cases(cases: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in cases:
        for flag in case["flags"]:
            counts[flag] = counts.get(flag, 0) + 1
    counts["total_cases"] = len(cases)
    return counts
