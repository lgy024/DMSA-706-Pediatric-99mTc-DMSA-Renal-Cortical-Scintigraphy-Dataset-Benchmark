"""Alignment and aggregation for T1 and T2 evaluation.

Reference and prediction records are matched by ``patient_id``. T1 answers are
matched by ``qid`` by default; ``strategy="order"`` uses positional pairing and
``strategy="semantic"`` falls back to a greedy similarity matcher. Only answer
text is scored, and reference-empty segments are skipped.
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Sequence, Tuple

from .labels import CATEGORY_KEYS, derive_diagnosis, label_name
from .metrics.diagnosis import classification_metrics
from .metrics.text import evaluate_text_pairs
from .qa import QUESTION_CATEGORIES


def index_by_patient(records: Sequence[dict]) -> Dict[str, dict]:
    return {record["patient_id"]: record for record in records}


def prediction_qa(pred_record: dict) -> List[Dict[str, str]]:
    return pred_record.get("qa_pred") or pred_record.get("qa") or []


def align_answers(
    ref_qa: Sequence[Dict[str, str]],
    pred_qa: Sequence[Dict[str, str]],
    strategy: str = "question_id",
) -> List[Tuple[int, str, str]]:
    """Return ``[(qid, ref_answer, pred_answer), ...]``.

    Empty reference answers are dropped. A missing prediction becomes an empty
    string, so it is scored as zero rather than silently excluded.
    """
    if strategy == "question_id":
        predicted = {int(item["qid"]): item.get("answer", "") for item in pred_qa}
        aligned = []
        for item in ref_qa:
            reference = (item.get("answer") or "").strip()
            if not reference:
                continue
            qid = int(item["qid"])
            aligned.append((qid, reference, (predicted.get(qid) or "").strip()))
        return aligned

    if strategy == "order":
        aligned = []
        for item, prediction in zip(ref_qa, pred_qa):
            reference = (item.get("answer") or "").strip()
            if not reference:
                continue
            qid = int(item.get("qid", 0))
            aligned.append((qid, reference, (prediction.get("answer") or "").strip()))
        return aligned

    if strategy == "semantic":
        return _semantic_align(ref_qa, pred_qa)

    raise ValueError(f"unknown alignment strategy: {strategy}")


def _semantic_align(
    ref_qa: Sequence[Dict[str, str]],
    pred_qa: Sequence[Dict[str, str]],
) -> List[Tuple[int, str, str]]:
    """Greedy one-to-one matching of predicted answers to reference questions.

    This deterministic string-similarity matcher is the default. It returns the
    same ``[(qid, ref_answer, pred_answer), ...]`` structure as an embedding
    matcher, so an embedding-based implementation can be substituted without
    changing the metric code.
    """
    used = set()
    aligned: List[Tuple[int, str, str]] = []
    for reference in ref_qa:
        ref_text = (reference.get("answer") or "").strip()
        if not ref_text:
            continue
        best_index, best_score = -1, -1.0
        for index, prediction in enumerate(pred_qa):
            if index in used:
                continue
            score = difflib.SequenceMatcher(
                None,
                reference.get("question", ""),
                prediction.get("question", ""),
            ).ratio()
            if score > best_score:
                best_index, best_score = index, score
        if best_index >= 0:
            used.add(best_index)
            pred_text = (pred_qa[best_index].get("answer") or "").strip()
        else:
            pred_text = ""
        aligned.append((int(reference.get("qid", 0)), ref_text, pred_text))
    return aligned


def evaluate_t1(
    ref_records: Sequence[dict],
    pred_records: Sequence[dict],
    strategy: str = "question_id",
    cider_reduction: str = "sum",
    cider_scale: float = 1.0,
) -> Dict:
    """Compute overall and per-question T1 scores."""
    predictions = index_by_patient(pred_records)
    overall_pairs: List[Tuple[str, str]] = []
    per_question: Dict[int, List[Tuple[str, str]]] = {
        c["qid"]: [] for c in QUESTION_CATEGORIES
    }

    for reference in ref_records:
        patient_id = reference["patient_id"]
        prediction = predictions.get(patient_id)
        if prediction is None:
            continue
        aligned = align_answers(reference.get("qa", []), prediction_qa(prediction), strategy)
        if not aligned:
            continue
        # Report-level concatenation in qid order.
        ref_text = "".join(item[1] for item in aligned)
        pred_text = "".join(item[2] for item in aligned)
        if ref_text.strip():
            overall_pairs.append((patient_id, pred_text, ref_text))
        for qid, ref_answer, pred_answer in aligned:
            if ref_answer.strip():
                per_question.setdefault(qid, []).append((pred_answer, ref_answer))

    pairs_only = [(pred_text, ref_text) for _, pred_text, ref_text in overall_pairs]
    result = {
        "n_patients": len(overall_pairs),
        "strategy": strategy,
        "overall": evaluate_text_pairs(pairs_only, cider_reduction, cider_scale),
        "per_question_mean": {},
        "per_patient": [],
    }
    for patient_id, pred_text, ref_text in overall_pairs:
        scores = evaluate_text_pairs([(pred_text, ref_text)], cider_reduction, cider_scale)
        scores["patient_id"] = patient_id
        result["per_patient"].append(scores)
    for qid in sorted(per_question):
        result["per_question_mean"][str(qid)] = evaluate_text_pairs(
            per_question[qid], cider_reduction, cider_scale
        )
        result["per_question_mean"][str(qid)]["n"] = len(per_question[qid])
    return result


def normalize_diagnosis_prediction(text: Optional[str]) -> Optional[str]:
    """Map a model output to a category key, or derive it from free text."""
    if not text:
        return None
    candidate = str(text).strip().lower()
    if candidate in CATEGORY_KEYS:
        return candidate
    for key in CATEGORY_KEYS:
        if key in candidate:
            return key
        if label_name(key, "zh") in str(text):
            return key
    return derive_diagnosis(str(text))


def evaluate_t2(
    ref_records: Sequence[dict],
    pred_records: Sequence[dict],
    labels: Optional[Sequence[str]] = None,
) -> Dict:
    """Compute Accuracy, Macro-P, Macro-R and Macro-F1 for T2."""
    predictions = index_by_patient(pred_records)
    y_true, y_pred, patient_ids = [], [], []

    for reference in ref_records:
        truth = reference.get("diagnosis")
        if truth is None:
            continue
        prediction = predictions.get(reference["patient_id"])
        if prediction is None:
            predicted = None
        else:
            predicted = prediction.get("diagnosis")
            if predicted is None:
                predicted = normalize_diagnosis_prediction(
                    prediction.get("report_draft")
                    or prediction.get("conclusion")
                )
        patient_ids.append(reference["patient_id"])
        y_true.append(truth)
        y_pred.append(predicted)

    metrics = classification_metrics(y_true, y_pred, labels=labels)
    metrics["n_patients"] = len(y_true)
    metrics["per_patient"] = [
        {"patient_id": pid, "true": truth, "pred": predicted, "correct": int(truth == predicted)}
        for pid, truth, predicted in zip(patient_ids, y_true, y_pred)
    ]
    return metrics
