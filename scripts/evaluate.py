#!/usr/bin/env python
"""Compute T1 or T2 metrics from a predictions JSONL file.

Examples:
    python scripts/evaluate.py --task t1 --refs data/records.jsonl \
        --preds outputs/qwen25_vl_test.jsonl --out outputs/qwen25_vl_t1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.evaluation import evaluate_t1, evaluate_t2  # noqa: E402
from dmsa706.io import load_records  # noqa: E402
from dmsa706.labels import CATEGORY_KEYS  # noqa: E402
from dmsa706.metrics.text import evaluate_text_pairs, set_tokenizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["t1", "t2"])
    parser.add_argument("--refs", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy", default="question_id", choices=["question_id", "order", "semantic"])
    parser.add_argument("--tokenizer", default="auto", choices=["auto", "jieba", "char"])
    parser.add_argument("--cider-reduction", default="sum", choices=["sum", "mean"])
    parser.add_argument("--cider-scale", type=float, default=1.0)
    parser.add_argument("--clinical", action="store_true", help="add slot-level clinical metrics")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for CIs")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def evaluate_report_text(refs, preds, args):
    ref_map = {record["patient_id"]: record for record in refs}
    pairs = []
    for prediction in preds:
        reference = ref_map.get(prediction["patient_id"])
        if reference is None:
            continue
        ref_text = "".join(
            item.get("answer", "")
            for item in reference.get("qa", [])
            if item.get("answer", "").strip()
        )
        pred_text = prediction.get("report_draft", "")
        if ref_text.strip():
            pairs.append((pred_text, ref_text))
    return {
        "n_patients": len(pairs),
        "overall": evaluate_text_pairs(
            pairs, args.cider_reduction, args.cider_scale
        ),
    }


def main():
    args = parse_args()
    set_tokenizer(args.tokenizer)
    refs = load_records(args.refs)
    preds = load_records(args.preds)

    if args.task == "t1":
        has_qa = any(prediction.get("qa_pred") for prediction in preds)
        if has_qa:
            result = evaluate_t1(
                refs,
                preds,
                strategy=args.strategy,
                cider_reduction=args.cider_reduction,
                cider_scale=args.cider_scale,
            )
        else:
            result = evaluate_report_text(refs, preds, args)
    else:
        result = evaluate_t2(refs, preds, labels=CATEGORY_KEYS)

    if args.task == "t1" and args.clinical:
        from dmsa706.metrics.clinical import evaluate_clinical

        ref_map = {
            record["patient_id"]: "".join(
                item.get("answer", "")
                for item in record.get("qa", [])
                if item.get("answer", "").strip()
            )
            for record in refs
        }
        pred_map = {
            record["patient_id"]: record.get("report_draft")
            or "".join(
                item.get("answer", "") for item in record.get("qa_pred", [])
            )
            for record in preds
        }
        result["clinical"] = evaluate_clinical(pred_map, ref_map)

    if args.bootstrap and result.get("per_patient"):
        from dmsa706.metrics.bootstrap import bootstrap_ci

        if args.task == "t1":
            keys = ("ROUGE_L", "F1_token", "CIDEr")
            result["bootstrap"] = {
                key: bootstrap_ci(
                    [row[key] for row in result["per_patient"]],
                    n_boot=args.bootstrap,
                    seed=args.seed,
                )
                for key in keys
            }
        else:
            result["bootstrap"] = {
                "accuracy": bootstrap_ci(
                    [row["correct"] for row in result["per_patient"]],
                    n_boot=args.bootstrap,
                    seed=args.seed,
                )
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2)[:1200])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
