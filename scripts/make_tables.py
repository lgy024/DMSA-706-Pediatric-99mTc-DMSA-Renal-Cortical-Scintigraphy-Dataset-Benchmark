#!/usr/bin/env python
"""Build Markdown and CSV tables from evaluation JSON files.

Example:
    python scripts/make_tables.py --task t1 \
        --input cnn_lstm=outputs/cnn_lstm_t1.json \
        --input qwen25_vl=outputs/qwen25_vl_t1.json \
        --out outputs/table_t1
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["t1", "t2"])
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="model name and evaluation JSON; repeat for each model",
    )
    parser.add_argument("--out", required=True, help="output prefix")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for item in args.input:
        name, path = item.split("=", 1)
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        if args.task == "t1":
            metrics = result.get("overall", result)
            rows.append(
                [
                    name,
                    metrics.get("BLEU4", 0.0),
                    metrics.get("ROUGE_L", 0.0),
                    metrics.get("CIDEr", 0.0),
                    metrics.get("F1_token", 0.0),
                ]
            )
        else:
            rows.append(
                [
                    name,
                    result.get("accuracy", 0.0),
                    result.get("macro_precision", 0.0),
                    result.get("macro_recall", 0.0),
                    result.get("macro_f1", 0.0),
                ]
            )

    if args.task == "t1":
        header = ["Model", "BLEU-4", "ROUGE-L", "CIDEr", "F1_token"]
    else:
        header = ["Model", "Accuracy", "Macro-P", "Macro-R", "Macro-F1"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[0]] + [f"{value:.3f}" for value in row[1:]])

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join([row[0]] + [f"{value:.3f}" for value in row[1:]]) + " |")
    table = "\n".join(lines)
    with open(args.out + ".md", "w", encoding="utf-8") as handle:
        handle.write(table + "\n")
    print(table)
    print(f"wrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
