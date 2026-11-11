#!/usr/bin/env python
"""Flag laterality errors, contradictions and generic answers."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.analysis import collect_error_cases, summarize_cases  # noqa: E402
from dmsa706.io import load_records  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    references = load_records(args.refs)
    predictions = load_records(args.preds)
    cases = collect_error_cases(references, predictions)
    summary = summarize_cases(cases)

    payload = {"summary": summary, "cases": cases}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
