"""Descriptive statistics helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def mean_std(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
    }


def percentile(values: Iterable[float], q: Sequence[float] = (25, 50, 75)) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {f"p{int(point)}": 0.0 for point in q}
    points = np.percentile(array, q)
    return {f"p{int(point)}": float(value) for point, value in zip(q, points)}


def distribution(values: Iterable) -> dict:
    counts = Counter(values)
    total = sum(counts.values())
    return {
        str(key): {
            "count": int(count),
            "ratio": count / total if total else 0.0,
        }
        for key, count in counts.most_common()
    }


def length_stats(texts: Iterable[str]) -> dict:
    lengths = [len(text or "") for text in texts]
    result = mean_std(lengths)
    result.update(percentile(lengths))
    return result
