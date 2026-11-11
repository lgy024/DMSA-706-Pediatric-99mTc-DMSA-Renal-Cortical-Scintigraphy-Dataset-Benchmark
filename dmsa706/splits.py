"""Patient-level cohort splits for N = 250, 500 and 706.

Each setting samples ``N`` patients, then allocates them 70 / 10 / 20 to
train / validation / test. Patients are disjoint across partitions within a
setting, and all models in a setting share the same partitions.

Note: the three cohort settings are independently sampled, so changes across
N also reflect test-set composition. They are descriptive settings, not a
learning curve. Pass ``nested=True`` to make smaller cohorts subsets of the
largest one if you need a nested comparison.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Sequence

COHORT_SIZES = (250, 500, 706)


def make_cohort_splits(
    patient_ids: Sequence[str],
    n: int,
    seed: int = 2025,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    nested: bool = False,
    base_order: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}``.

    ``nested=True`` requires ``base_order`` (the shuffled full cohort) and
    takes its first ``n`` patients, so smaller cohorts are subsets of larger
    ones. The default reproduces the independent sampling used in the paper.
    """
    if n > len(patient_ids):
        raise ValueError(f"n={n} exceeds available patients ({len(patient_ids)})")

    if nested:
        if base_order is None:
            raise ValueError("nested=True requires base_order")
        selected = list(base_order[:n])
    else:
        shuffled = list(patient_ids)
        random.Random(seed).shuffle(shuffled)
        selected = shuffled[:n]

    # Keep the paper's exact partition sizes (175/25/50, 350/50/100,
    # 494/70/142) by truncating and taking the remainder as test.
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = selected[:n_train]
    val = selected[n_train : n_train + n_val]
    test = selected[n_train + n_val : n_train + n_val + n_test]

    return {
        "n": n,
        "seed": seed,
        "nested": nested,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def partition_sizes(n: int) -> Dict[str, int]:
    """Partition sizes as reported in Table 2 of the paper."""
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    return {"train": n_train, "val": n_val, "test": n - n_train - n_val}


def save_splits(path: str, splits: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, ensure_ascii=False, indent=2)


def load_splits(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_split(splits: Dict, n: int, split: str) -> List[str]:
    """Read one partition from either a single-split dict or a dict of splits."""
    if "train" in splits:
        return splits[split]
    key = str(n)
    if key in splits:
        return splits[key][split]
    raise KeyError(f"no split for N={n}")
