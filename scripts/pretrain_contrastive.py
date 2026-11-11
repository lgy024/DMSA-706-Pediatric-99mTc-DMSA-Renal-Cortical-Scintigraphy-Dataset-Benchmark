#!/usr/bin/env python
"""Image-text contrastive pretraining on DMSA-706 reports."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import load_records  # noqa: E402
from dmsa706.qa import format_report_draft  # noqa: E402
from dmsa706.splits import get_split, load_splits  # noqa: E402
from dmsa706.utils.config import load_config  # noqa: E402
from dmsa706.utils.seed import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.n is not None:
        cfg.data.n = args.n
    if args.seed is not None:
        cfg.seed = args.seed
    if args.save_dir is not None:
        cfg.train.save_dir = args.save_dir
    set_seed(int(cfg.get("seed", 2025)))

    records = load_records(cfg.data.records)
    splits = load_splits(cfg.data.splits)
    n = int(cfg.data.get("n", 706))
    train_ids = get_split(splits, n, "train")
    val_ids = get_split(splits, n, "val")
    image_size = int(cfg.data.get("image_size", 256))

    from dmsa706.data import ContrastiveDataset, build_vocab
    from dmsa706.models.contrastive import ImageTextContrastiveModel
    from dmsa706.trainers import ContrastiveTrainer

    train_set = set(train_ids)
    texts = [
        format_report_draft(record.get("qa", []))
        for record in records
        if record["patient_id"] in train_set
    ]
    vocab = build_vocab(texts)
    train_ds = ContrastiveDataset(
        records, train_ids, vocab, image_size=image_size, train=True
    )
    val_ds = ContrastiveDataset(
        records, val_ids, vocab, image_size=image_size, train=False
    )
    model = ImageTextContrastiveModel(cfg, len(vocab))
    ContrastiveTrainer(cfg, model, train_ds, val_ds).train()


if __name__ == "__main__":
    main()
