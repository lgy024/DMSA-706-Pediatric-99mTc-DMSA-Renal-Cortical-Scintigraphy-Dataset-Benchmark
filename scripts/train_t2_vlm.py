#!/usr/bin/env python
"""Train a T2 classification head on a vision-language backbone.

Example:
    python scripts/train_t2_vlm.py --config configs/qwen25_vl.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import load_records  # noqa: E402
from dmsa706.splits import get_split, load_splits  # noqa: E402
from dmsa706.utils.config import load_config  # noqa: E402
from dmsa706.utils.logging import get_logger  # noqa: E402
from dmsa706.utils.seed import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="also fine-tune the backbone instead of training the head only",
    )
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
    image_size = int(cfg.data.get("image_size", 512))
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    from dmsa706.data import T2ImageDataset
    from dmsa706.labels import CATEGORY_KEYS
    from dmsa706.models.vlm_classifier import VLMClassifier
    from dmsa706.trainers import VLMClassificationTrainer

    train_ds = T2ImageDataset(records, train_ids, image_size=image_size)
    val_ds = T2ImageDataset(records, val_ids, image_size=image_size)
    logger = get_logger("train_t2_vlm")
    logger.info("train=%d val=%d classes=%d", len(train_ds), len(val_ds), len(CATEGORY_KEYS))

    model = VLMClassifier(
        cfg,
        len(CATEGORY_KEYS),
        freeze_backbone=not args.unfreeze,
    )
    VLMClassificationTrainer(cfg, model, train_ds, val_ds).train()


if __name__ == "__main__":
    main()
