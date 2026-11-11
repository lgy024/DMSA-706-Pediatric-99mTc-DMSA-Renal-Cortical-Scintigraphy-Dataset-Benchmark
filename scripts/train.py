#!/usr/bin/env python
"""Train one DMSA-706 configuration.

Examples:
    python scripts/train.py --config configs/qwen25_vl.yaml --task t1
    python scripts/train.py --config configs/cnn_lstm.yaml --task t1
    python scripts/train.py --config configs/vit_transformer.yaml --task t2
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
    parser.add_argument("--task", default="t1", choices=["t1", "t2"])
    parser.add_argument("--n", type=int, default=None, help="cohort size")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.n is not None:
        cfg.data.n = args.n
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.train.device = args.device
    if args.save_dir is not None:
        cfg.train.save_dir = args.save_dir

    set_seed(int(cfg.get("seed", 2025)))
    logger = get_logger("train")
    records = load_records(cfg.data.records)
    splits = load_splits(cfg.data.splits)
    n = int(cfg.data.get("n", 706))
    train_ids = get_split(splits, n, "train")
    val_ids = get_split(splits, n, "val")
    image_size = int(cfg.data.get("image_size", 512))
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    logger.info(
        "model=%s task=%s n=%d train=%d val=%d",
        cfg.model.name,
        args.task,
        n,
        len(train_ids),
        len(val_ids),
    )

    from dmsa706.data import (  # imported here so --help works without torch
        T1ConversationDataset,
        T1Seq2SeqDataset,
        T2Dataset,
        build_vocab,
        build_weighted_sampler,
        labels_from_dataset,
    )
    from dmsa706.models.registry import VLM_NAMES, build_model
    from dmsa706.models.seq2seq import build_classifier
    from dmsa706.qa import format_report_draft
    from dmsa706.trainers import (
        ClassificationTrainer,
        Seq2SeqTrainer,
        VLMTrainer,
    )

    if args.task == "t2":
        if cfg.model.name in VLM_NAMES:
            raise SystemExit(
                "T2 for vision-language models is prompt-based; "
                "run scripts/infer.py --task t2 instead."
            )
        from dmsa706.labels import CATEGORY_KEYS

        train_ds = T2Dataset(records, train_ids, image_size=image_size)
        val_ds = T2Dataset(records, val_ids, image_size=image_size)
        model = build_classifier(cfg.model.name, len(CATEGORY_KEYS), cfg)
        sampler = None
        if bool(cfg.data.get("balanced", False)):
            sampler = build_weighted_sampler(
                labels_from_dataset(train_ds), seed=int(cfg.get("seed", 2025))
            )
        ClassificationTrainer(cfg, model, train_ds, val_ds, sampler=sampler).train()
        return

    if cfg.model.name in VLM_NAMES:
        train_ds = T1ConversationDataset(records, train_ids, image_size=image_size)
        val_ds = T1ConversationDataset(records, val_ids, image_size=image_size)
        bundle = build_model(cfg)
        VLMTrainer(cfg, bundle, train_ds, val_ds).train()
        return

    train_set = set(train_ids)
    texts = [
        format_report_draft(record.get("qa", []))
        for record in records
        if record["patient_id"] in train_set
    ]
    vocab = build_vocab(texts)
    seq_len = int(cfg.train.get("max_len", 256))
    train_ds = T1Seq2SeqDataset(
        records, train_ids, vocab, image_size=256, max_len=seq_len
    )
    val_ds = T1Seq2SeqDataset(
        records, val_ids, vocab, image_size=256, max_len=seq_len
    )
    model = build_model(cfg, vocab_size=len(vocab))
    Seq2SeqTrainer(cfg, model, train_ds, val_ds, pad_id=vocab["<pad>"]).train()


if __name__ == "__main__":
    main()
