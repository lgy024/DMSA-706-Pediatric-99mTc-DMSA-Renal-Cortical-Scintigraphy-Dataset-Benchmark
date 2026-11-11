"""HuggingFace Trainer wrapper for the vision-language backends."""

from __future__ import annotations

from dmsa706.data.collator import VLMDataCollator
from dmsa706.utils.logging import get_logger


class VLMTrainer:
    def __init__(self, cfg, model_bundle, train_dataset, val_dataset=None):
        self.cfg = cfg
        self.bundle = model_bundle
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = get_logger("dmsa706.vlm")

    def train(self):
        from transformers import Trainer, TrainingArguments

        if hasattr(self.bundle.model, "config"):
            self.bundle.model.config.use_cache = False
        if bool(self.cfg.train.get("gradient_checkpointing", True)) and hasattr(
            self.bundle.model, "enable_input_require_grads"
        ):
            self.bundle.model.enable_input_require_grads()

        common = dict(
            output_dir=self.cfg.train.save_dir,
            num_train_epochs=float(self.cfg.train.epochs),
            per_device_train_batch_size=int(self.cfg.train.batch_size),
            per_device_eval_batch_size=int(self.cfg.eval.get("batch_size", 1)),
            gradient_accumulation_steps=int(self.cfg.train.grad_accum),
            learning_rate=float(self.cfg.train.lr),
            warmup_ratio=float(self.cfg.train.warmup_ratio),
            weight_decay=float(self.cfg.train.weight_decay),
            max_grad_norm=float(self.cfg.train.max_grad_norm),
            bf16=bool(self.cfg.train.get("bf16", True)),
            logging_steps=int(self.cfg.train.get("logging_steps", 10)),
            save_strategy="epoch",
            report_to=["wandb"] if bool(self.cfg.train.get("wandb", False)) else [],
            remove_unused_columns=False,
            dataloader_num_workers=int(self.cfg.train.get("num_workers", 4)),
            gradient_checkpointing=bool(self.cfg.train.get("gradient_checkpointing", True)),
            seed=int(self.cfg.get("seed", 2025)),
        )
        try:
            args = TrainingArguments(
                **common,
                eval_strategy="epoch" if self.val_dataset is not None else "no",
            )
        except TypeError:
            # transformers < 4.46
            args = TrainingArguments(
                **common,
                evaluation_strategy="epoch" if self.val_dataset is not None else "no",
            )

        collator = VLMDataCollator(
            self.bundle.processor,
            image_style=self.bundle.image_style,
            max_length=int(self.cfg.train.get("max_len", 2048)),
        )
        trainer = Trainer(
            model=self.bundle.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(self.cfg.train.save_dir)
        self.bundle.save(self.cfg.train.save_dir)
        self.logger.info("saved adapter/checkpoint to %s", self.cfg.train.save_dir)
        return self.bundle
