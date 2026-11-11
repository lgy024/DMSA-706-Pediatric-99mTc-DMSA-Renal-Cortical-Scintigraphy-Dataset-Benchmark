"""Training loop for image-text contrastive pretraining."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from dmsa706.data.contrastive_dataset import ContrastiveCollator
from dmsa706.models.contrastive import info_nce_loss
from dmsa706.trainers.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from dmsa706.trainers.optim import build_optimizer
from dmsa706.trainers.schedulers import build_scheduler
from dmsa706.utils.checkpoint import save_checkpoint
from dmsa706.utils.logging import get_logger


class ContrastiveTrainer:
    def __init__(self, cfg, model, train_dataset, val_dataset=None):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = get_logger("dmsa706.contrastive")

    def _device(self):
        requested = self.cfg.train.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.train.get("num_workers", 0)),
            collate_fn=ContrastiveCollator(pad_id=0),
            pin_memory=torch.cuda.is_available(),
        )

    def _run_epoch(self, loader, optimizer=None, scheduler=None):
        training = optimizer is not None
        self.model.train(training)
        device = self._device()
        total, steps = 0.0, 0
        autocast = torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(self.cfg.train.get("bf16", True)) and device.type == "cuda",
        )
        for images, text_ids, mask in loader:
            images = images.to(device, non_blocking=True)
            text_ids = text_ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast:
                logits, _, _ = self.model(images, text_ids, mask)
                loss = info_nce_loss(logits)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg.train.max_grad_norm)
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += loss.item()
            steps += 1
        return total / max(steps, 1)

    def train(self):
        device = self._device()
        self.model.to(device)
        train_loader = self._loader(self.train_dataset, shuffle=True)
        val_loader = self._loader(self.val_dataset, shuffle=False) if self.val_dataset else None
        optimizer = build_optimizer(self.model, self.cfg)
        total_steps = max(int(self.cfg.train.epochs) * len(train_loader), 1)
        scheduler = build_scheduler(optimizer, self.cfg, total_steps)

        os.makedirs(self.cfg.train.save_dir, exist_ok=True)
        checkpoint = ModelCheckpoint(
            os.path.join(self.cfg.train.save_dir, "best.pt"), mode="min"
        )
        early_stopping = EarlyStopping(
            patience=int(self.cfg.train.get("patience", 5)), mode="min"
        )
        logger = CSVLogger(
            os.path.join(self.cfg.train.save_dir, "train_log.csv"),
            fields=["epoch", "train_loss", "val_loss", "lr"],
        )

        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            train_loss = self._run_epoch(train_loader, optimizer, scheduler)
            row = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
            message = f"epoch {epoch} train_loss={train_loss:.4f}"
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader)
                row["val_loss"] = val_loss
                message += f" val_loss={val_loss:.4f}"
                checkpoint(
                    val_loss,
                    self.model,
                    optimizer,
                    epoch=epoch,
                    extra={"vocab": getattr(self.train_dataset, "vocab", None)},
                )
                if early_stopping(val_loss):
                    self.logger.info("early stopping at epoch %d", epoch)
                    logger.log(row)
                    break
            else:
                row["val_loss"] = ""
            logger.log(row)
            self.logger.info(message)

        save_checkpoint(
            os.path.join(self.cfg.train.save_dir, "last.pt"),
            self.model,
            optimizer,
            epoch=int(self.cfg.train.epochs),
            extra={"vocab": getattr(self.train_dataset, "vocab", None)},
        )
        return self.model
