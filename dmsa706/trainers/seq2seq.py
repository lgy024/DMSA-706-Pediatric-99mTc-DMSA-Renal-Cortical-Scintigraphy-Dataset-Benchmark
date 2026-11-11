"""Training loop for CNN+LSTM and ViT+Transformer."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dmsa706.data.collator import Seq2SeqCollator
from dmsa706.utils.checkpoint import save_checkpoint
from dmsa706.utils.logging import get_logger


class Seq2SeqTrainer:
    def __init__(self, cfg, model, train_dataset, val_dataset=None, pad_id: int = 0):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.pad_id = pad_id
        self.logger = get_logger("dmsa706.seq2seq")

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.train.get("num_workers", 0)),
            collate_fn=Seq2SeqCollator(self.pad_id),
            pin_memory=torch.cuda.is_available(),
        )

    def _device(self):
        requested = self.cfg.train.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _run_epoch(self, loader, optimizer=None, scheduler=None):
        training = optimizer is not None
        self.model.train(training)
        total_loss, steps = 0.0, 0
        device = self._device()
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(self.cfg.train.get("bf16", True)) and device.type == "cuda")

        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inputs, outputs = targets[:, :-1], targets[:, 1:]
            with autocast:
                logits = self.model(images, inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    outputs.reshape(-1),
                    ignore_index=self.pad_id,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg.train.max_grad_norm)
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item()
            steps += 1
        return total_loss / max(steps, 1)

    def train(self):
        device = self._device()
        self.model.to(device)
        train_loader = self._loader(self.train_dataset, shuffle=True)
        val_loader = self._loader(self.val_dataset, shuffle=False) if self.val_dataset else None

        from dmsa706.trainers.optim import build_optimizer
        from dmsa706.trainers.schedulers import build_scheduler

        optimizer = build_optimizer(self.model, self.cfg)
        epochs = int(self.cfg.train.epochs)
        total_steps = max(epochs * len(train_loader), 1)
        scheduler = build_scheduler(optimizer, self.cfg, total_steps)

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer, scheduler)
            message = f"epoch {epoch}/{epochs} train_loss={train_loss:.4f}"
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader)
                message += f" val_loss={val_loss:.4f}"
            self.logger.info(message)

            save_checkpoint(
                os.path.join(self.cfg.train.save_dir, f"epoch_{epoch}.pt"),
                self.model,
                optimizer,
                scheduler,
                epoch,
                extra={"vocab": getattr(self.train_dataset, "vocab", None)},
            )

        save_checkpoint(
            os.path.join(self.cfg.train.save_dir, "last.pt"),
            self.model,
            optimizer,
            scheduler,
            epochs,
            extra={"vocab": getattr(self.train_dataset, "vocab", None)},
        )
        self.logger.info("saved checkpoints to %s", self.cfg.train.save_dir)
        return self.model
