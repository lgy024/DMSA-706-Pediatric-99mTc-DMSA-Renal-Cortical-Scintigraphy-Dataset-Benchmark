"""T2 classification head on a vision-language backbone."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from dmsa706.data.formatting import format_messages


class VLMClassifier(nn.Module):
    """Pool the backbone's last hidden state and classify the diagnosis."""

    def __init__(self, cfg, num_classes: int, bundle=None, freeze_backbone=None):
        super().__init__()
        if bundle is None:
            from .registry import build_model

            bundle = build_model(cfg)
        self.bundle = bundle
        self.model = bundle.model
        self.processor = bundle.processor
        self.prompt = cfg.model.get(
            "t2_prompt", "请判断该肾皮质显像图像最符合的诊断类别。"
        )
        hidden_size = self._hidden_size(self.model.config)
        self.head = nn.Linear(hidden_size, num_classes)

        if freeze_backbone is None:
            freeze_backbone = bool(cfg.model.get("freeze_backbone", True))
        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _hidden_size(config) -> int:
        hidden = getattr(config, "hidden_size", None)
        if hidden is None:
            text_config = getattr(config, "text_config", None)
            hidden = getattr(text_config, "hidden_size", 1024)
        return int(hidden)

    def _pooled(self, images):
        formatted = [
            format_messages(
                [{"role": "user", "content": self.prompt}],
                [image],
                self.bundle.image_style,
            )
            for image in images
        ]
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in formatted
        ]
        inputs = self.processor(
            text=texts, images=list(images), padding=True, return_tensors="pt"
        )
        device = self.head.weight.device
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs.get("attention_mask")
        if mask is not None:
            last = mask.sum(dim=1) - 1
        else:
            last = torch.full(
                (hidden.size(0),), hidden.size(1) - 1, device=hidden.device
            )
        index = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[index, last]

    def forward(self, images):
        return self.head(self._pooled(images))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "head": self.head.state_dict(),
                "num_classes": self.head.out_features,
                "prompt": self.prompt,
            },
            os.path.join(path, "head.pt"),
        )
        # A frozen backbone is large; save only the adapter when one is used.
        try:
            from peft import PeftModel

            if isinstance(self.model, PeftModel):
                self.model.save_pretrained(path)
        except ImportError:
            pass

    def load_head(self, path: str) -> None:
        checkpoint = torch.load(os.path.join(path, "head.pt"), map_location="cpu")
        self.head.load_state_dict(checkpoint["head"])
