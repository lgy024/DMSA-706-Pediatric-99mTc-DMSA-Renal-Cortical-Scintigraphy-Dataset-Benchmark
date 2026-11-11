"""Shared helpers for the HuggingFace vision-language backends."""

from __future__ import annotations


class BaseVLM:
    image_style = "llava"

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def device(self):
        return next(self.model.parameters()).device

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        if hasattr(self, "processor"):
            self.processor.save_pretrained(path)
        if hasattr(self, "tokenizer"):
            self.tokenizer.save_pretrained(path)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _maybe_lora(self):
        lora = getattr(self.cfg.model, "lora", None)
        if lora is not None and bool(lora.get("enabled", False)):
            from .lora import apply_lora

            self.model = apply_lora(self.model, lora)

    @staticmethod
    def flatten_messages(messages):
        """Plain-text prompt used by backends with a string-only chat API."""
        lines = []
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
