"""Batch collators for VLM and encoder-decoder training."""

from __future__ import annotations

from typing import Sequence

import torch

from .formatting import format_messages


class VLMDataCollator:
    """Tokenise conversations and mask non-assistant tokens."""

    def __init__(self, processor, image_style: str = "llava", max_length: int = 2048):
        self.processor = processor
        self.image_style = image_style
        self.max_length = max_length

    def _assistant_mask(self, messages):
        try:
            encoded = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
            mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
            if mask is not None:
                return torch.tensor(mask, dtype=torch.bool)
        except (AttributeError, TypeError, ValueError, KeyError):
            pass
        return None

    def __call__(self, batch: Sequence[dict]) -> dict:
        formatted = [
            format_messages(item["messages"], item["images"], self.image_style)
            for item in batch
        ]
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in formatted
        ]
        images = [image for item in batch for image in item["images"]]

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        padding_side = getattr(tokenizer, "padding_side", "right")
        for index, messages in enumerate(formatted):
            mask = self._assistant_mask(messages)
            if mask is not None:
                pad = labels.shape[1] - mask.numel()
                if pad > 0:
                    zeros = torch.zeros(pad, dtype=torch.bool)
                    if padding_side == "left":
                        mask = torch.cat([zeros, mask])
                    else:
                        mask = torch.cat([mask, zeros])
                elif pad < 0:
                    mask = mask[: labels.shape[1]]
                labels[index, ~mask] = -100
            else:
                # Fallback: supervise the final assistant answer only.
                prompt = self.processor.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
                labels[index, : len(prompt_ids)] = -100

        inputs["labels"] = labels
        return inputs


class T2ImageCollator:
    """Keep images as PIL objects for the vision-language classifier."""

    def __call__(self, batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels, dtype=torch.long)


class Seq2SeqCollator:
    """Pad target token sequences for the encoder-decoder baselines."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        images, targets = zip(*batch)
        max_len = max(len(target) for target in targets)
        padded = torch.full((len(targets), max_len), self.pad_id, dtype=torch.long)
        for index, target in enumerate(targets):
            padded[index, : len(target)] = target
        return torch.stack(images), padded
