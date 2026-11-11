"""Qwen2.5-VL backend."""

from __future__ import annotations

import torch

from dmsa706.data.formatting import format_messages

from .vlm_base import BaseVLM


class Qwen25VLModel(BaseVLM):
    image_style = "qwen"

    def __init__(self, cfg):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if bool(cfg.train.get("bf16", True)) else torch.float16
        self.processor = AutoProcessor.from_pretrained(cfg.model.pretrained)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model.pretrained,
            torch_dtype=dtype,
            device_map="auto",
        )
        self._maybe_lora()

    def _prepare_inputs(self, messages, images):
        formatted = format_messages(messages, images, self.image_style)
        text = self.processor.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(formatted)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            inputs = self.processor(
                text=[text], images=images, padding=True, return_tensors="pt"
            )
        return inputs.to(self.device)

    def generate(self, messages, images, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        inputs = self._prepare_inputs(messages, images)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
