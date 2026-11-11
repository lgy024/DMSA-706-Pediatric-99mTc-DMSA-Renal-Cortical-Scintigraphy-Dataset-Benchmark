"""LLaVA-1.5 backend."""

from __future__ import annotations

import torch

from dmsa706.data.formatting import format_messages

from .vlm_base import BaseVLM


class LlavaModel(BaseVLM):
    image_style = "llava"

    def __init__(self, cfg):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        dtype = torch.bfloat16 if bool(cfg.train.get("bf16", True)) else torch.float16
        self.processor = AutoProcessor.from_pretrained(cfg.model.pretrained)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            cfg.model.pretrained,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._maybe_lora()

    def generate(self, messages, images, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        formatted = format_messages(messages, images, self.image_style)
        prompt = self.processor.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        return self.processor.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
