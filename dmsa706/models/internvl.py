"""InternVL2.5 backend."""

from __future__ import annotations

import torch
from torchvision import transforms

from .vlm_base import BaseVLM


class InternVLModel(BaseVLM):
    image_style = "internvl"

    def __init__(self, cfg):
        from transformers import AutoModel, AutoTokenizer

        dtype = torch.bfloat16 if bool(cfg.train.get("bf16", True)) else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model.pretrained,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._maybe_lora()

    def _pixel_values(self, images):
        tensors = [self.transform(image.convert("RGB")) for image in images]
        return torch.stack(tensors).to(device=self.device, dtype=self.model.dtype)

    def generate(self, messages, images, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        question = self.flatten_messages(messages)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = temperature
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                self._pixel_values(images),
                question,
                generation_config,
            )
        return str(response).strip()
