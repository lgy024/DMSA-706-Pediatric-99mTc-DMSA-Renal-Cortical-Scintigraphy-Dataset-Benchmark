"""OpenAI-compatible client for vision-language endpoints.

Works with vLLM, SGLang, LM Studio, Ollama's OpenAI shim, and hosted APIs that
accept the ``image_url`` content type. The client is model-agnostic: pass the
served model name, e.g. ``Qwen/Qwen2.5-VL-7B-Instruct`` or an InternVL
checkpoint served by vLLM.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from typing import Dict, List, Optional, Sequence

PathLike = str


def encode_image_data_url(path: PathLike) -> str:
    """Return a ``data:image/...;base64,...`` URL for an image file."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(ext, "image/jpeg")
    with open(path, "rb") as fh:
        payload = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class VLMClient:
    """Thin retrying wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "openai is required for the endpoint client: pip install openai"
            ) from exc

        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - retry all endpoint errors
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"chat request failed: {last_error}")

    def answer(
        self,
        question: str,
        image_paths: Sequence[PathLike],
        history: Optional[Sequence[Dict[str, str]]] = None,
        max_tokens: int = 512,
    ) -> str:
        """Convenience helper for a single T1-style question."""
        from .prompts import _image_part, _text_part

        messages: List[Dict] = []
        for item in history or []:
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item.get("answer", "")})
        content = [_text_part(question)] + [
            _image_part(encode_image_data_url(p)) for p in image_paths
        ]
        messages.append({"role": "user", "content": content})
        return self.chat(messages, max_tokens=max_tokens)
