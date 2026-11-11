"""T1 datasets: five-turn conversation and single-sequence baselines."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dmsa706.preprocessing import preprocess_image
from dmsa706.prompts import SYSTEM_PROMPT
from dmsa706.qa import format_report_draft

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


def build_messages(record: dict, system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """Turn one record's five QA pairs into a chat conversation."""
    messages = [{"role": "system", "content": system_prompt}]
    for item in record.get("qa", []):
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": item["answer"]})
    return messages


class T1ConversationDataset(Dataset):
    """Returns PIL images and a neutral conversation for a VLM processor."""

    def __init__(
        self,
        records: Sequence[dict],
        patient_ids: Sequence[str],
        image_size: int = 512,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        index = {record["patient_id"]: record for record in records}
        self.items = [index[pid] for pid in patient_ids if pid in index]
        self.image_size = image_size
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        record = self.items[index]
        images = []
        for path in record["images"]:
            image = preprocess_image(path, max_edge=self.image_size)
            images.append(image.convert("RGB"))
        return {
            "patient_id": record["patient_id"],
            "images": images,
            "messages": build_messages(record, self.system_prompt),
        }


def build_vocab(texts: Sequence[str], min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(list(text or ""))
    vocab = {token: index for index, token in enumerate(SPECIALS)}
    for token, count in counter.most_common():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 256) -> List[int]:
    ids = [vocab[SOS]]
    ids += [vocab.get(char, vocab[UNK]) for char in (text or "")]
    ids.append(vocab[EOS])
    return ids[:max_len]


class T1Seq2SeqDataset(Dataset):
    """Image-to-report sequence dataset for CNN+LSTM and ViT+Transformer."""

    def __init__(
        self,
        records: Sequence[dict],
        patient_ids: Sequence[str],
        vocab: Dict[str, int],
        image_size: int = 256,
        max_len: int = 256,
    ) -> None:
        index = {record["patient_id"]: record for record in records}
        self.items = []
        for pid in patient_ids:
            record = index.get(pid)
            if record is None or not record.get("images"):
                continue
            target = format_report_draft(record.get("qa", [])).strip()
            if target:
                self.items.append((record["images"][0], target))
        self.vocab = vocab
        self.image_size = image_size
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, target = self.items[index]
        image = preprocess_image(path, max_edge=self.image_size, as_array=True)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)
        ids = encode_text(target, self.vocab, self.max_len)
        return image, torch.tensor(ids, dtype=torch.long)
