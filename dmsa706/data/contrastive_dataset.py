"""Image-text pairs for contrastive pretraining."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from dmsa706.qa import format_report_draft

from .transforms import build_eval_transform, build_train_transform


def build_vocab(texts: Sequence[str], min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(list(text or ""))
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for token, count in counter.most_common():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 128) -> List[int]:
    ids = [vocab["<sos>"]]
    ids += [vocab.get(char, vocab["<unk>"]) for char in (text or "")]
    ids.append(vocab["<eos>"])
    return ids[:max_len]


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict],
        patient_ids: Sequence[str],
        vocab: Dict[str, int],
        image_size: int = 256,
        max_len: int = 128,
        train: bool = True,
    ) -> None:
        index = {record["patient_id"]: record for record in records}
        self.items = []
        for pid in patient_ids:
            record = index.get(pid)
            if record is None or not record.get("images"):
                continue
            text = format_report_draft(record.get("qa", [])).strip()
            if text:
                self.items.append((record["images"][0], text))
        self.vocab = vocab
        self.max_len = max_len
        self.transform = (
            build_train_transform(image_size) if train else build_eval_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, text = self.items[index]
        image = Image.open(path).convert("L")
        image = self.transform(image)
        ids = encode_text(text, self.vocab, self.max_len)
        return image, torch.tensor(ids, dtype=torch.long)


class ContrastiveCollator:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch):
        images, texts = zip(*batch)
        max_len = max(len(text) for text in texts)
        padded = torch.full((len(texts), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(texts), max_len), dtype=torch.long)
        for index, text in enumerate(texts):
            padded[index, : len(text)] = text
            mask[index, : len(text)] = 1
        return torch.stack(images), padded, mask
