"""T2 diagnostic-category dataset for the encoder baselines."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dmsa706.labels import CATEGORY_KEYS
from dmsa706.preprocessing import preprocess_image

LABEL_TO_ID: Dict[str, int] = {key: index for index, key in enumerate(CATEGORY_KEYS)}
ID_TO_LABEL: Dict[int, str] = {index: key for key, index in LABEL_TO_ID.items()}


class T2Dataset(Dataset):
    """Image classification dataset; skips records without a diagnosis."""

    def __init__(
        self,
        records: Sequence[dict],
        patient_ids: Sequence[str],
        image_size: int = 256,
    ) -> None:
        index = {record["patient_id"]: record for record in records}
        self.items = []
        for pid in patient_ids:
            record = index.get(pid)
            if record is None or not record.get("images"):
                continue
            label = record.get("diagnosis")
            if label not in LABEL_TO_ID:
                continue
            self.items.append((record["images"][0], LABEL_TO_ID[label]))
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = preprocess_image(path, max_edge=self.image_size, as_array=True)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)
        return image, torch.tensor(label, dtype=torch.long)


class T2ImageDataset(Dataset):
    """Returns PIL images for the vision-language classification head."""

    def __init__(
        self,
        records: Sequence[dict],
        patient_ids: Sequence[str],
        image_size: int = 512,
    ) -> None:
        index = {record["patient_id"]: record for record in records}
        self.items = []
        for pid in patient_ids:
            record = index.get(pid)
            if record is None or not record.get("images"):
                continue
            label = record.get("diagnosis")
            if label not in LABEL_TO_ID:
                continue
            self.items.append((record["images"][0], LABEL_TO_ID[label]))
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = preprocess_image(path, max_edge=self.image_size).convert("RGB")
        return image, label
