"""Prepared data loader. TokenBatch makes tensor shapes explicit at the boundary."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from storage import digest, read_json


@dataclass(frozen=True)
class TokenBatch:
    """inputs/targets: int64 [batch, context]; targets=-100 masks prompt/padding."""

    inputs: torch.Tensor
    targets: torch.Tensor

    def to(self, device: str):
        return TokenBatch(self.inputs.to(device), self.targets.to(device))


class Dataset:
    def __init__(self, path, context):
        self.path, self.context = Path(path), context
        self.manifest = read_json(self.path / "manifest.json")
        self.fingerprint = digest(self.path / "manifest.json")
        for name, expected in self.manifest["files"].items():
            file = (self.path / name).resolve()
            if file.parent != self.path.resolve() or digest(file) != expected:
                raise ValueError(f"Dataset file changed or unsafe: {name}")
        self.kind = self.manifest["kind"]
        self.vocab_size = self.manifest["vocab_size"]
        self.eos_id = self.manifest["eos_id"]
        self.splits = {}
        for split in self.manifest["splits"]:
            if self.kind == "language":
                data = np.load(self.path / f"{split}.npy", mmap_mode="r", allow_pickle=False)
                if len(data) < context + 1:
                    raise ValueError(f"{split} is too small for context {context}")
                if int(data.min()) < 0 or int(data.max()) >= self.manifest["vocab_size"]:
                    raise ValueError("Token outside vocabulary")
            else:
                data = read_json(self.path / f"{split}.json")
                if max(len(r["prompt_ids"]) + len(r["answer_ids"]) - 1 for r in data) > context:
                    raise ValueError(f"{split} exceeds context; do not silently truncate reasoning")
            self.splits[split] = data

    def batch(
        self, split: str, batch_size: int, generator: torch.Generator | None = None, offset: int = 0
    ) -> TokenBatch | None:
        data = self.splits[split]
        if self.kind == "language":
            available = (len(data) - 1) // self.context
            if generator is None and offset >= available:
                return None
            starts = (
                torch.randint(len(data) - self.context, (batch_size,), generator=generator)
                if generator is not None
                else torch.arange(offset, min(offset + batch_size, available)) * self.context
            )
            if not len(starts):
                return None
            rows = np.stack([data[int(i) : int(i) + self.context + 1] for i in starts])
            rows = torch.from_numpy(rows.astype(np.int64))
            return TokenBatch(rows[:, :-1], rows[:, 1:])
        indices = (
            torch.randint(len(data), (batch_size,), generator=generator).tolist()
            if generator is not None
            else list(range(offset, min(offset + batch_size, len(data))))
        )
        if not indices:
            return None
        x = torch.ones((len(indices), self.context), dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, index in enumerate(indices):
            row = data[index]
            ids = row["prompt_ids"] + row["answer_ids"]
            x[i, : len(ids) - 1] = torch.tensor(ids[:-1])
            start = len(row["prompt_ids"]) - 1
            y[i, start : len(ids) - 1] = torch.tensor(ids[start + 1 :])
        return TokenBatch(x, y)
