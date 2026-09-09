"""Prevent silent data leakage and shifted-target/evaluation mistakes."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.core.data import TokenData, load_manifest, normalized_hash
from src.core.reproducibility import sha256


def cache(tmp_path: Path) -> Path:
    for split in ("train", "valid"):
        np.save(tmp_path / f"{split}.npy", np.arange(101, dtype=np.uint16))
    manifest = {"files": {p.name: sha256(p) for p in tmp_path.iterdir()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_target_shift_and_sampling_reproducibility(tmp_path):
    path = cache(tmp_path)
    a, b = TokenData(path, "cpu", 13), TokenData(path, "cpu", 13)
    for _ in range(3):
        x, y = a.batch(4, 8)
        other_x, other_y = b.batch(4, 8)
        torch.testing.assert_close(x, other_x)
        torch.testing.assert_close(y, other_y)
        torch.testing.assert_close(y, x + 1)


def test_validation_targets_are_unique_and_repeatable(tmp_path):
    data = TokenData(cache(tmp_path), "cpu", 13)
    first = torch.cat([y.flatten() for _, y in data.validation(3, 8, 4)])
    second = torch.cat([y.flatten() for _, y in data.validation(3, 8, 4)])
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, torch.arange(1, 97))
    assert len(first.unique()) == len(first)


def test_cache_tampering_detected(tmp_path):
    path = cache(tmp_path)
    (path / "train.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(path)


def test_whitespace_normalization():
    assert normalized_hash("a  b\nc") == normalized_hash("a b c")
