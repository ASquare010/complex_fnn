"""Four tests for the corrected tensor artifact serializer."""

import hashlib
import struct

import torch

from results.factor_balance_v1.source.diagnosis import tensor_hash as old_hash
from results.factor_balance_v2.source.logging_fix import tensor_hash


def test_known_bfloat16_bytes():
    t = torch.tensor([1.0, -2.0, 0.0, -0.0], dtype=torch.bfloat16)
    expected = hashlib.sha256(
        str((t.dtype, tuple(t.shape))).encode() + struct.pack("<HHHH", 0x3F80, 0xC000, 0, 0x8000)
    ).hexdigest()
    assert tensor_hash(t) == expected
    assert tensor_hash(torch.tensor(0.0, dtype=torch.bfloat16)) != tensor_hash(
        torch.tensor(-0.0, dtype=torch.bfloat16)
    )


def test_legacy_hash_compatibility():
    for dtype in (torch.float32, torch.float64, torch.int64):
        t = torch.arange(12, dtype=dtype).reshape(3, 4)
        assert tensor_hash(t) == old_hash(t)
        assert tensor_hash(t.T) == old_hash(t.T)
        assert tensor_hash(t) != tensor_hash(t.flatten())


def test_scalar_empty_and_noncontiguous():
    for dtype in (torch.float32, torch.bfloat16):
        t = torch.tensor(1.0, dtype=dtype)
        assert tensor_hash(t) != tensor_hash(t.reshape(1))
        assert tensor_hash(torch.empty(0, dtype=dtype)) != tensor_hash(
            torch.empty(0, 1, dtype=dtype)
        )
        t = torch.arange(24, dtype=dtype).reshape(4, 6)[:, ::2]
        assert not t.is_contiguous()
        assert tensor_hash(t) == tensor_hash(t.contiguous())


def test_cuda_bfloat16_matches_cpu():
    assert torch.cuda.is_available()
    t = torch.randn(16, 128, 384, generator=torch.Generator().manual_seed(692)).bfloat16()
    assert tensor_hash(t) == tensor_hash(t.cuda())
