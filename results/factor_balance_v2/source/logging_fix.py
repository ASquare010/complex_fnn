"""Canonical byte hashing for H069; no model or balancing change."""

import hashlib

import torch


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    header = str((t.dtype, tuple(t.shape))).encode()
    raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest()
