"""Research-only approximate storage for whole-block checkpoint inputs."""

from contextlib import contextmanager
from types import MethodType

import torch

from src.core.transformer import Block, Transformer


@contextmanager
def compress_inputs(model, half=False, capture=False, allow_cpu=False):
    if type(model) is not Transformer:
        raise TypeError("Expected ordinary Transformer")
    blocks = list(model.blocks)
    if any(
        type(b) is not Block or b.recompute_scope != "block" or "forward" in b.__dict__
        for b in blocks
    ):
        raise ValueError("Expected original whole-block checkpoint forwards")
    records = []
    originals = []
    unpack_count = [0]

    def forward(this, x):
        if not (this.training and torch.is_grad_enabled() and (x.is_cuda or allow_cpu)):
            return Block.forward(this, x)
        if x.dtype != torch.float32 or x.ndim != 3:
            raise ValueError("Expected FP32 three-dimensional checkpoint input")
        # Capture primitives only. Closing over x would retain the original GPU
        # tensor and defeat compression, even after a successful FP16 cast.
        key = (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.storage_offset())
        index = len(records)
        record = dict(
            shape=list(x.shape),
            original_bytes=x.numel() * 4,
            stored_bytes=x.numel() * (2 if half else 4),
        )
        records.append(record)

        def pack(t):
            if t.numel() == 0:
                return t.detach(), t.dtype, False
            actual = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.storage_offset())
            if actual != key:
                raise RuntimeError("Unexpected saved tensor: refusing to compress it")
            if capture:
                originals.append(dict(index=index, input=t.detach().cpu()))
            return t.detach().to(torch.float16 if half else t.dtype), t.dtype, True

        def unpack(packet):
            payload, dtype, is_input = packet
            if is_input:
                unpack_count[0] += 1
            return payload.to(dtype)

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return Block.forward(this, x)

    installed = []
    try:
        for block in blocks:
            block.forward = MethodType(forward, block)
            installed.append(block)
        yield dict(records=records, originals=originals, unpack_count=unpack_count)
    finally:
        for block in installed:
            del block.forward
