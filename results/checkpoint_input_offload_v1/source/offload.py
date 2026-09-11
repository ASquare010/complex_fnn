"""Existing block checkpointing, optionally surrounded by native CPU save hooks."""

import time
import types
import weakref

import torch

from results.fp32_classifier_profile_v1.source.common import tensor_hash
from src.core.transformer import Block


class Trace:
    """Observe packed-object lifetimes without holding GPU inputs or packed values."""

    def __init__(self, offloaded: bool):
        self.offloaded = offloaded
        self.api = torch.autograd.graph.save_on_cpu(pin_memory=True) if offloaded else None
        self.rows, self.events, self.timers = [], [], []
        self.live_bytes = self.peak_bytes = 0
        self.audit_d2h_bytes = 0

    def event(self, kind, index):
        self.events.append(
            dict(kind=kind, index=index, at_ns=time.perf_counter_ns(), live_bytes=self.live_bytes)
        )

    def release(self, index, size):
        self.live_bytes -= size
        self.event("release", index)

    def context(self, block):
        def pack(tensor):
            size = tensor.numel() * tensor.element_size()
            index = len(self.rows)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            wall = time.perf_counter()
            start.record()
            packed = (
                self.api.pack_hook(tensor) if self.offloaded else (tensor.device, tensor.detach())
            )
            end.record()
            host_wall_ms = 1000 * (time.perf_counter() - wall)
            device, value = packed
            torch.cuda.synchronize()  # CPU pinned data must be complete before hashing.
            expected = tensor_hash(tensor)
            self.audit_d2h_bytes += size if tensor.is_cuda else 0
            if self.offloaded:
                assert tensor_hash(value) == expected
            self.live_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            self.rows.append(
                dict(
                    block=block,
                    index=index,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    source_device=str(tensor.device),
                    packed_device=str(value.device),
                    pinned=value.is_pinned(),
                    bytes=size,
                    source_hash=expected,
                    unpack_hash=None,
                    unpack_count=0,
                    pack_host_wall_ms=host_wall_ms,
                )
            )
            self.timers.append((index, "pack", start, end))
            self.event("pack", index)
            weakref.finalize(value, self.release, index, size)
            return device, value, index

        def unpack(packed):
            device, value, index = packed
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            wall = time.perf_counter()
            start.record()
            tensor = self.api.unpack_hook((device, value)) if self.offloaded else value
            end.record()
            self.rows[index]["unpack_host_wall_ms"] = 1000 * (time.perf_counter() - wall)
            torch.cuda.synchronize()
            digest = tensor_hash(tensor)
            self.audit_d2h_bytes += self.rows[index]["bytes"] if tensor.is_cuda else 0
            assert digest == self.rows[index]["source_hash"]
            self.rows[index]["unpack_hash"] = digest
            self.rows[index]["unpack_count"] += 1
            self.timers.append((index, "unpack", start, end))
            self.event("unpack", index)
            return tensor

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def result(self):
        torch.cuda.synchronize()
        for index, kind, start, end in self.timers:
            self.rows[index][kind + "_event_ms"] = start.elapsed_time(end)
        return dict(
            rows=self.rows,
            events=self.events,
            live_payload_bytes=self.live_bytes,
            peak_payload_bytes=self.peak_bytes,
            audit_extra_d2h_bytes=self.audit_d2h_bytes,
            offloaded=self.offloaded,
        )


def install(model, offloaded: bool, trace: Trace | None = None):
    """Return a restorer; parameter identities and registered model paths do not change."""
    identifiers = {k: id(p) for k, p in model.named_parameters()}
    wrapped = []
    for index, block in enumerate(model.blocks):
        assert "forward" not in block.__dict__
        if not offloaded and trace is None:
            continue

        def forward(this, x, block_index=index):
            if this.training and torch.is_grad_enabled() and this.recompute_scope == "block":
                context = (
                    trace.context(block_index)
                    if trace is not None
                    else torch.autograd.graph.save_on_cpu(pin_memory=True)
                )
                with context:
                    return Block.forward(this, x)
            return Block.forward(this, x)

        block.forward = types.MethodType(forward, block)
        wrapped.append(block)
    assert identifiers == {k: id(p) for k, p in model.named_parameters()}

    def restore():
        for block in wrapped:
            del block.forward
        assert identifiers == {k: id(p) for k, p in model.named_parameters()}

    return restore
