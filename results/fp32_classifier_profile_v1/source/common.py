"""One loss adapter and accounting helpers shared by all H114 executions."""

import gc
import hashlib
import json

import torch
from torch.nn import functional as F

from src.core.config import ModelConfig
from src.core.token_memory import chunked_linear_cross_entropy
from src.core.transformer import Transformer

POLICIES = ("block_bf16", "chunks_bf16", "block_fp32", "chunks_fp32")


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def tree_description(value):
    if isinstance(value, torch.Tensor):
        return dict(sha256=tensor_hash(value), shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {str(k): tree_description(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_description(v) for v in value]
    return value


def tree_hash(value):
    return hashlib.sha256(json.dumps(tree_description(value), sort_keys=True).encode()).hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def set_blocks(model):
    for block in model.blocks:
        block.recompute_scope = "block"


def training_loss(model, tokens, targets, policy, chunk=512):
    assert policy in POLICIES
    hidden = model.embedding(tokens)
    for block in model.blocks:
        hidden = block(hidden)
    hidden = model.norm(hidden)
    # This subregion changes only the classifier; checkpoint captures its context.
    with torch.autocast(
        hidden.device.type,
        dtype=torch.bfloat16,
        enabled=hidden.is_cuda and policy.endswith("bf16"),
    ):
        if policy.endswith("fp32") and hidden.dtype != torch.float64:
            hidden = hidden.float()
        if policy.startswith("chunks"):
            return chunked_linear_cross_entropy(hidden, model.embedding.weight, targets, chunk)
        logits = F.linear(hidden, model.embedding.weight)
        if logits.dtype in (torch.bfloat16, torch.float16):
            logits = logits.float()
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def clear_boundary():
    gc.collect()
    torch.clear_autocast_cache()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    value = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
    assert value == dict(allocated=0, reserved=0), value
    torch.cuda.reset_peak_memory_stats()
    return value


class MemoryLedger:
    """Record every interval before any peak reset, including untimed phases."""

    def __init__(self):
        self.records = []

    def mark(self, phase):
        torch.cuda.synchronize()
        row = dict(
            phase=phase,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            live_allocated_bytes=torch.cuda.memory_allocated(),
            live_reserved_bytes=torch.cuda.memory_reserved(),
        )
        self.records.append(row)
        torch.cuda.reset_peak_memory_stats()
        return row


def gradient_error(actual, reference):
    per_tensor, square_error, square_reference = {}, 0.0, 0.0
    for name in reference:
        a, b = actual[name].double(), reference[name].double()
        difference, scale = (a - b).norm().item(), b.norm().item()
        per_tensor[name] = difference / max(scale, 1e-8)
        square_error += difference**2
        square_reference += scale**2
    return dict(
        global_relative_l2=square_error**0.5 / max(square_reference**0.5, 1e-12),
        per_tensor_relative_l2=per_tensor,
        max_tensor_relative_l2=max(per_tensor.values()),
    )


def qualify():
    checks = []
    for variant in ("gelu", "swiglu"):
        for shape in ((2, 7), (1, 11)):
            cfg = ModelConfig(
                variant=variant,
                vocab_size=32,
                width=24,
                hidden=48,
                layers=2,
                heads=3,
                context=shape[1],
            )
            tokens = torch.arange(shape[0] * shape[1]).reshape(shape) % 31
            targets = tokens + 1
            targets.reshape(-1)[::3] = -100
            reference = Transformer(cfg, 114).double()
            native = F.cross_entropy(reference(tokens).flatten(0, 1), targets.flatten())
            native.backward()
            expected = {k: p.grad for k, p in reference.named_parameters()}
            for policy in POLICIES:
                model = Transformer(cfg, 114).double()
                set_blocks(model)
                before = {k: id(p) for k, p in model.named_parameters()}
                loss = training_loss(model, tokens, targets, policy, 5)
                loss.backward()
                torch.testing.assert_close(loss, native, atol=1e-10, rtol=1e-10)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad, expected[name], atol=1e-10, rtol=1e-10
                    )
                assert before == {k: id(p) for k, p in model.named_parameters()}
                checks.append(dict(variant=variant, shape=shape, policy=policy, passed=True))
    return dict(passed=True, double_model_checks=checks, optimizer_updates=0)
