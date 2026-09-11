"""Controlled classifier policies, graph inspection and double qualification."""

import gc
import hashlib

import torch
from torch.nn import functional as F

from src.core.token_memory import chunked_linear_cross_entropy

POLICIES = ("native_bf16", "chunks_cached", "chunks_uncached", "native_fp32", "chunks_fp32")


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def clear():
    gc.collect()
    torch.clear_autocast_cache()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0


def loss_for(hidden, weight, targets, policy, chunk=512):
    assert policy in POLICIES
    use_amp = (
        hidden.is_cuda
        and hidden.dtype != torch.float64
        and policy not in ("native_fp32", "chunks_fp32")
    )
    with torch.autocast(
        hidden.device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
        cache_enabled=policy != "chunks_uncached",
    ):
        if policy.startswith("native"):
            logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), weight)
            if logits.dtype in (torch.bfloat16, torch.float16):
                logits = logits.float()
            return F.cross_entropy(logits, targets.reshape(-1))
        return chunked_linear_cross_entropy(hidden, weight, targets, chunk)


def weight_casts(loss, weight):
    """Count actual cast nodes immediately above this exact FP32 leaf."""
    pending, seen, matches = [loss.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        children = [child for child, _ in node.next_functions]
        if "ToCopyBackward" in type(node).__name__ and any(
            getattr(child, "variable", None) is weight for child in children
        ):
            matches.append(node)
        pending.extend(children)
    return matches


def inspect_casts(loss, weight):
    events, handles = [], []
    nodes = weight_casts(loss, weight)
    for node in nodes:

        def hook(values):
            events.append({"dtype": str(values[0].dtype), "shape": list(values[0].shape)})

        handles.append(node.register_prehook(hook))
    return events, handles, len(nodes)


def error(actual, reference):
    difference = actual.double() - reference.double()
    return {
        "relative_l2": (difference.norm() / reference.double().norm().clamp_min(1e-30)).item(),
        "max_abs": difference.abs().max().item(),
        "norm": actual.double().norm().item(),
    }


def qualify():
    checks = []
    for shape in ((2, 5, 3), (1, 7, 3)):
        generator = torch.Generator().manual_seed(113)
        hidden = torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
        weight = torch.randn(5, 3, generator=generator, dtype=torch.float64).requires_grad_()
        targets = torch.arange(shape[0] * shape[1]).reshape(shape[:-1]) % 5
        targets.reshape(-1)[::3] = -100
        reference = None
        for policy in POLICIES:
            loss = loss_for(hidden, weight, targets, policy, 3)
            gradients = torch.autograd.grad(loss, (hidden, weight))
            if reference is None:
                reference = loss.detach(), gradients
            torch.testing.assert_close(loss, reference[0], atol=1e-10, rtol=1e-10)
            for actual, expected in zip(gradients, reference[1], strict=True):
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
            checks.append({"shape": shape, "policy": policy, "double_value_and_gradient": True})
    numerical = []
    for policy in ("chunks_cached", "chunks_uncached", "chunks_fp32"):
        assert torch.autograd.gradcheck(
            lambda h, w: loss_for(h, w, targets, policy, 3),
            (hidden, weight),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        numerical.append(policy)
    witness = []
    for device in ("cpu", "cuda"):
        for terms in ([1, 1, 1, 1, 256], [256, 1, 1, 1, 1]):
            for cached in (True, False):
                w = torch.ones(1, 1, device=device, requires_grad=True)
                with torch.autocast(device, dtype=torch.bfloat16, cache_enabled=cached):
                    values = [F.linear(torch.tensor([[float(t)]], device=device), w) for t in terms]
                    total = torch.stack(values).sum()
                events, handles, count = inspect_casts(total, w)
                total.backward()
                assert count == (1 if cached else 5)
                assert all(e["dtype"] == "torch.bfloat16" for e in events)
                if not cached:
                    assert w.grad.item() == 260.0
                witness.append(
                    {
                        "device": device,
                        "terms": terms,
                        "cached": cached,
                        "gradient": w.grad.item(),
                        "exact_sum": 260,
                        "weight_cast_nodes": count,
                        "events": events,
                    }
                )
                for handle in handles:
                    handle.remove()
                del total, values, w, events, handles
    return {
        "passed": True,
        "double_checks": checks,
        "finite_difference_policies": numerical,
        "linear_witness": witness,
        "optimizer_updates": 0,
    }
