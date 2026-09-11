"""Reuse all H127 probes; add the actual contiguous classifier shape."""

# ruff: noqa: I001
from results.native_buffer_loss_v1 import study as old
from results.native_buffer_layout_v1.operator import LayoutBufferLoss, model_loss
from pathlib import Path
from unittest.mock import patch

ROOT = Path("results/native_buffer_layout_v1")
torch = old.torch
original_qualify = old.qualify


def qualify():
    q = original_qualify()
    gen = torch.Generator().manual_seed(128)
    h = torch.randn(8, 512, 384, generator=gen).cuda().requires_grad_()
    w = (torch.randn(4096, 384, generator=gen) * 0.1).cuda().requires_grad_()
    y = (torch.arange(4096, device="cuda") * 3 % 4096).reshape(8, 512)
    before = [old.tensor_hash(x) for x in (h, w, y)]
    results = []
    for fn in (
        lambda h, w, y: old.F.cross_entropy(old.F.linear(h, w).flatten(0, 1), y.flatten()),
        LayoutBufferLoss.apply,
    ):
        loss = fn(h, w, y)
        grad = torch.autograd.grad(loss, (h, w))
        results.append((loss.detach(), *grad))
    exact = [torch.equal(a, b) for a, b in zip(*results, strict=True)]
    finite = all(bool(torch.isfinite(v).all()) for v in results[1])
    untouched = before == [old.tensor_hash(x) for x in (h, w, y)]
    q["full_shape"] = dict(
        shape=[8, 512, 384, 4096],
        bitwise_equal=exact,
        finite=finite,
        inputs_unchanged=untouched,
        passed=all(exact) and finite and untouched,
    )
    q["passed"] = q["passed"] and q["full_shape"]["passed"]
    q["backwards"] = 18
    return q


if __name__ == "__main__":
    with (
        patch.object(old, "ROOT", ROOT),
        patch.object(old, "NativeBufferLoss", LayoutBufferLoss),
        patch.object(old, "model_loss", model_loss),
        patch.object(old, "qualify", qualify),
    ):
        old.run()
    # The inherited top-level counter assumes 16 qualifier backwards; correct
    # that metadata explicitly for the two prospectively added backwards.
    r = old.read(ROOT / "result.json")
    assert r["qualification"]["backwards"] == 18
    r["backwards"] += 2
    old.write_json(ROOT / "result.json", r)
