"""Independent numerical/layout and adapter-lifetime checks."""

import pytest
import torch
from torch.nn import functional as F

from src.core.config import ModelConfig
from src.core.training_memory import (
    buffer_cross_entropy,
    buffer_model_loss,
    offload_checkpoint_inputs,
)
from src.core.transformer import Transformer


@pytest.mark.parametrize("layout", ["row", "column", "slice", "3d"])
@pytest.mark.parametrize("masked", [False, True])
def test_native_loss_gradients_and_input_ownership(layout, masked):
    torch.manual_seed(41)
    x = torch.randn(6, 4, dtype=torch.float64)
    if layout == "column":
        x = x.t().contiguous().t()
    elif layout == "slice":
        x = torch.randn(6, 8, dtype=torch.float64)[:, ::2]
    elif layout == "3d":
        x = x.reshape(2, 3, 4)
    x.requires_grad_()
    w = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    y = torch.arange(6).reshape(x.shape[:-1])
    if masked:
        y.reshape(-1)[::2] = -100
    snapshots = [v.detach().clone() for v in (x, w, y)]
    reference = F.cross_entropy(F.linear(x, w).reshape(-1, 7), y.reshape(-1))
    actual = buffer_cross_entropy(x, w, y)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    ga = torch.autograd.grad(actual * 1.7, (x, w))
    gb = torch.autograd.grad(reference * 1.7, (x, w))
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
        assert left.stride() == right.stride()
    for value, snapshot in zip((x, w, y), snapshots, strict=True):
        assert torch.equal(value, snapshot)


def test_finite_difference_and_all_ignored():
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 1, -100])
    assert torch.autograd.gradcheck(
        lambda a, b: buffer_cross_entropy(a, b, y), (x, w), fast_mode=True
    )
    ignored = torch.full_like(y, -100)
    actual = buffer_cross_entropy(x, w, ignored)
    reference = F.cross_entropy(F.linear(x, w), ignored)
    assert torch.isnan(actual) and torch.isnan(reference)
    for ga, gb in zip(
        torch.autograd.grad(actual, (x, w)), torch.autograd.grad(reference, (x, w)), strict=True
    ):
        torch.testing.assert_close(ga, gb, equal_nan=True)


def model():
    return Transformer(
        ModelConfig(width=8, heads=2, layers=2, hidden=12, vocab_size=17, context=8), 41
    )


def test_tied_classifier_and_restore_on_exception():
    m = model()
    m.set_recompute_scope("block")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[2, 3, 4], [5, 6, 7]])
    identity = {k: id(v) for k, v in m.named_parameters()}
    keys = list(m.state_dict())
    reference = m.loss(x, y)
    native = torch.autograd.grad(reference, tuple(m.parameters()))
    with offload_checkpoint_inputs(m, 1):
        actual = buffer_model_loss(m, x, y)
        changed = torch.autograd.grad(actual, tuple(m.parameters()))
        assert "forward" not in m.blocks[0].__dict__ and "forward" in m.blocks[1].__dict__
        with pytest.raises(ValueError, match="original forward"):
            with offload_checkpoint_inputs(m, 1):
                pass
    for ga, gb in zip(native, changed, strict=True):
        torch.testing.assert_close(ga, gb, rtol=1e-5, atol=1e-7)
    with pytest.raises(RuntimeError, match="sentinel"):
        with offload_checkpoint_inputs(m, 2):
            raise RuntimeError("sentinel")
    assert all("forward" not in b.__dict__ for b in m.blocks)
    assert identity == {k: id(v) for k, v in m.named_parameters()} and keys == list(m.state_dict())


@pytest.mark.parametrize("count", [-1, 3, 1.5, True])
def test_bad_count_is_atomic(count):
    m = model()
    with pytest.raises(ValueError):
        with offload_checkpoint_inputs(m, count):
            pass
    assert all("forward" not in b.__dict__ for b in m.blocks)


def test_preflight_and_zero_count():
    m = model()
    with offload_checkpoint_inputs(m, 0):
        assert all("forward" not in b.__dict__ for b in m.blocks)
    with pytest.raises(ValueError, match="whole-block"):
        with offload_checkpoint_inputs(m, 2):
            pass
    m.set_recompute_scope("block")
    m.blocks[1].forward = lambda x: x
    with pytest.raises(ValueError, match="original forward"):
        with offload_checkpoint_inputs(m, 2):
            pass
    assert "forward" not in m.blocks[0].__dict__


def test_input_scope():
    x = torch.randn(3, 2)
    w = torch.randn(4, 2)
    y = torch.zeros(3, dtype=torch.long)
    for xx, ww, yy in [
        (x.half(), w.half(), y),
        (x, w, y.float()),
        (x, w, y[:2]),
        (x, w.t(), y),
        (torch.randn(2, 3, 2).transpose(0, 1), w, torch.zeros(3, 2, dtype=torch.long)),
    ]:
        with pytest.raises(ValueError):
            buffer_cross_entropy(xx, ww, yy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            buffer_cross_entropy(x, w, y)
