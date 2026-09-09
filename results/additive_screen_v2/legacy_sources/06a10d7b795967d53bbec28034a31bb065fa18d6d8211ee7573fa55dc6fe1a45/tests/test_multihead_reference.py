"""Independent loop and derivative checks for the untrained multi-head reference."""

import pytest
import torch
from torch.nn import functional as F

from src.multihead_ffn import ParallelMultiheadFFN, normalized_sigmoid


def explicit_loop(model, x):
    q = model.input_projection(x).reshape(-1, model.heads, model.head_width)
    heads = []
    for h in range(model.heads):
        query = q[:, h]
        raw = torch.sigmoid(query @ model.router[h])
        weights = raw / (raw.sum(-1, keepdim=True) + model.epsilon)
        total = torch.zeros_like(query)
        for e in range(model.subnetworks):
            features = F.silu(query @ model.gate[h, e]) * (query @ model.value[h, e])
            total = total + weights[:, e, None] * (features @ model.down[h, e])
        heads.append(total)
    return model.output_projection(torch.cat(heads, -1).reshape(x.shape))


def test_forward_and_every_derivative_match_explicit_loop():
    torch.manual_seed(61)
    model = ParallelMultiheadFFN(8, 2, 3, 5).double()
    # Enlarge tiny standalone weights so derivative comparisons are informative.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(10)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(x)
    vectorized = model(x)
    reference = explicit_loop(model, x)
    torch.testing.assert_close(vectorized, reference, rtol=1e-11, atol=1e-12)
    variables = (x, *tuple(model.parameters()))
    actual = torch.autograd.grad((vectorized * direction).sum(), variables)
    expected = torch.autograd.grad((reference * direction).sum(), variables)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
        assert torch.isfinite(a).all()


def test_normalization_retains_epsilon_and_avoids_premature_underflow():
    logits = torch.tensor([[0.0, 0.0], [-100.0, -100.0], [100.0, -100.0]], requires_grad=True)
    actual = normalized_sigmoid(logits, epsilon=1e-6)
    reference = torch.sigmoid(logits.double())
    reference = reference / (reference.sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual.double(), reference, rtol=2e-5, atol=1e-44)
    assert actual[1].min() > 0 and actual[0].sum() < 1
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_budget_matches_actual_tensors():
    model = ParallelMultiheadFFN(384, 48, 2, 24)
    assert model.parameter_count == sum(p.numel() for p in model.parameters()) == 350976
    assert 8 * model.parameter_count == 2807808


@pytest.mark.parametrize("args", [(8, 3, 2, 4), (8, 2, 0, 4), (8, 2, 2, 0)])
def test_invalid_dimensions_reject(args):
    with pytest.raises(ValueError):
        ParallelMultiheadFFN(*args)


def test_invalid_epsilon_rejects():
    with pytest.raises(ValueError):
        normalized_sigmoid(torch.zeros(1, 2), epsilon=0)
