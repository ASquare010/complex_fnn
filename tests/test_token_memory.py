"""Independent equations and failure-prone cases for token memory helpers."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.core.token_memory import chunked_ffn, chunked_linear_cross_entropy


@pytest.mark.parametrize("size", [1, 4, 99])
def test_ffn_double_outputs_and_input_weight_gradients(size):
    torch.manual_seed(18)
    ffn = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 3)).double()
    x = torch.randn(2, 3, 3, dtype=torch.float64).transpose(0, 1).requires_grad_()
    probe = torch.randn_like(x)
    reference, actual = ffn(x), chunked_ffn(ffn, x, size)
    torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-11)
    inputs = (x, *ffn.parameters())
    for a, b in zip(
        torch.autograd.grad((reference * probe).sum(), inputs),
        torch.autograd.grad((actual * probe).sum(), inputs),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("size", [1, 4, 99])
@pytest.mark.parametrize("masked", [False, True])
def test_uneven_cross_entropy_and_classifier_gradients(size, masked):
    torch.manual_seed(13)
    x = torch.randn(3, 2, 3, dtype=torch.float64).transpose(0, 1).requires_grad_()
    w = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 1, 2], [3, 4, 5]])
    if masked:
        labels[0, 1:] = -100
    reference = F.cross_entropy(F.linear(x, w).flatten(0, 1), labels.flatten())
    actual = chunked_linear_cross_entropy(x, w, labels, size)
    torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-11)
    for a, b in zip(
        torch.autograd.grad(reference, (x, w)), torch.autograd.grad(actual, (x, w)), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


def test_gradcheck_and_all_ignored():
    torch.manual_seed(7)
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    y = torch.arange(5)
    assert torch.autograd.gradcheck(
        lambda a, b: chunked_linear_cross_entropy(a, b, y, 2), (x, w), fast_mode=True
    )
    assert torch.isnan(chunked_linear_cross_entropy(x, w, torch.full_like(y, -100), 2))
    with torch.no_grad():
        torch.testing.assert_close(chunked_ffn(torch.sin, x, 2), x.sin())


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_chunk_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        chunked_ffn(torch.sin, torch.ones(2, 3), size)
    with pytest.raises(ValueError, match="positive integer"):
        chunked_linear_cross_entropy(torch.ones(2, 3), torch.ones(4, 3), torch.zeros(2), size)
