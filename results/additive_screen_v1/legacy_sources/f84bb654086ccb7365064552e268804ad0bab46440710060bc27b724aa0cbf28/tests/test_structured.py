"""Independent dense-equivalence and gradient checks for grouped FFNs."""

from dataclasses import replace

import pytest
import torch

from src.core.config import ModelConfig
from src.core.transformer import Transformer
from src.grouped_ffn import GroupedLinear, StructuredFFN, shuffle_channels


def test_grouped_linear_matches_explicit_block_diagonal():
    model = GroupedLinear(12, 20, 4).double()
    x = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    dense = torch.block_diag(*model.weight.unbind(0))
    torch.testing.assert_close(model(x), x @ dense.T, rtol=1e-14, atol=1e-14)
    assert torch.autograd.gradcheck(model, (x,))


def test_shuffle_is_a_bijection_with_expected_interleaving():
    x = torch.arange(12).reshape(1, 12)
    expected = torch.tensor([[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]])
    torch.testing.assert_close(shuffle_channels(x, 3), expected)


@pytest.mark.parametrize("variant", ["structured_gelu", "structured_swiglu", "swiglu_narrow"])
def test_exact_parameter_budget_and_common_initialization(variant):
    cfg = ModelConfig(variant=variant, groups=4)
    assert cfg.ffn_parameters == 73728
    candidate = Transformer(cfg)
    baseline = Transformer(replace(cfg, variant="gelu"))
    for name, p in candidate.named_parameters():
        if ".ffn." not in name:
            torch.testing.assert_close(p, dict(baseline.named_parameters())[name], atol=0, rtol=0)


@pytest.mark.parametrize("gated", [False, True])
def test_structured_ffn_matches_dense_expansion(gated):
    ffn = StructuredFFN(8, 16, 2, gated=gated).double()
    x = torch.randn(4, 8, dtype=torch.float64)
    up = torch.block_diag(*ffn.up.weight.unbind(0))
    down = torch.block_diag(*ffn.down.weight.unbind(0))
    z = x @ up.T
    if gated:
        gate = torch.block_diag(*ffn.gate.weight.unbind(0))
        z = torch.nn.functional.silu(z) * (x @ gate.T)
    else:
        z = torch.nn.functional.gelu(z)
    expected = shuffle_channels(z, 2) @ down.T
    torch.testing.assert_close(ffn(x), expected, atol=1e-14, rtol=1e-14)
