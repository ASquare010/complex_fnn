"""Independent algebra, rank limits and real optimization gates for paired features."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.core.benchmark import forward_flops
from src.core.config import ModelConfig
from src.core.transformer import Transformer
from src.paired_feature_ffn import MODES, PairedFeatureFFN, paired_features


def test_antipodal_bilinear_and_smooth_basis_identities():
    a = torch.linspace(-9, 9, 71, dtype=torch.float64)
    b = torch.linspace(2, -3, 71, dtype=torch.float64)
    z = torch.stack((a, b), dim=-1)
    plus, minus = paired_features(z, "antipodal").unbind(-1)
    torch.testing.assert_close(plus - minus, a * b, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(plus + minus, a * b * torch.tanh(a / 2), atol=1e-14, rtol=1e-14)
    # One real FFN pair constructs a pure product without relying on finite fitting.
    model = PairedFeatureFFN(2, 2, "antipodal").double()
    with torch.no_grad():
        model.up.weight.copy_(torch.eye(2))
        model.down.weight.copy_(torch.tensor([[1.0, -1.0], [0.0, 0.0]]))
    torch.testing.assert_close(model(z)[:, 0], a * b, atol=1e-14, rtol=1e-14)


def test_gate_direction_floor_does_not_imply_a_nonsingular_full_jacobian():
    a = torch.linspace(-80, 80, 321, dtype=torch.float64, requires_grad=True)
    z = torch.stack((a, torch.ones_like(a)), dim=-1)
    phi = paired_features(z, "antipodal")
    derivatives = [torch.autograd.grad(phi[:, i].sum(), a, retain_graph=True)[0] for i in (0, 1)]
    squared_norm = sum(d.square() for d in derivatives)
    assert squared_norm.min().item() >= 0.5 - 1e-14
    point = torch.tensor([0.7, -1.2], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda x: paired_features(x, "antipodal"), point)
    a, b = point
    expected = -b * a.square() * a.sigmoid() * (1 - a.sigmoid())
    torch.testing.assert_close(torch.linalg.det(jacobian), expected, atol=1e-14, rtol=1e-14)
    origin = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    zero_jacobian = torch.autograd.functional.jacobian(
        lambda x: paired_features(x, "antipodal"), origin
    )
    assert torch.count_nonzero(zero_jacobian) == 0


@pytest.mark.parametrize("mode", MODES)
def test_reference_equation_and_first_derivatives(mode):
    model = PairedFeatureFFN(4, 6, mode).double()
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    a, b = (x @ model.up.weight.T).chunk(2, -1)
    first = a * a.sigmoid() * b
    second = (
        -a * (-a).sigmoid() * b
        if mode == "antipodal"
        else b * b.sigmoid() * a
        if mode == "reciprocal"
        else first
    )
    expected = torch.cat((first, second), -1) @ model.down.weight.T
    torch.testing.assert_close(model(x), expected, atol=1e-14, rtol=1e-14)
    assert torch.autograd.gradcheck(model, (x,))
    assert torch.autograd.gradgradcheck(
        lambda z: paired_features(z, mode),
        (torch.randn(2, 4, dtype=torch.float64, requires_grad=True),),
    )


def test_duplicate_collapses_to_one_readout_and_has_rank_one_feature_pair():
    model = PairedFeatureFFN(4, 6, "duplicate").double()
    x = torch.randn(9, 4, dtype=torch.float64)
    a, b = model.up(x).chunk(2, -1)
    left, right = model.down.weight.chunk(2, -1)
    torch.testing.assert_close(model(x), F.linear(F.silu(a) * b, left + right))
    points = torch.tensor([[1.0, 2.0], [2.0, 1.0], [-1.0, 2.0]], dtype=torch.float64)
    assert torch.linalg.matrix_rank(paired_features(points, "duplicate")) == 1
    assert torch.linalg.matrix_rank(paired_features(points, "antipodal")) == 2
    assert torch.linalg.matrix_rank(paired_features(points, "reciprocal")) == 2


def test_exact_budget_flops_and_common_initial_weights():
    configs = [ModelConfig(variant=f"paired_{mode}_swiglu", hidden=228) for mode in MODES]
    configs += [
        ModelConfig(variant="swiglu_narrow", hidden=152),
        ModelConfig(variant="gelu", hidden=228),
    ]
    assert all(c.unique_ffn_parameters == 350208 and c.total_parameters == 1728192 for c in configs)
    assert all(forward_flops(c)["ffn_matrix_forward_flops_per_token"] == 700416 for c in configs)
    models = [
        Transformer(replace(c, width=24, hidden=12, heads=3, layers=2, vocab_size=32, context=8))
        for c in configs[:3] + configs[-1:]
    ]
    first = dict(models[0].named_parameters())
    for model in models[1:]:
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, first[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_pair_modes_cuda_bf16_backward_and_output_equation():
    for mode in MODES:
        model = PairedFeatureFFN(24, 36, mode).cuda()
        x = torch.randn(2, 8, 24, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
            reference = F.linear(
                paired_features(F.linear(x, model.up.weight), mode), model.down.weight
            )
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        output.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
