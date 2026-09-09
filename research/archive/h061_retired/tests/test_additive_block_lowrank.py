"""Independent references for the additive block/low-rank research candidate."""

import math

import pytest
import torch
from torch.func import functional_call

from src.additive_block_lowrank_ffn import AdditiveBlockLowRankFFN, AdditiveBlockLowRankLinear
from src.additive_block_lowrank_ffn.witness import set_quadratic_witness


def relative_error(actual, expected):
    return (actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)


@pytest.mark.parametrize("n,m,groups,rank", [(8, 16, 2, 3), (16, 8, 4, 3)])
def test_dense_reference_and_all_factor_derivatives(n, m, groups, rank):
    layer = AdditiveBlockLowRankLinear(n, m, groups, rank).double()
    x = torch.randn(2, 3, n, dtype=torch.float64, requires_grad=True)
    weight = torch.block_diag(*layer.local.weight.unbind()) + layer.left.weight @ layer.right.weight
    expected, actual = x @ weight.T, layer(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    parameters = dict(layer.named_parameters())
    a = torch.autograd.grad(actual.square().sum(), (x, *parameters.values()), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (x, *parameters.values()))
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-12)

    def call(z, *values):
        return functional_call(layer, dict(zip(parameters, values, strict=True)), (z,))

    arguments = (x, *parameters.values())
    assert torch.autograd.gradcheck(call, arguments, fast_mode=True)
    assert torch.autograd.gradgradcheck(call, arguments, fast_mode=True)


@pytest.mark.parametrize("width", [24, 96, 192, 384])
def test_actual_parameter_budget(width):
    ffn = AdditiveBlockLowRankFFN(width)
    expected = 19 * width * width // 8
    assert sum(p.numel() for p in ffn.parameters()) == expected
    assert 1 - expected / (8 * width**2) == 0.703125
    assert ffn.hidden == 8 * width // 3 and ffn.rank == width // 8


@pytest.mark.parametrize("width", [24, 96])
def test_constructed_quadratic_forward(width):
    ffn = AdditiveBlockLowRankFFN(width).double()
    set_quadratic_witness(ffn)
    generator = torch.Generator().manual_seed(105)
    x = torch.randn(3, width, dtype=torch.float64, generator=generator)
    expected = torch.zeros_like(x)
    expected[:, 0] = 0.5 * x.square().sum(-1)
    expected[:, 1] = 0.5 * (x.square() * torch.arange(1, width + 1)).sum(-1)
    expected[:, 2] = 0.5 * x.sum(-1).square()
    torch.testing.assert_close(ffn(x), expected, rtol=1e-12, atol=1e-10)


def test_constructed_quadratic_jacobians_and_hessians():
    d = 24
    ffn = AdditiveBlockLowRankFFN(d).double()
    set_quadratic_witness(ffn)
    weights = torch.arange(1, d + 1, dtype=torch.float64)
    expected_hessians = (
        torch.eye(d, dtype=torch.float64),
        torch.diag(weights),
        torch.ones(d, d, dtype=torch.float64),
    )
    for x in (
        torch.zeros(d, dtype=torch.float64),
        torch.linspace(-0.7, 0.4, d, dtype=torch.float64),
    ):
        jacobian = torch.autograd.functional.jacobian(ffn, x)
        expected_jacobian = torch.zeros(d, d, dtype=torch.float64)
        expected_jacobian[:3] = torch.stack((x, weights * x, torch.ones_like(x) * x.sum()))
        torch.testing.assert_close(jacobian, expected_jacobian, rtol=1e-11, atol=1e-11)
        for index, expected in enumerate(expected_hessians):
            hessian = torch.autograd.functional.hessian(lambda z: ffn(z)[index], x)
            torch.testing.assert_close(hessian, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_initialization_component_energies_and_no_dead_global_factor(seed):
    ffn = AdditiveBlockLowRankFFN(384)
    ffn.initialize(seed, "blocks.0.ffn", 8)
    for label in ("up", "down", "gate"):
        layer = getattr(ffn, label)
        sigma = 0.02 / 4 if label == "down" else 0.02
        energy = 0.5 * layer.input_width * layer.output_width * sigma**2
        assert layer.local.weight.double().square().sum().item() == pytest.approx(energy, rel=1e-5)
        low_rank = layer.left.weight.double() @ layer.right.weight.double()
        assert low_rank.square().sum().item() == pytest.approx(energy, rel=1e-5)
        assert all(torch.count_nonzero(p) > 0 for p in layer.parameters())
    x = torch.randn(2, 384)
    ffn(x).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in ffn.parameters()
    )


def test_frozen_lr_rule_matches_independent_first_order_perturbations():
    layer = AdditiveBlockLowRankLinear(24, 64, 8, 3).double()
    layer.initialize(17, "perturbation", 0.02)
    n, m, rank, draws = 24, 64, 3, 256
    scales = layer.parameter_lr_multipliers()
    generator = torch.Generator().manual_seed(601)
    delta_s = torch.zeros(draws, m, n, dtype=torch.float64)
    for group in range(8):
        delta_s[:, group * 8 : (group + 1) * 8, group * 3 : (group + 1) * 3] = (
            torch.randn(draws, 8, 3, dtype=torch.float64, generator=generator)
            * scales[id(layer.local.weight)]
        )
    delta_l = (
        torch.randn(draws, m, rank, dtype=torch.float64, generator=generator)
        * scales[id(layer.left.weight)]
    )
    delta_r = (
        torch.randn(draws, rank, n, dtype=torch.float64, generator=generator)
        * scales[id(layer.right.weight)]
    )
    actual = delta_s + delta_l @ layer.right.weight.detach() + layer.left.weight.detach() @ delta_r
    assert actual.square().sum((-2, -1)).mean().item() / (m * n) == pytest.approx(1.0, rel=0.10)
    assert scales[id(layer.local.weight)] == 2
    assert all(math.isfinite(value) and value > 0 for value in scales.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification")
def test_native_bf16_forward_and_all_gradients():
    torch.backends.cuda.matmul.allow_tf32 = False
    ffn = AdditiveBlockLowRankFFN(384)
    ffn.initialize(17, "blocks.0.ffn", 8)
    ffn = ffn.cuda()
    generator = torch.Generator(device="cuda").manual_seed(601)
    x = torch.randn(2, 16, 384, device="cuda", generator=generator, requires_grad=True)
    cotangent = torch.randn(2, 16, 384, device="cuda", generator=generator)
    reference = ffn(x)
    reference_grads = torch.autograd.grad((reference * cotangent).sum(), (x, *ffn.parameters()))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = ffn(x)
    actual_grads = torch.autograd.grad((actual * cotangent).sum(), (x, *ffn.parameters()))
    assert relative_error(actual, reference) <= 0.02
    for a, b in zip(actual_grads, reference_grads, strict=True):
        assert torch.isfinite(a).all() and relative_error(a, b) <= 0.05


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_gate_recomputation_preserves_changed_weight_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA qualification")
    ffn = AdditiveBlockLowRankFFN(24).to(device)
    x = torch.linspace(-1, 1, 48, device=device).reshape(2, 24).requires_grad_()
    for changed in (False, True):
        if changed:
            with torch.no_grad():
                for p in ffn.parameters():
                    p.mul_(1.2).add_(0.01)
        values, gradients = [], []
        for recompute in (False, True):
            ffn.recompute_gate = recompute
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                value = ffn(x)
            values.append(value)
            gradients.append(torch.autograd.grad(value.square().sum(), (x, *ffn.parameters())))
        torch.testing.assert_close(*values, rtol=0, atol=0)
        for a, b in zip(*gradients, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_invalid_dimensions_and_policy_are_rejected():
    for args in ((0, 8, 2, 1), (8, 12, 3, 2), (8, 12, 2, 9)):
        with pytest.raises(ValueError):
            AdditiveBlockLowRankLinear(*args)
    with pytest.raises(ValueError, match="divisible by 24"):
        AdditiveBlockLowRankFFN(32)
    ffn = AdditiveBlockLowRankFFN(24)
    ffn.recompute_gate = True
    ffn.gate_recompute_method = "checkpoint"
    with pytest.raises(ValueError, match="native"):
        ffn(torch.ones(2, 24))


def test_transformer_count_optimizer_membership_and_independent_lr_values():
    from src.core.benchmark import forward_flops
    from src.core.config import ModelConfig, TrainConfig
    from src.core.optimization import parameter_groups
    from src.core.trainer import configure_gate_recomputation
    from src.core.transformer import Transformer

    config = ModelConfig(variant="additive_block_lowrank_swiglu", width=384, layers=8)
    model = Transformer(config)
    assert sum(p.numel() for p in model.parameters()) == config.total_parameters == 9099648
    assert config.unique_ffn_parameters == 2801664
    assert forward_flops(config)["ffn_matrix_forward_flops_per_token"] == 5603328
    tc = TrainConfig(
        ffn_lr_mode="perturbation", recompute_gate=True, gate_recompute_method="native"
    )
    configure_gate_recomputation(model, tc)
    groups = parameter_groups(model, tc)
    lookup = {id(p): g for g in groups for p in g["params"]}
    assert len(lookup) == sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    ffn = model.blocks[0].ffn
    for layer, left, right in (
        (ffn.up, 1.25, math.sqrt(25 / 6)),
        (ffn.gate, 1.25, math.sqrt(25 / 6)),
        (ffn.down, math.sqrt(50 / 3), 2.5),
    ):
        for parameter, scale in (
            (layer.local.weight, 2.0),
            (layer.left.weight, left),
            (layer.right.weight, right),
        ):
            assert lookup[id(parameter)]["lr_scale"] == pytest.approx(scale)
            assert lookup[id(parameter)]["weight_decay"] == 0.1
    assert lookup[id(model.embedding.weight)]["lr_scale"] == 1
    assert all(b.ffn.recompute_gate for b in model.blocks)


def test_integrated_recipe_rejects_undeclared_dimensions_and_optimizer_policy():
    from src.core.config import ModelConfig, TrainConfig
    from src.core.optimization import parameter_groups
    from src.core.transformer import Transformer

    for values in ({"hidden": 128}, {"groups": 4}, {"width": 32, "heads": 4}):
        with pytest.raises(ValueError, match="Additive recipe"):
            ModelConfig(variant="additive_block_lowrank_swiglu", **values).validate()
    model = Transformer(
        ModelConfig(
            variant="additive_block_lowrank_swiglu",
            width=24,
            layers=2,
            heads=3,
            context=8,
            vocab_size=32,
        )
    )
    for tc in (
        TrainConfig(),
        TrainConfig(ffn_lr_mode="fan_in"),
        TrainConfig(ffn_lr_mode="perturbation", ffn_decay_mode="product"),
    ):
        with pytest.raises(ValueError, match="Additive qualification"):
            parameter_groups(model, tc)
    dense = Transformer(ModelConfig(width=24, layers=2, heads=3, context=8, vocab_size=32))
    with pytest.raises(ValueError, match="specific to the additive"):
        parameter_groups(dense, TrainConfig(ffn_lr_mode="perturbation"))
