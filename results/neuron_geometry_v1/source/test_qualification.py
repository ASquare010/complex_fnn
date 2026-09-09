"""Frozen H088 formula, geometry, gradient and GPU qualification."""

import copy
import math
from pathlib import Path

import pytest
import torch
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from results.blast_operator_recovery_v1.source.storage import record_observation
from results.neuron_geometry_v1.source.model import (
    CANDIDATES,
    COUNTS,
    EXTRAS,
    FORMS,
    GeometryFFN,
    GroupSort,
    LocalCurve,
    PairTwist,
    twist,
)

ROOT = Path("results/neuron_geometry_v1/qualification")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {name: cpu(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(item) for item in value]
    return value


def record(label, data, payload=None):
    record_observation(
        ROOT, label, label.split("_")[0], data, cpu(payload) if payload is not None else None
    )


def reference(z, curve):
    work = z if z.dtype == torch.float64 else z.float()
    if isinstance(curve, PairTwist):
        groups = curve.theta.numel()
        paired = work.reshape(*z.shape[:-1], groups, -1, 2)
        first, second = paired[..., 0] - curve.center, paired[..., 1]
        radius2 = first * first + second * second
        phi = 2 * torch.atan(2 * curve.theta.tanh().unsqueeze(-1) * radius2 / (1 + radius2))
        out = torch.stack(
            (
                curve.center + phi.cos() * first - phi.sin() * second,
                phi.sin() * first + phi.cos() * second,
            ),
            -1,
        )
        return out.reshape_as(z).to(z.dtype)
    grouped = work.reshape(*z.shape[:-1], curve.groups, -1)
    u = grouped.clamp(0, 1)
    controls = curve.theta.tanh()
    if curve.family == "bezier":
        correction = controls[:, 0, None] * (u - u**2)
    else:
        symmetric = u**2 - 2 * u**3 + u**4
        asymmetric = -(u**2) + 4 * u**3 - 5 * u**4 + 2 * u**5
        correction = 2 * controls[:, 0, None] * symmetric + 2 * controls[:, 1, None] * asymmetric
    correction = torch.where((grouped > 0) & (grouped < 1), correction, 0)
    return F.relu(z) + correction.reshape_as(z).to(z.dtype)


@pytest.mark.parametrize("form", FORMS)
def test_counts(form):
    model = GeometryFFN(form)
    actual = sum(p.numel() for p in model.parameters())
    controls = sum(p.numel() for p in model.curve.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    assert actual == COUNTS[form] and controls == EXTRAS[form]
    assert buffers == (4 if form == "twist_fixed" else 0)
    if not form.startswith("full_"):
        assert 1 - actual / 1179648 >= 0.70
    record(
        f"count_{form}",
        {
            "parameters": actual,
            "controls": controls,
            "buffer_elements": buffers,
            "hidden": model.hidden,
            "ffn_reduction_percent": 100 * (1 - actual / 1179648),
        },
    )


@pytest.mark.parametrize("kind", ["twist1", "twist4", "bezier1", "bump4"])
def test_formula_and_gradcheck(kind):
    curve = (
        PairTwist(int(kind[-1]))
        if kind.startswith("twist")
        else LocalCurve("bezier" if kind.startswith("bezier") else "bump", int(kind[-1]))
    ).double()
    with torch.no_grad():
        curve.theta.copy_(torch.linspace(-0.3, 0.4, curve.theta.numel()).reshape_as(curve.theta))
    z = torch.linspace(-0.73, 1.93, 32, dtype=torch.float64).reshape(2, 16).requires_grad_()
    actual, expected = curve(z), reference(z, curve)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    cotangent = torch.linspace(0.2, 1.2, 32, dtype=torch.float64).reshape_as(z)
    ga = torch.autograd.grad(actual, (z, curve.theta), cotangent, retain_graph=True)
    ge = torch.autograd.grad(expected, (z, curve.theta), cotangent)
    for a, b in zip(ga, ge):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    calls = [0]

    def evaluate(x, theta):
        calls[0] += 1
        return functional_call(curve, {"theta": theta}, (x,))

    assert torch.autograd.gradcheck(
        evaluate, (z, curve.theta), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True
    )
    record(
        f"formula_{kind}",
        {"numerical_check_forwards": calls[0]},
        {
            "actual": [actual, *ga],
            "expected": [expected, *ge],
            "input": z,
            "theta": curve.theta,
            "cotangent": cotangent,
        },
    )


@pytest.mark.parametrize("center", [0.0, 1.0])
def test_geometry(center):
    inputs, outputs, inverses, jacobians, singular_values, amplitudes = [], [], [], [], [], []
    for amplitude in (-1.9, -1.0, 0.0, 1.0, 1.9):
        theta = torch.tensor([math.atanh(amplitude / 2)], dtype=torch.float64)
        for radius in (0, 0.01, 0.3, 1, 3, 100):
            for angle in (0, 0.4, 1.2, 2.7):
                x = torch.tensor(
                    [center + radius * math.cos(angle), radius * math.sin(angle)],
                    dtype=torch.float64,
                )
                y = twist(x, theta, center)
                restored = twist(y, -theta, center)
                jacobian = torch.autograd.functional.jacobian(lambda q: twist(q, theta, center), x)
                values = torch.linalg.svdvals(jacobian)
                origin = x.new_tensor([center, 0])
                torch.testing.assert_close(
                    (y - origin).square().sum(), (x - origin).square().sum(), atol=1e-10, rtol=1e-10
                )
                torch.testing.assert_close(restored, x, atol=1e-10, rtol=1e-10)
                assert abs(torch.linalg.det(jacobian).item() - 1) <= 1e-10
                assert values.min() >= math.sqrt(2) - 1 - 1e-10
                assert values.max() <= math.sqrt(2) + 1 + 1e-10
                inputs.append(x)
                outputs.append(y)
                inverses.append(restored)
                jacobians.append(jacobian)
                singular_values.append(values)
                amplitudes.append(amplitude)
    record(
        f"geometry_{int(center)}",
        {"center": center, "cases": len(inputs)},
        {
            "input": torch.stack(inputs),
            "output": torch.stack(outputs),
            "inverse": torch.stack(inverses),
            "jacobian": torch.stack(jacobians),
            "singular_values": torch.stack(singular_values),
            "amplitude": torch.tensor(amplitudes, dtype=torch.float64),
        },
    )


def test_initial_functions():
    x = torch.linspace(-1.3, 1.4, 768).reshape(2, 384)
    tested = []
    for candidate, base in (
        ("bezier_1p", "narrow_relu"),
        ("bump_2p", "narrow_relu"),
        ("twist_identity", "linear"),
        ("twist_learned", "twist_fixed"),
    ):
        a, b = GeometryFFN(candidate), GeometryFFN(base)
        assert torch.equal(a.up.weight, b.up.weight) and torch.equal(a.down.weight, b.down.weight)
        assert torch.equal(a(x), b(x))
        tested.append([candidate, base])
    record("initial_functions", {"exact_pairs": tested})


@pytest.mark.parametrize("form", CANDIDATES)
def test_cuda_ffn(form):
    model = GeometryFFN(form).cuda()
    with torch.no_grad():
        model.curve.theta.copy_(
            torch.linspace(-0.25, 0.3, model.curve.theta.numel(), device="cuda").reshape_as(
                model.curve.theta
            )
        )
    checkpointed = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(9830)
    original = torch.randn(3, 2, 384, generator=generator).transpose(0, 1).cuda()
    cotangent = torch.randn(2, 3, 384, generator=generator).cuda()
    x, xr, xc = (original.detach().clone().requires_grad_() for _ in range(3))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(x)
        expected = model.down(reference(model.up(xr), model.curve))
        recomputed = checkpoint(checkpointed, xc, use_reentrant=False)
    ga = torch.autograd.grad(actual, (x, *model.parameters()), cotangent)
    ge = torch.autograd.grad(expected, (xr, *model.parameters()), cotangent)
    gc = torch.autograd.grad(recomputed, (xc, *checkpointed.parameters()), cotangent)
    for a, b, c in zip((actual, *ga), (expected, *ge), (recomputed, *gc)):
        assert torch.isfinite(a).all() and a.norm() > 0
        torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)
        assert torch.equal(a, c)
    record(
        f"cuda_{form}",
        {"pairs": 1 + len(ga), "exact_checkpoint_pairs": 1 + len(gc)},
        {"actual": [actual, *ga], "expected": [expected, *ge], "checkpoint": [recomputed, *gc]},
    )


def test_groupsort():
    x = torch.tensor([[2.0, -1.0, 0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    actual = GroupSort()(x)
    expected = x[:, [1, 0, 2, 3]]
    assert torch.equal(actual, expected)
    cotangent = torch.tensor([[0.1, 0.2, -0.3, 0.4]], dtype=torch.float64)
    a = torch.autograd.grad(actual, x, cotangent)[0]
    b = cotangent[:, [1, 0, 2, 3]]
    assert torch.equal(a, b)
    record("groupsort_reference", {}, {"actual": [actual, a], "expected": [expected, b]})


def test_scalar_slopes():
    observations = {}
    for family, groups in (("bezier", 1), ("bump", 4)):
        curve = LocalCurve(family, groups).double()
        with torch.no_grad():
            curve.theta.copy_(torch.linspace(-4, 4, curve.theta.numel()).reshape_as(curve.theta))
        x = (
            torch.linspace(0.0001, 0.9999, 4096, dtype=torch.float64)
            .reshape(-1, groups)
            .requires_grad_()
        )
        derivative = torch.autograd.grad(curve(x).sum(), x)[0]
        margin = 1.0 if family == "bezier" else 2 / (3 * math.sqrt(3)) + 0.25
        assert derivative.min() >= 1 - margin - 1e-10
        assert derivative.max() <= 1 + margin + 1e-10
        endpoints = (
            torch.tensor([-2.0, 0.0, 1.0, 3.0], dtype=torch.float64)
            .repeat(groups)
            .reshape(4, groups)
        )
        assert torch.equal(curve(endpoints), F.relu(endpoints))
        observations[family] = {
            "minimum": derivative.min().item(),
            "maximum": derivative.max().item(),
        }
    record("scalar_slopes", observations)
