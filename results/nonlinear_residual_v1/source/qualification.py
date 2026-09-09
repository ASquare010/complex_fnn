"""Twenty-five frozen numerical checks and saved-pair verification for H099."""

import math
from pathlib import Path

import torch
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from results.blast_operator_recovery_v1.source.storage import read, record_observation, sha
from results.nonlinear_residual_v1.source.model import (
    FORMS,
    TASKS,
    Curve,
    ResidualTaskFFN,
    counts,
    rational,
)

ROOT = Path("results/nonlinear_residual_v1/qualification")


def record(name, meta, pairs=()):
    saved = []
    for actual, expected, tol in pairs:
        actual, expected = actual.detach().cpu(), expected.detach().cpu()
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        saved.append({"actual": actual, "expected": expected, "tolerance": tol})
    record_observation(ROOT, name, name.split("_")[0], meta, saved if saved else None)


def curve_reference(z, curve):
    w = z if z.dtype == torch.float64 else z.float()
    if hasattr(curve, "theta"):
        q = w.reshape(*z.shape[:-1], curve.theta.numel(), -1)
        even = q**2 / (1 + q.abs())
        odd = q**3 / (1 + q.abs()) ** 2
        value = even + 2 * curve.theta.tanh().unsqueeze(-1) * odd
        return value.reshape_as(z).to(z.dtype)
    positive = torch.where(w > 0, w * w, torch.zeros_like(w))
    value = curve.scale * positive + curve.offset if curve.form == "starrelu" else positive
    return value.to(z.dtype)


def test_count(form):
    actual = counts(form)
    expected = (
        6144
        if form == "feature_aware"
        else 1182080
        if form == "full_swiglu"
        else (
            1181568
            if form.startswith("full_")
            else 351200
            if form == "narrow_swiglu"
            else 351050
            if form == "starrelu"
            else 351052
            if form == "learned_mix"
            else 351048
        )
    )
    assert actual["parameters"] == expected
    assert actual["parameters"] == (
        actual["matrix_parameters"] + actual["bias_parameters"] + actual["activation_parameters"]
    )
    model = ResidualTaskFFN(form)
    assert torch.count_nonzero(model.down.weight) == 0
    if model.down.bias is not None:
        assert torch.count_nonzero(model.down.bias) == 0
    record(f"count_{form}", actual)


def test_formula(form):
    curve = Curve(form).double()
    if form == "learned_mix":
        with torch.no_grad():
            curve.theta.copy_(torch.tensor([-0.7, 0.0, 0.4, 1.2], dtype=torch.float64))
    elif form == "starrelu":
        with torch.no_grad():
            curve.scale.fill_(1.3)
            curve.offset.fill_(-0.2)
    x = torch.tensor(
        [[-3.0, -1.0, 0.0, 0.1, 0.5, 1.0, 2.0, 4.0]], dtype=torch.float64, requires_grad=True
    )
    a, b = curve(x), curve_reference(x, curve)
    variables = (x, *curve.parameters())
    ga = torch.autograd.grad(a.sum(), variables, retain_graph=True)
    gb = torch.autograd.grad(b.sum(), variables)
    names = list(dict(curve.named_parameters()))
    calls = 0

    def operation(z, *parameters):
        nonlocal calls
        calls += 1
        return functional_call(curve, dict(zip(names, parameters)), (z,))

    assert torch.autograd.gradcheck(
        operation, variables, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True
    )
    record(
        f"formula_{form}",
        {"gradcheck_forwards": calls},
        [(aa, bb, 1e-11) for aa, bb in zip((a, *ga), (b, *gb))],
    )


def test_bound():
    theta = torch.tensor([-3.0, -0.3, 0.3, 3.0], dtype=torch.float64)
    x = torch.linspace(-8, 8, 4097, dtype=torch.float64)[:, None].repeat(1, 4).requires_grad_()
    value = rational(x, theta)
    opposite = rational(-x, theta)
    grad = torch.autograd.grad(value.sum(), x)[0]
    t = x.abs() / (1 + x.abs())
    expected = x.sign() * (2 * t - t**2) + 2 * theta.tanh() * t**2 * (3 - 2 * t)
    assert grad.abs().max() <= 3 + 1e-11
    record(
        "bound_parity",
        {"derivative_bound": 3, "points": 4097},
        [
            (grad, expected, 1e-11),
            ((value + opposite) / 2, x**2 / (1 + x.abs()), 1e-11),
            ((value - opposite) / 2, 2 * theta.tanh() * x**3 / (1 + x.abs()) ** 2, 1e-11),
        ],
    )


def test_oracle(data, task):
    model = ResidualTaskFFN("feature_aware", task=task).double()
    with torch.no_grad():
        model.down.weight.copy_((data["rotation"] / data["scales"][task]).T)
    x = data["x"][:257].double()
    q = x[:, data["indices"]]
    if task == "quadratic":
        latent = (q**2 - 1) / math.sqrt(2)
    elif task == "cubic":
        latent = (q**3 - 3 * q) / math.sqrt(6)
    elif task == "product":
        latent = q * q[:, (torch.arange(16) + 1) % 16]
    else:
        latent = (torch.where(q > 0, q * q, 0) - 0.5 - math.sqrt(2 / math.pi) * q) / math.sqrt(
            1.25 - 2 / math.pi
        )
    expected = latent @ (data["rotation"] / data["scales"][task])
    actual = model(x)
    record(
        f"oracle_{task}",
        {"samples": 257, "privileged_features": True},
        [(actual, expected, 1e-11), (actual, data["targets"][task][:257].double(), 5e-6)],
    )


def test_cuda(form):
    model = ResidualTaskFFN(form)
    with torch.no_grad():
        model.down.weight.normal_(0, 0.02, generator=torch.Generator().manual_seed(9872))
    model.cuda()
    gen = torch.Generator().manual_seed(9873)
    x0 = torch.randn(3, 2, 384, generator=gen).transpose(0, 1).cuda()
    cotangent = torch.randn(3, 2, 384, generator=gen).transpose(0, 1).cuda()
    pairs = []
    for bf16, tol in ((False, 1e-5), (True, 0.02)):
        observed = []
        for mode in ("native", "reference", "checkpoint"):
            x = x0.detach().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                if mode == "reference":
                    z = F.linear(x, model.up.weight, model.up.bias)
                    y = F.linear(
                        curve_reference(z, model.curve), model.down.weight, model.down.bias
                    )
                elif mode == "checkpoint":
                    y = checkpoint(model, x, use_reentrant=False)
                else:
                    y = model(x)
            grads = torch.autograd.grad(y, (x, *model.parameters()), cotangent)
            observed.append((y, *grads))
        pairs += [(a, b, tol) for a, b in zip(observed[0], observed[1])]
        pairs += [(a, b, 0) for a, b in zip(observed[0], observed[2])]
    record(
        f"cuda_{form}",
        {"fp32_atol_rtol": 1e-5, "bf16_atol_rtol": 0.02, "checkpoint_exact": True},
        pairs,
    )


def test_zero(data):
    x = data["x"][:8].double()
    record(
        "initial_zero",
        {"forms": len(FORMS)},
        [
            (ResidualTaskFFN(f).double()(x), torch.zeros(8, 384, dtype=torch.float64), 0)
            for f in FORMS
        ],
    )


def run(data):
    assert not list(ROOT.iterdir())
    for form in FORMS:
        test_count(form)
    for form in ("learned_mix", "narrow_relu2", "starrelu"):
        test_formula(form)
    test_bound()
    for task in TASKS:
        test_oracle(data, task)
    for form in ("even_rational", "fixed_mix", "learned_mix", "narrow_relu2", "starrelu"):
        test_cuda(form)
    test_zero(data)
    files = list(ROOT.glob("*.json"))
    assert len(files) == 25
    pairs = 0
    for path in files:
        row = read(path)
        if "tensor_file" not in row:
            continue
        p = path.with_name(row["tensor_file"])
        assert sha(p) == row["tensor_sha256"]
        for item in torch.load(p, map_location="cpu", weights_only=True):
            a, b, tol = item["actual"], item["expected"], item["tolerance"]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert ((a - b).abs() <= tol + tol * b.abs()).all()
            pairs += 1
    return {"checks": 25, "saved_pairs_independently_rechecked": pairs, "status": "PASS"}
