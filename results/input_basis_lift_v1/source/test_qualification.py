"""H092 formula, kernel, gradient and mixed-precision qualification; no training."""

import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from results.blast_operator_recovery_v1.source.storage import record_observation
from results.input_basis_lift_v1.source.model import FORMS, PAIRS, InputBasisFFN, pair_basis

ROOT = Path("results/input_basis_lift_v1/qualification")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


def record(label, data, payload=None):
    record_observation(ROOT, label, label.split("_")[0], data, cpu(payload))


def reference(z, kind):
    w = z if z.dtype == torch.float64 else z.float()
    if kind in ("rational", "duplicate"):
        a = w**2 / (1 + w.abs())
        b = w**3 / (1 + w.abs()) ** 2 if kind == "rational" else a
    elif kind == "hermite":
        a, b = (w**2 - 1) / math.sqrt(2), (w**3 - 3 * w) / math.sqrt(6)
    elif kind == "trig":
        a, b = 2 * (w / 2).sin() ** 2, 2 * (w / 2).sin() * (w / 2).cos() - w
    elif kind == "linear":
        a, b = w, w
    elif kind == "antipodal":
        a = w / 2 * (1 + (w / math.sqrt(2)).erf())
        b = -w / 2 * (1 - (w / math.sqrt(2)).erf())
    else:
        raise ValueError(kind)
    return torch.cat((a, b), -1).to(z.dtype)


@pytest.mark.parametrize("form", FORMS)
def test_count(form):
    model = InputBasisFFN(form)
    count = sum(p.numel() for p in model.parameters())
    assert count == (147456 if form == "core_linear" else 350208)
    assert torch.count_nonzero(model.down.weight[:, 384:]) == 0
    assert sum(b.numel() for b in model.buffers()) == 0
    record(f"count_{form}", {"parameters": count, "shape_parameters": 0})


@pytest.mark.parametrize("form", PAIRS)
def test_formula(form):
    x = torch.tensor(
        [[-3.2, -1.0, -0.1, 0.0, 0.1, 1.0, 3.2]], dtype=torch.float64, requires_grad=True
    )
    actual, expected = pair_basis(x, form), reference(x, form)
    grads_a, grads_b = [], []
    for a, b in zip(actual.chunk(2, -1), expected.chunk(2, -1)):
        grads_a.append(torch.autograd.grad(a.sum(), x, retain_graph=True)[0])
        grads_b.append(torch.autograd.grad(b.sum(), x, retain_graph=True)[0])
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    for a, b in zip(grads_a, grads_b):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    calls = 0

    def counted(z):
        nonlocal calls
        calls += 1
        return pair_basis(z, form)

    assert torch.autograd.gradcheck(counted, (x,), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
    record(
        f"formula_{form}",
        {"gradcheck_forwards": calls, "atol": 1e-11, "rtol": 1e-11},
        {"input": x, "actual": [actual, *grads_a], "reference": [expected, *grads_b]},
    )


def test_initial():
    x = torch.randn(3, 384, generator=torch.Generator().manual_seed(9840), dtype=torch.float64)
    expected = InputBasisFFN("core_linear").double()(x)
    actual = {form: InputBasisFFN(form).double()(x) for form in FORMS}
    for value in actual.values():
        torch.testing.assert_close(value, expected, atol=1e-11, rtol=1e-11)
    record(
        "initial_function",
        {"forms": len(FORMS), "atol": 1e-11, "rtol": 1e-11},
        {"actual": actual, "reference": expected},
    )


def test_collapse():
    generator = torch.Generator().manual_seed(9841)
    x = torch.randn(3, 384, generator=generator, dtype=torch.float64, requires_grad=True)
    payload = {}
    for form in ("duplicate", "linear", "antipodal"):
        model = InputBasisFFN(form).double()
        with torch.no_grad():
            model.down.weight[:, 384:].normal_(0, 0.02, generator=generator)
        a, v, w = torch.split(model.down.weight, [384, 176, 176], dim=1)
        u = model.up.weight
        if form == "duplicate":
            z = F.linear(x, u)
            expected = F.linear(x, a) + F.linear(z**2 / (1 + z.abs()), v + w)
        elif form == "linear":
            expected = F.linear(x, a + (v + w) @ u)
        else:
            expected = F.linear(x, a - w @ u) + F.linear(F.gelu(F.linear(x, u)), v + w)
        actual = model(x)
        variables = (x, *model.parameters())
        ga = torch.autograd.grad(actual.sum(), variables, retain_graph=True)
        gb = torch.autograd.grad(expected.sum(), variables)
        for aa, bb in zip((actual, *ga), (expected, *gb)):
            torch.testing.assert_close(aa, bb, atol=1e-11, rtol=1e-11)
        payload[form] = {"actual": [actual, *ga], "reference": [expected, *gb]}
    record("collapse_controls", {"atol": 1e-11, "rtol": 1e-11}, payload)


def test_kernel():
    generator = torch.Generator().manual_seed(9841)
    q, _ = torch.linalg.qr(torch.randn(384, 384, generator=generator, dtype=torch.float64))
    u, v = q[:, :304].T, q[:, 304]
    x = torch.randn(384, generator=generator, dtype=torch.float64)
    projection = u @ v
    torch.testing.assert_close(projection, torch.zeros_like(projection), atol=1e-10, rtol=0)
    a, b = pair_basis(u @ x, "rational"), pair_basis(u @ (x + v), "rational")
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    direct = torch.cat((x + v, b)) - torch.cat((x, a))
    torch.testing.assert_close(direct[:384], v, atol=1e-10, rtol=1e-10)
    assert direct.norm() >= v.norm() - 1e-10
    trace = (torch.eye(384, dtype=torch.float64) - u.T @ u).trace().item()
    assert abs(trace - 80) < 1e-10
    record(
        "kernel_witness",
        {
            "missing_dimensions": 80,
            "projector_trace": trace,
            "gaussian_identity_lower_bound": 80 / 384,
        },
        {"u": u, "v": v, "uv": projection, "feature_a": a, "feature_b": b, "direct_delta": direct},
    )


def test_bounds():
    x = torch.linspace(-8, 8, 4097, dtype=torch.float64, requires_grad=True)
    payload = {}
    for form, bounds in (("rational", (1, 1)), ("trig", (1, 2))):
        actual = [
            torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
            for y in pair_basis(x, form).chunk(2)
        ]
        if form == "rational":
            t = x.abs() / (1 + x.abs())
            expected = [x.sign() * (2 * t - t**2), t**2 * (3 - 2 * t)]
        else:
            expected = [x.sin(), x.cos() - 1]
        for a, b, bound in zip(actual, expected, bounds):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
            assert a.abs().max() <= bound + 1e-11
        payload[form] = {"actual": actual, "reference": expected, "bounds": list(bounds)}
    record("bounds_derivatives", {"points": 4097, "atol": 1e-11, "rtol": 1e-11}, payload)


def whole(model, x, use_reference):
    if not use_reference:
        return model(x)
    z = F.linear(x, model.up.weight)
    extra = F.gelu(z.float()).to(z.dtype) if model.form == "raw_gelu" else reference(z, model.form)
    return F.linear(torch.cat((x.to(z.dtype), extra), -1), model.down.weight)


@pytest.mark.parametrize("form", FORMS[:-1])
def test_cuda(form):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = InputBasisFFN(form)
    with torch.no_grad():
        model.down.weight[:, 384:].normal_(0, 0.02, generator=torch.Generator().manual_seed(9843))
    model.cuda()
    generator = torch.Generator().manual_seed(9842)
    x0 = torch.randn(3, 2, 384, generator=generator).transpose(0, 1).cuda()
    cotangent = torch.randn(3, 2, 384, generator=generator).transpose(0, 1).cuda()
    payload = {}
    for precision, tol in (("fp32", 1e-5), ("bf16", 0.02)):
        results = {}
        for mode in ("native", "reference", "checkpoint"):
            x = x0.detach().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                y = (
                    checkpoint(model, x, use_reentrant=False)
                    if mode == "checkpoint"
                    else whole(model, x, mode == "reference")
                )
            grad = torch.autograd.grad(y, (x, *model.parameters()), cotangent)
            results[mode] = [y, *grad]
        for a, b, c in zip(results["native"], results["reference"], results["checkpoint"]):
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)
            torch.testing.assert_close(a, c, atol=0, rtol=0)
            assert torch.isfinite(a).all()
        payload[precision] = results
    record(
        f"cuda_{form}",
        {"fp32_atol_rtol": 1e-5, "bf16_atol_rtol": 0.02, "checkpoint_exact": True},
        payload,
    )
