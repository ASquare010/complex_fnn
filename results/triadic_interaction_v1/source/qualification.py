"""Thirty prespecified H100 checks; save compared tensors before training."""

import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from results.blast_operator_recovery_v1.source.storage import read, record_observation, sha
from results.triadic_interaction_v1.source.model import (
    FORMS,
    TASKS,
    InteractionFFN,
    counts,
    interaction,
    target_features,
)

ROOT = Path("results/triadic_interaction_v1/qualification")
NEW = ("cubic_ridge", "cp2", "cp3", "bounded_cp3")


def record(name, meta, pairs=()):
    saved = []
    for a, b, tol in pairs:
        a, b = a.detach().cpu(), b.detach().cpu()
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        saved.append({"actual": a, "expected": b, "tolerance": tol})
    record_observation(ROOT, name, name.split("_")[0], meta, saved if saved else None)


def reference(form, *values):
    dtype = values[0].dtype
    values = [a if dtype == torch.float64 else a.float() for a in values]
    if form == "cubic_ridge":
        u = values[0]
        output = (u**3 - 3 * u) / math.sqrt(6)
    else:
        stacked = torch.stack(values, 0)
        output = stacked.prod(0)
        if form == "bounded_cp3":
            output = 4 * output / (1 + stacked.square().sum(0))
    return output.to(dtype)


def test_count(form):
    actual = counts(form)
    expected = (
        6144
        if form == "feature_aware"
        else 1182080
        if form == "full_swiglu"
        else 1181568
        if form.startswith("full_")
        else 351200
        if form in ("narrow_swiglu", "cp2")
        else 351276
        if form in ("cp3", "bounded_cp3")
        else 351050
        if form == "starrelu"
        else 351048
    )
    assert actual["parameters"] == expected
    assert actual["parameters"] == sum(
        actual[k] for k in ("matrix_parameters", "bias_parameters", "activation_parameters")
    )
    assert actual["fixed_buffer_entries"] == (
        6144 if form == "feature_aware" else 1 if form == "fixed_mix" else 0
    )
    model = InteractionFFN(form)
    assert model.down.weight.count_nonzero() == 0
    assert model.down.bias is None or model.down.bias.count_nonzero() == 0
    record(f"count_{form}", actual)


def test_formula(form):
    gen = torch.Generator().manual_seed(9903)
    number = 1 if form == "cubic_ridge" else 2 if form == "cp2" else 3
    values = tuple(
        torch.randn(2, 7, dtype=torch.float64, generator=gen).requires_grad_()
        for _ in range(number)
    )
    a, b = interaction(form, *values), reference(form, *values)
    ga = torch.autograd.grad(a.sum(), values)
    gb = torch.autograd.grad(b.sum(), values)
    assert torch.autograd.gradcheck(
        lambda *v: interaction(form, *v), values, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True
    )
    record(
        f"formula_{form}",
        {"gradcheck": True},
        [(aa, bb, 1e-11) for aa, bb in zip((a, *ga), (b, *gb))],
    )


def test_bound():
    gen = torch.Generator().manual_seed(9904)
    values = (
        torch.randn(4097, 3, dtype=torch.float64, generator=gen)
        * torch.logspace(-4, 4, 4097, dtype=torch.float64)[:, None]
    ).requires_grad_()
    u, v, w = values.unbind(-1)
    output = interaction("bounded_cp3", u, v, w)
    grad = torch.autograd.grad(output.sum(), values)[0]
    r2 = values.square().sum(-1)
    expected = (
        torch.stack(
            (
                4 * v * w * (1 - u * u + v * v + w * w),
                4 * u * w * (1 + u * u - v * v + w * w),
                4 * u * v * (1 + u * u + v * v - w * w),
            ),
            -1,
        )
        / (1 + r2[:, None]).square()
    )
    assert grad.abs().max() <= 2 + 1e-11
    assert (output.abs() <= 4 * r2.sqrt() / (3 * math.sqrt(3)) + 1e-11).all()
    record(
        "bound_derivative",
        {"points": 4097, "partial_bound": 2, "gradient_norm_bound": math.sqrt(12)},
        [(grad, expected, 1e-11)],
    )


def test_known(data, task):
    model = InteractionFFN("feature_aware", task=task).double()
    with torch.no_grad():
        model.down.weight.copy_((data["rotation"] / data["scales"][task]).T)
    x = data["x"][:257].double()
    expected = target_features(x @ data["basis"], task) @ (data["rotation"] / data["scales"][task])
    record(
        f"known_{task}",
        {"privileged": True},
        [(model(x), expected, 1e-11), (model(x), data["targets"][task][:257].double(), 5e-6)],
    )


def test_representation(data, task):
    model = InteractionFFN("cp3", task=task).double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        basis = data["basis"].T
        model.up.weight[:16].copy_(basis)
        model.gate.weight[:16].copy_(basis if task != "product" else basis.roll(-1, 0))
        if task == "cubic":
            model.third.weight[:16].copy_(basis)
            model.gate.bias[:16].fill_(-math.sqrt(3))
            model.third.bias[:16].fill_(math.sqrt(3))
        else:
            model.third.bias[:16].fill_(1)
            if task == "quadratic":
                model.up.bias[:16].fill_(-1)
                model.gate.bias[:16].fill_(1)
        factor = math.sqrt(2) if task == "quadratic" else math.sqrt(6) if task == "cubic" else 1
        model.down.weight[:, :16].copy_((data["rotation"] / data["scales"][task]).T / factor)
    x = data["x"][:257].double().requires_grad_()
    q = x @ data["basis"]
    expected = target_features(q, task) @ (data["rotation"] / data["scales"][task])
    actual = model(x)
    ga = torch.autograd.grad(actual.sum(), x)[0]
    gb = torch.autograd.grad(expected.sum(), x)[0]
    record(
        f"representation_{task}",
        {"privileged_construction": True, "used_features": 16},
        [(actual, expected, 1e-10), (ga, gb, 1e-10)],
    )


def test_cuda(form):
    model = InteractionFFN(form)
    with torch.no_grad():
        model.down.weight.normal_(0, 0.02, generator=torch.Generator().manual_seed(9905))
    model.cuda()
    gen = torch.Generator().manual_seed(9906)
    x0 = torch.randn(3, 2, 384, generator=gen).transpose(0, 1).cuda()
    cotangent = torch.randn(3, 2, 384, generator=gen).transpose(0, 1).cuda()
    pairs = []
    for bf16, tol in ((False, 1e-5), (True, 0.02)):
        outputs = []
        for mode in ("native", "reference", "checkpoint"):
            x = x0.detach().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                if mode == "reference":
                    projections = [
                        F.linear(x, p.weight, p.bias)
                        for p in (model.up, model.gate, model.third)
                        if p is not None
                    ]
                    y = F.linear(reference(form, *projections), model.down.weight, model.down.bias)
                elif mode == "checkpoint":
                    y = checkpoint(model, x, use_reentrant=False)
                else:
                    y = model(x)
            grads = torch.autograd.grad(y, (x, *model.parameters()), cotangent)
            outputs.append((y, *grads))
        pairs += [(a, b, tol) for a, b in zip(outputs[0], outputs[1])]
        pairs += [(a, b, 0) for a, b in zip(outputs[0], outputs[2])]
    record(
        f"cuda_{form}",
        {"fp32_atol_rtol": 1e-5, "bf16_atol_rtol": 0.02, "checkpoint_exact": True},
        pairs,
    )


def run(data):
    assert not list(ROOT.iterdir())
    for form in FORMS:
        test_count(form)
    for form in NEW:
        test_formula(form)
    test_bound()
    for task in TASKS:
        test_known(data, task)
    for task in ("quadratic", "cubic", "product"):
        test_representation(data, task)
    for form in NEW:
        test_cuda(form)
    x = data["x"][:8].double()
    record(
        "initial_zero",
        {"forms": 13},
        [
            (InteractionFFN(f).double()(x), torch.zeros(8, 384, dtype=torch.float64), 0)
            for f in FORMS
        ],
    )
    files = list(ROOT.glob("*.json"))
    assert len(files) == 30
    pairs = 0
    for p in files:
        row = read(p)
        if "tensor_file" not in row:
            continue
        tensor = p.with_name(row["tensor_file"])
        assert sha(tensor) == row["tensor_sha256"]
        for item in torch.load(tensor, map_location="cpu", weights_only=True):
            a, b, tol = item["actual"], item["expected"], item["tolerance"]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert ((a - b).abs() <= tol + tol * b.abs()).all()
            pairs += 1
    return {"checks": 30, "saved_pairs_independently_rechecked": pairs, "status": "PASS"}
