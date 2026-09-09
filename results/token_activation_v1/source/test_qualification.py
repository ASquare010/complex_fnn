"""H067's frozen 21 checks, with durable per-case evidence and no optimizer updates."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from results.token_activation_v1.source.candidate import ResidualActivationFFN, mixed_product
from src.blockshuffle_ffn import BlockShuffleFFN
from src.core.native_recompute_audit import digest, finite_tree, read, sha, tensor_record, write_new

ROOT = Path("results/token_activation_v1")


@pytest.fixture(scope="session", autouse=True)
def frozen_source():
    protocol = read(ROOT / "protocol.json")
    assert sha("research/token_activation_plan.md") == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    assert all(sha(n) == h for n, h in protocol["sources"].items())


def record(name, value):
    write_new(ROOT / "checks" / f"{name}.json", value)


def comparison(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    a, b = left.contiguous(), right.contiguous()
    changed = (
        a.reshape(-1).view(torch.uint8).reshape(a.numel(), -1)
        != b.reshape(-1).view(torch.uint8).reshape(b.numel(), -1)
    ).any(-1)
    delta = b.double() - a.double()
    return {
        "exact": tensor_record(a) == tensor_record(b),
        "finite": finite_tree([a, b]),
        "mismatched_elements": int(changed.sum()),
        "max_abs_error": float(delta.abs().max()),
        "relative_l2_error": float(delta.norm() / a.double().norm().clamp_min(1e-30)),
        "reference": tensor_record(a),
        "candidate": tensor_record(b),
    }


def setup(kind, device, seed):
    torch.manual_seed(seed)
    shape, hidden, groups = ((2, 7, 24), 48, 3) if device == "cpu" else ((16, 128, 384), 2048, 8)
    baseline = BlockShuffleFFN(shape[-1], hidden, groups)
    for name in ("up", "gate", "down"):
        getattr(baseline, name).initialize(seed, name, residual_scale=0.25 if name == "down" else 1)
    candidate = ResidualActivationFFN(shape[-1], hidden, groups, adaptive=kind == "dynamic")
    with torch.no_grad():
        for name, p in candidate.named_parameters():
            if name in baseline.state_dict():
                p.copy_(baseline.state_dict()[name])
    for module in (baseline, candidate):
        module.recompute_gate = True
        module.gate_recompute_method = "checkpoint"
        module.to(device).train()
    generator = torch.Generator().manual_seed(80000 + seed)
    x = torch.randn(shape, generator=generator)
    incoming = torch.randn(shape, generator=generator)
    return baseline, candidate, x, incoming


def execute(module, x, incoming, device):
    x = x.to(device).clone().requires_grad_()
    before = digest(module.state_dict())
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        output = module(x)
    gradients = torch.autograd.grad(
        output, (x, *module.parameters()), incoming.to(device=device, dtype=output.dtype)
    )
    assert before == digest(module.state_dict()) and finite_tree([output, *gradients])
    names = ["input", *dict(module.named_parameters())]
    return {
        "output": output.detach().cpu(),
        "gradients": {n: g.detach().cpu() for n, g in zip(names, gradients)},
    }


@pytest.mark.parametrize("kind", ["static", "dynamic"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", [17, 29, 43])
def test_zero_initialization(kind, device, seed):
    assert device != "cuda" or torch.cuda.is_available()
    baseline, candidate, x, incoming = setup(kind, device, seed)
    a, b = execute(baseline, x, incoming, device), execute(candidate, x, incoming, device)
    shared = {
        "output": comparison(a["output"], b["output"]),
        **{n: comparison(g, b["gradients"][n]) for n, g in a["gradients"].items()},
    }
    added = {
        n: {"norm": float(g.double().norm()), "finite": finite_tree(g), "tensor": tensor_record(g)}
        for n, g in b["gradients"].items()
        if n not in a["gradients"]
    }
    row = {
        "case": f"zero_{kind}_{device}_{seed}",
        "seed": seed,
        "kind": kind,
        "device": device,
        "input": tensor_record(x),
        "incoming": tensor_record(incoming),
        "state_hash": digest(candidate.state_dict()),
        "shared": shared,
        "added_gradients": added,
        "all_shared_exact": all(v["exact"] for v in shared.values()),
        "new_gradients_finite_nonzero": all(v["finite"] and v["norm"] > 0 for v in added.values()),
        "optimizer_updates": 0,
    }
    record(row["case"], row)
    assert row["all_shared_exact"] and row["new_gradients_finite_nonzero"]


@pytest.mark.parametrize("kind", ["static", "dynamic"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonzero_checkpoint(kind, device):
    _, candidate, x, incoming = setup(kind, device, 17)
    with torch.no_grad():
        if candidate.adaptive:
            candidate.router.bias.fill_(0.4)
            candidate.router.weight.copy_(
                torch.linspace(-0.02, 0.02, x.shape[-1], device=device).reshape(1, -1)
            )
        else:
            candidate.gate_bias.fill_(0.4)
    candidate.recompute_gate = False
    a = execute(candidate, x, incoming, device)
    candidate.recompute_gate = True
    b = execute(candidate, x, incoming, device)
    checks = {
        "output": comparison(a["output"], b["output"]),
        **{n: comparison(g, b["gradients"][n]) for n, g in a["gradients"].items()},
    }
    row = {
        "case": f"nonzero_{kind}_{device}",
        "input": tensor_record(x),
        "incoming": tensor_record(incoming),
        "state_hash": digest(candidate.state_dict()),
        "comparisons": checks,
        "all_exact": all(c["exact"] for c in checks.values()),
        "optimizer_updates": 0,
    }
    record(row["case"], row)
    assert row["all_exact"]


def test_counts():
    rows = {}
    for kind in ("static", "dynamic"):
        model = ResidualActivationFFN(384, 2048, 8, adaptive=kind == "dynamic")
        count = sum(p.numel() for p in model.parameters())
        expected = 350208 + (385 if kind == "dynamic" else 1)
        assert count == expected and 1 - count * 8 / 9437184 >= 0.7
        rows[kind] = {
            "per_layer": count,
            "ffn_total": count * 8,
            "total_model": 6297984 + count * 8,
            "ffn_reduction_percent": 100 * (1 - count * 8 / 9437184),
        }
    record("counts", {"status": "PASS", "counts": rows})


def test_independent_gradcheck():
    generator = torch.Generator().manual_seed(81017)
    shapes = ((2, 4), (2, 4), (2, 3), (1, 3), (1,))
    args = tuple(
        (torch.randn(shape, generator=generator, dtype=torch.float64) * 0.3).requires_grad_()
        for shape in shapes
    )

    def fn(u, v, x, w, b):
        alpha = 0.25 * torch.tanh(F.linear(x, w, b))
        return mixed_product(u, v, alpha)

    passed = torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
    record(
        "gradcheck",
        {"status": "PASS" if passed else "FAIL", "input_router_independent_gradcheck": passed},
    )
    assert passed


def test_envelope_and_partial_derivative():
    v = torch.linspace(-64, 64, 1025, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    r = torch.linspace(-32, 32, 17, dtype=torch.float64).reshape(1, -1)
    alpha = 0.25 * r.tanh()
    multiplier = 1 + alpha * v.sigmoid()
    value = v + alpha * F.silu(v)
    derivative = 1 + alpha * (v.sigmoid() + v * v.sigmoid() * (1 - v.sigmoid()))
    partial = torch.autograd.grad(value.sum(), v)[0]
    torch.testing.assert_close(partial, derivative.sum(-1, keepdim=True), atol=1e-12, rtol=1e-12)
    bound = 1 - 0.25 * (1 + 1 / math.e)
    assert multiplier.min() >= 0.75 and multiplier.max() <= 1.25 and derivative.min() >= bound
    record(
        "bounds",
        {
            "status": "PASS",
            "sampled_gain_min": float(multiplier.min().detach()),
            "sampled_gain_max": float(multiplier.max().detach()),
            "partial_derivative_lower_bound": bound,
            "sampled_partial_derivative_min": float(derivative.min().detach()),
            "whole_network_lower_bound_claimed": False,
        },
    )


def test_structured_witness():
    module = ResidualActivationFFN(384, 2048, 8, adaptive=True).double()
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        for name, input_channel in (("up", 0), ("gate", 1), ("down", 0)):
            projection = getattr(module, name)
            projection.first.weight[0, 0, input_channel] = 1
            projection.second.weight[0, 0, 0] = 1
        module.router.weight[0, 2] = 1
    x = torch.zeros(65, 384, dtype=torch.float64)
    x[:, 0] = 1
    x[:, 1] = 1
    x[:, 2] = torch.linspace(-2, 2, 65, dtype=torch.float64)
    x.requires_grad_()
    y = module(x)
    expected = F.silu(x[:, 0]) * x[:, 1] + 0.25 * x[:, 2].tanh() * F.silu(x[:, 0]) * F.silu(x[:, 1])
    torch.testing.assert_close(y[:, 0], expected, rtol=2e-14, atol=2e-14)
    assert torch.count_nonzero(y[:, 1:]) == 0
    gradient = torch.autograd.grad(y[:, 0].sum(), x)[0][:, 2]
    expected_gradient = 0.25 * (1 - x[:, 2].tanh().square()) * F.silu(x[:, 0]) * F.silu(x[:, 1])
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-13, atol=2e-14)
    artifact = ROOT / "checks/witness_tensors.pt"
    assert not artifact.exists()
    torch.save(
        {
            "inputs": x.detach(),
            "outputs": y.detach(),
            "expected": expected.detach(),
            "router_coordinate_gradient": gradient.detach(),
            "expected_gradient": expected_gradient.detach(),
        },
        artifact,
    )
    record(
        "witness",
        {
            "status": "PASS",
            "shape": [384, 2048, 8],
            "parameters": sum(p.numel() for p in module.parameters()),
            "max_value_error": float((y[:, 0] - expected).detach().abs().max()),
            "max_derivative_error": float((gradient - expected_gradient).detach().abs().max()),
            "artifact_sha256": sha(artifact),
            "construction_only_not_learning": True,
        },
    )


def test_residue_identities():
    epsilon = 1e-6
    errors = []
    for n in (-2, -1, 0, 1):
        pole = 1j * np.pi * (n + 0.5)
        residue = 0.5 * epsilon * (np.tanh(pole + epsilon) - np.tanh(pole - epsilon))
        errors.append(abs(residue - 1))
        for slope in (-6.0, -2.0, 2.0, 6.0):
            a, b = 0.7, -0.4

            def fn(z):
                return (a * z + b) * (slope * z) / (1 + np.exp(-slope * z))

            actual = 0.5 * epsilon * (fn(pole + epsilon) - fn(pole - epsilon))
            expected = pole * (a * pole + b)
            errors.append(abs(actual - expected))
    maximum = float(max(errors))
    record(
        "residues",
        {
            "status": "PASS" if maximum < 1e-5 else "FAIL",
            "max_numerical_residue_error": maximum,
            "spot_checks": len(errors),
            "numerical_check_is_not_proof": True,
        },
    )
    assert maximum < 1e-5
