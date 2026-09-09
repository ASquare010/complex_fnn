"""Eight H082 checks; finite-difference forwards but no optimizer updates."""

import copy
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from results.latent_activation_v1.source import model as m
from results.latent_activation_v1.source import study as s
from src.core.benchmark import autocast
from src.core.native_recompute_audit import finite_tree, tensor_record


def record(name, value):
    root = s.ROOT / "preflight"
    root.mkdir(exist_ok=True)
    s.durable_json(root / (name + ".json"), {"status": "PASS", "optimizer_updates": 0, **value})


def test_scalar_bounds_and_bezier_identity():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {}
    for family in ("affine", "tanh", "sine"):
        maximum_error = 0.0
        lo, hi = (1-2/(3*math.sqrt(3)), 1+2/(3*math.sqrt(3))) if family == "tanh" else (0.5, 1.5)
        for raw_a in (-40.0, -0.3, 0.0, 0.3, 40.0):
            for raw_b in (-40.0, 0.0, 40.0):
                a = torch.full((1,), raw_a, dtype=torch.float64)
                b = torch.full((1,), raw_b, dtype=torch.float64)
                scale = (math.log(4)*b.tanh()).exp()
                u = torch.cat((torch.linspace(-32, 32, 257, dtype=torch.float64),
                               torch.tensor([math.atanh(1/math.sqrt(3)), -math.atanh(1/math.sqrt(3)),
                                             math.pi/4, -math.pi/4], dtype=torch.float64)))
                z = (u*scale).reshape(-1, 1).requires_grad_(True)
                output = m.curve(z, a, b, family)
                derivative = torch.autograd.grad(output.sum(), z)[0]
                analytic = m.slope(z, a, b, family)
                torch.testing.assert_close(derivative, analytic, rtol=1e-10, atol=1e-11)
                assert derivative.min() >= lo-1e-11 and derivative.max() <= hi+1e-11
                assert finite_tree(output) and finite_tree(derivative)
                maximum_error = max(maximum_error, (derivative-analytic).abs().max().item())
                if family != "affine":
                    q = (z/scale).tanh() if family == "tanh" else (z/scale).sin()
                    t, amplitude = (1+q)/2, 0.5*a.tanh()*scale
                    bezier = amplitude*((1-t).square()-2*(1-t)*t+t.square())
                    torch.testing.assert_close(output-z, bezier, rtol=1e-10, atol=1e-11)
        results[family] = {"lower_slope_bound": lo, "upper_slope_bound": hi,
                           "analytic_max_error": maximum_error, "control_pairs": 15}
    record("scalar", {"families": results})


@pytest.mark.parametrize("family", ("affine", "tanh", "sine"))
def test_finite_differences(family):
    z = torch.linspace(-1.3, 1.5, 32, dtype=torch.float64).reshape(2, 16).requires_grad_(True)
    a = torch.linspace(-0.4, 0.3, 4, dtype=torch.float64).requires_grad_(True)
    b = torch.linspace(-0.3, 0.5, 4, dtype=torch.float64).requires_grad_(True)
    calls = [0]

    def function(*args):
        calls[0] += 1
        return m.curve(*args, family)

    assert torch.autograd.gradcheck(function, (z, a, b), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
    record("gradcheck_"+family, {"forward_calls": calls[0]})


def execute(model, x, precision, recompute=False):
    x = x.detach().clone().requires_grad_(True)
    with autocast(x.device.type, precision):
        out = checkpoint(model, x, use_reentrant=False, preserve_rng_state=True) if recompute else model(x)
        probe = torch.linspace(-0.7, 1.1, out.numel(), device=x.device).reshape_as(out)
        loss = (out.float()*probe).mean()
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, (x, *named.values()))
    return {"output": out.detach().cpu().clone(),
            "gradients": {n: g.detach().cpu().clone() for n, g in zip(("input", *named), gradients)}}


def test_identity_and_checkpoint_fidelity():
    rows = {}
    for device, precision, shape in (("cpu", "fp32", (2, 7, 384)), ("cuda", "bf16", (16, 128, 384))):
        torch.backends.cuda.matmul.allow_tf32 = False
        x = torch.randn(shape, generator=torch.Generator().manual_seed(9384)).to(device)
        plain = m.make_model("plain", 17).to(device)
        original = execute(plain, x, precision)
        for family in ("affine", "tanh", "sine"):
            candidate = m.make_model("latent_"+family, 17).to(device)
            initial = execute(candidate, x, precision)
            assert torch.equal(initial["output"], original["output"])
            assert all(torch.equal(initial["gradients"][n], g) for n, g in original["gradients"].items())
            if family != "affine":
                assert all(torch.count_nonzero(g) == 0 for n, g in initial["gradients"].items()
                           if n.endswith("theta_b"))
            with torch.no_grad():
                for projection in (candidate.up, candidate.gate, candidate.down):
                    projection.curve.theta_a.copy_(torch.linspace(-0.4, 0.3, 8, device=device))
                    projection.curve.theta_b.copy_(torch.linspace(-0.2, 0.4, 8, device=device))
            eager = execute(candidate, x, precision)
            recomputed = execute(copy.deepcopy(candidate), x, precision, recompute=True)
            assert torch.equal(eager["output"], recomputed["output"])
            assert eager["gradients"].keys() == recomputed["gradients"].keys()
            assert all(torch.equal(g, recomputed["gradients"][n]) for n, g in eager["gradients"].items())
            payload = {"identity_plain": original, "identity_curve": initial,
                       "nonzero_eager": eager, "nonzero_checkpoint": recomputed}
            assert finite_tree(payload)
            label = device+"_"+family
            path = s.ROOT / "preflight" / (label+".pt")
            path.parent.mkdir(exist_ok=True)
            s.save_tensor(payload, path)
            rows[label] = {"precision": precision, "shape": list(shape), "tensor_sha256": s.sha(path),
                           "identity_gradient_pairs": len(original["gradients"]),
                           "checkpoint_gradient_pairs": len(eager["gradients"]),
                           "tensor_file": path.as_posix()}
    record("fidelity", {"cases": rows, "tensor_files": 6})


def test_counts_shared_initialization_and_optimizer():
    evidence = {}
    for seed in s.SEEDS:
        with s.bound():
            models = {form: m.make_model(form, seed) for form in m.FORMS}
            x = torch.linspace(-1, 1, 2*384).reshape(2, 384)
            for form, model in models.items():
                assert sum(p.numel() for p in model.parameters()) == m.COUNTS[form]
                if form.startswith("latent_"):
                    assert torch.equal(model(x), models["plain"](x))
                    assert all(torch.equal(p, models["plain"].state_dict()[n]) for n, p in model.state_dict().items()
                               if ".curve." not in n)
                groups = s.base.optimizer_groups(model, form, 0.001)
                ids = [id(p) for g in groups for p in g["params"]]
                assert len(ids) == len(set(ids)) == len(list(model.parameters()))
                lookup = {id(p): g for g in groups for p in g["params"]}
                for n, p in model.named_parameters():
                    assert lookup[id(p)]["weight_decay"] == 0
                    if ".curve." in n:
                        assert lookup[id(p)]["lr_scale"] == 1
                if form.startswith("narrow_"):
                    expected = (1536 if "gelu" in form else 1024) / m.HIDDEN[form]
                    assert lookup[id(model.down.weight)]["lr_scale"] == expected
                if form == "plain" or form.startswith("latent_"):
                    assert lookup[id(model.up.first.weight)]["lr_scale"] == 4
                    assert lookup[id(model.down.second.weight)]["lr_scale"] == 2048/96
                evidence[f"{form}_s{seed}"] = {"parameters": m.COUNTS[form], "groups": s.base.group_summary(groups)}
    record("counts_optimizer", {"models": evidence})


def test_scaling_sampler_and_permutation():
    y = torch.tensor([[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]])
    scaled, std = s.base.scale_targets(y, 2)
    y[-1] *= 100
    scaled2, std2 = s.base.scale_targets(y, 2)
    assert torch.equal(std, std2) and torch.equal(scaled[:2], scaled2[:2])
    assert not torch.equal(scaled[:2].mean(0), torch.zeros(2))
    streams = {}
    for seed in s.SEEDS:
        a = torch.randint(65536, (600, 256), generator=torch.Generator().manual_seed(20000+seed))
        b = torch.randint(65536, (600, 256), generator=torch.Generator().manual_seed(20000+seed))
        assert torch.equal(a, b) and a.min() >= 0 and a.max() < 65536
        streams[str(seed)] = tensor_record(a)
    p = s.permutation()
    assert torch.equal(p, s.permutation()) and torch.equal(p.sort().values, torch.arange(384))
    record("data_contract", {"streams": streams, "permutation": tensor_record(p),
                              "neighbor_cross_group_fraction": ((p//48)!=(p.roll(-1)//48)).float().mean().item()})


def test_promotion_and_failure_rules():
    rows = [{"task": t, "form": f, "seed": seed, "finite": True, "zero_mse": 1.0,
             "heldout_mse": 0.7 if f in ("latent_tanh", "latent_sine") else 0.85 if f == "latent_affine"
             else 0.92 if f.startswith("narrow_") else 0.8 if f.startswith("full_") else 1.0}
            for t in s.TASKS for f in s.FORMS for seed in s.SEEDS]
    good = s.summarize(rows)
    assert good["assay_passed"] and all(g["earns_full_model_resource_qualification"] for g in good["gates"].values())
    bad = copy.deepcopy(rows)
    for r in bad:
        if r["form"] == "full_gelu":
            r["heldout_mse"] = 1.0
    assert s.summarize(bad)["scientific_verdict"] == "INCONCLUSIVE_ASSAY_FAILURE"
    bad = copy.deepcopy(rows)
    for r in bad:
        if r["form"] == "latent_tanh" and r["task"] == "oscillatory":
            r["heldout_mse"] = 1.06
    checks = s.summarize(bad)["gates"]["latent_tanh"]["tests"]
    assert checks["two_percent_over_plain"] and not checks["per_task_regression_cap_vs_plain"]
    assert not checks["per_task_regression_cap_vs_zero"]
    bad = copy.deepcopy(rows)
    next(r for r in bad if r["form"] == "plain")["finite"] = False
    assert not any(g["earns_full_model_resource_qualification"] for g in s.summarize(bad)["gates"].values())
    with pytest.raises(ValueError):
        s.summarize(rows[:-1])
    record("gates", {"selection_rows": 96, "allocated_cells": 192})
