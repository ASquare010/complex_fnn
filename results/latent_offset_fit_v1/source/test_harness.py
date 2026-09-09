"""Eight isolated H085 qualification cases, with zero optimizer updates."""

import copy
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from results.latent_activation_v1.source.model import curve
from results.latent_offset_fit_v1.source import study as s
from src.core.benchmark import autocast
from src.core.native_recompute_audit import finite_tree


def record(name, value):
    s.previous.durable_json(s.ROOT / "preflight" / (name+".json"), {"status": "PASS", "optimizer_updates": 0, **value})


def execute(model, x, precision, recompute=False):
    x = x.detach().clone().requires_grad_(True)
    with autocast(x.device.type, precision):
        output = checkpoint(model, x, use_reentrant=False, preserve_rng_state=True) if recompute else model(x)
        work = output if output.dtype == torch.float64 else output.float()
        probe = torch.linspace(-0.7, 1.1, work.numel(), device=x.device, dtype=work.dtype).reshape_as(work)
        loss = (work*probe).mean()
    params = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, (x, *params.values()))
    return {"output": output.detach().cpu().clone(),
            "gradients": {name: value.detach().cpu().clone() for name, value in zip(("input", *params), gradients)}}


def test_counts_initialization_and_optimizer():
    torch.set_num_threads(4)
    evidence = {}
    with s.bound(), torch.no_grad():
        for seed in s.SEEDS:
            models = {form: s.make_model(form, seed) for form in s.FORMS}
            x = torch.linspace(-1, 1, 768).reshape(2, 384)
            plain_output = models["plain"](x)
            for form, model in models.items():
                params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
                if form in s.FORMS[:6]:
                    assert torch.equal(model(x), plain_output)
                    assert all(torch.equal(value, models["plain"].state_dict()[name])
                               for name, value in model.state_dict().items() if ".curve." not in name)
                groups = s.optimizer_groups(model, form, 0.001)
                ids = [id(p) for g in groups for p in g["params"]]
                assert len(ids) == len(set(ids)) == len(params)
                assert set(ids) == {id(p) for p in params.values()}
                lookup = {id(p): g for g in groups for p in g["params"]}
                for name, p in params.items():
                    expected = 1.0
                    if form in s.FORMS[:6] and ".curve." not in name:
                        expected = 2048/96 if name == "down.second.weight" else 4.0
                        if form.endswith("_first_lr") and name.endswith("first.weight"):
                            expected *= 5/8
                    elif form.startswith("narrow_") and name == "down.weight":
                        expected = 1536/456 if "gelu" in form else 1024/304
                    assert lookup[id(p)]["lr_scale"] == expected and lookup[id(p)]["lr"] == 0.001*expected
                    assert lookup[id(p)]["weight_decay"] == 0
                if form in ("latent_gain", "latent_offset", "latent_offset_first_lr"):
                    fixed = "theta_b" if form == "latent_gain" else "theta_a"
                    active = "theta_a" if form == "latent_gain" else "theta_b"
                    assert sum(v.numel() for v in buffers.values()) == 24
                    assert all(name.endswith(fixed) and not v.requires_grad and torch.count_nonzero(v) == 0 for name, v in buffers.items())
                    assert sum(p.numel() for name, p in params.items() if name.endswith(active)) == 24
                assert sum(p.numel() for p in params.values()) == s.COUNTS[form]
                evidence[f"{form}_s{seed}"] = {"parameters": s.COUNTS[form], "buffers": sum(v.numel() for v in buffers.values()),
                                              "groups": s.base.group_summary(groups)}
    record("counts_optimizer", {"models": evidence})


@pytest.mark.parametrize("form", ("latent_gain", "latent_offset"))
@pytest.mark.parametrize("device,precision,shape", (("cpu", "fp32", (2, 7, 384)), ("cuda", "bf16", (16, 128, 384))))
def test_partial_control_and_checkpoint_fidelity(form, device, precision, shape):
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float64 if device == "cpu" else torch.float32
    x = torch.randn(shape, generator=torch.Generator().manual_seed(9384), dtype=dtype).to(device)
    reference = s.make_model("latent_affine", 17).to(device=device, dtype=dtype)
    candidate = s.make_model(form, 17).to(device=device, dtype=dtype)
    active = "theta_a" if form == "latent_gain" else "theta_b"
    with torch.no_grad():
        for model in (reference, candidate):
            for projection in (model.up, model.gate, model.down):
                getattr(projection.curve, active).copy_(torch.linspace(-0.4, 0.3, 8, device=device, dtype=dtype))
    full = execute(reference, x, precision)
    eager = execute(candidate, x, precision)
    recomputed = execute(copy.deepcopy(candidate), x, precision, recompute=True)
    assert torch.equal(full["output"], eager["output"]) and torch.equal(eager["output"], recomputed["output"])
    for name, value in eager["gradients"].items():
        assert torch.equal(value, full["gradients"][name]) and torch.equal(value, recomputed["gradients"][name])
    payload = {"full_affine": full, "partial_eager": eager, "partial_checkpoint": recomputed}
    assert finite_tree(payload)
    label = device+"_"+form
    path = s.ROOT / "preflight" / (label+".pt")
    s.previous.save_tensor(payload, path)
    record(label, {"tensor_file": path.as_posix(), "tensor_sha256": s.previous.sha(path),
                   "retained_gradient_pairs": len(eager["gradients"]), "checkpoint_gradient_pairs": len(eager["gradients"])})


def test_active_control_finite_differences():
    calls = {}
    for family in ("gain", "offset"):
        z = torch.linspace(-1.3, 1.5, 32, dtype=torch.float64).reshape(2, 16).requires_grad_(True)
        theta = torch.linspace(-0.4, 0.3, 4, dtype=torch.float64).requires_grad_(True)
        counter = [0]

        def function(x, t):
            counter[0] += 1
            zero = torch.zeros_like(t)
            return curve(x, t if family == "gain" else zero, zero if family == "gain" else t, "affine")

        assert torch.autograd.gradcheck(function, (z, theta), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
        calls[family] = counter[0]
    record("gradcheck", {"forward_calls": calls})


def test_fresh_data_contract():
    x = (2*torch.rand(16, 384, generator=torch.Generator().manual_seed(9813))-1)*math.sqrt(3)
    previous = (2*torch.rand(16, 384, generator=torch.Generator().manual_seed(9812))-1)*math.sqrt(3)
    assert not torch.equal(x, previous)
    assert torch.equal(x, (2*torch.rand(16, 384, generator=torch.Generator().manual_seed(9813))-1)*math.sqrt(3))
    y = torch.tensor([[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]])
    scaled, std = s.base.scale_targets(y, 2)
    y[-1] *= 100
    scaled2, std2 = s.base.scale_targets(y, 2)
    assert torch.equal(std, std2) and torch.equal(scaled[:2], scaled2[:2])
    p = s.previous.permutation()
    assert torch.equal(p.sort().values, torch.arange(384))
    streams = {}
    for seed in s.SEEDS:
        stream = torch.randint(65536, (600, 256), generator=torch.Generator().manual_seed(20000+seed))
        assert stream.min() >= 0 and stream.max() < 65536
        streams[str(seed)] = s.base.tensor_sha(stream)
    record("data_contract", {"fresh_seed": 9813, "probe_sha256": s.base.tensor_sha(x),
                              "permutation_sha256": s.base.tensor_sha(p), "streams": streams})


def test_selection_and_gates():
    errors = dict(zip(s.FORMS, (1.0, 0.97, 0.96, 0.74, 0.73, 0.74, 0.8, 0.8, 0.65, 0.6)))
    rate_rows = [{"task": t, "form": f, "seed": seed, "rate": rate, "heldout_mse": errors[f], "zero_mse": 1.0, "finite": True}
                 for t in s.TASKS for seed in s.SEEDS for f in s.FORMS for rate in s.RATES]
    rows = [dict(v) for v in rate_rows if v["rate"] == 0.003]
    good = s.decisions(rows, rate_rows)
    assert all(g["earns_full_model_resource_qualification"] for g in good["gates"].values())
    with pytest.raises(ValueError):
        s.decisions(rows[:-1], rate_rows)
    bad = copy.deepcopy(rate_rows)
    bad[0]["finite"] = False
    assert not any(g["earns_full_model_resource_qualification"] for g in s.decisions(rows, bad)["gates"].values())
    bad_rows = copy.deepcopy(rows)
    for v in bad_rows:
        if v["form"] == "full_gelu":
            v["heldout_mse"] = 1.0
    assert s.decisions(bad_rows, rate_rows)["scientific_verdict"] == "INCONCLUSIVE_ASSAY_FAILURE"
    bad = copy.deepcopy(rate_rows)
    for v in bad:
        if v["form"] == "latent_offset" and v["rate"] == 0.001:
            v["heldout_mse"] = 0.995
    decision = s.decisions(rows, bad)
    assert not decision["gates"]["latent_offset"]["tests"]["matched_rate_offset_benefit"]
    for reference, key in (("latent_gain", "one_percent_over_equal_count_gain"),
                           ("plain_first_lr", "one_percent_over_fixed_lr_plain"),
                           ("narrow_gelu", "beats_both_calibrated_narrows")):
        bad_rows = copy.deepcopy(rows)
        for v in bad_rows:
            if v["form"] == reference:
                v["heldout_mse"] = 0.7
        assert not s.decisions(bad_rows, rate_rows)["gates"]["latent_offset"]["tests"][key]
    records = [{"selection_mse": 1.0, "base_rate": rate} for rate in reversed(s.RATES)]
    assert min(records, key=lambda r: (r["selection_mse"], r["base_rate"]))["base_rate"] == 0.001
    record("gates", {"selected_rows": 120, "allocated_cells": 240, "matched_rate_robustness_tested": True})
