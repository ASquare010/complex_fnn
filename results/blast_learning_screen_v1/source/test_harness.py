"""Nine H081 preflight checks; 24 explicitly allocated temporary updates."""

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from results.blast_learning_screen_v1.source import study as s
from src.core.native_recompute_audit import digest, finite_tree
from src.core.optimization import group_summary, initialize_dense_width, parameter_groups


def record(name, value):
    root = s.ROOT / "preflight"
    root.mkdir(exist_ok=True)
    s.write_json(root / (name + ".json"), {"status": "PASS", **value})


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {}
    for form in s.FORMS:
        mc, tc = s.configuration(form, 0.0006)
        model = s.make_model(mc, 17, s.MODES[form])
        initialize_dense_width(model, tc)
        result[form] = model, mc, tc
    return result


def test_counts_and_initialization(models):
    common = None
    evidence = {}
    for form, (model, mc, tc) in models.items():
        assert sum(p.numel() for b in model.blocks for p in b.ffn.parameters()) == mc.unique_ffn_parameters
        assert sum(p.numel() for p in model.parameters()) == mc.total_parameters
        assert mc.unique_ffn_parameters == (9437184 if form.startswith("full_") else 2801664)
        assert s.operation_counts(mc)["ffn_matrix_forward_flops_per_token"] == 2 * mc.unique_ffn_parameters
        shared = {n: p for n, p in model.state_dict().items() if ".ffn." not in n}
        if common is None:
            common = shared
        assert shared.keys() == common.keys() and all(torch.equal(p, common[n]) for n, p in shared.items())
        if form not in s.NEW_FORMS:
            old = s.read(f"results/ungated_lm_screen_v1/runs/{form}_lr600/initial_state_signature.json")
            assert digest(model.state_dict()) == old["weights"]
            old_mc, old_tc = s.previous.configuration(form, 0.0006)
            assert asdict(mc) == asdict(old_mc) and asdict(tc) == asdict(old_tc)
        if form.startswith("blast_"):
            assert digest(model.blocks[0].ffn.state_dict()) != digest(model.blocks[1].ffn.state_dict())
            for layer in (0, 7):
                for name in ("up", "gate", "down"):
                    module = getattr(model.blocks[layer].ffn, name)
                    if module is not None:
                        expected = s.BlastLinear(module.input_width, module.output_width, seed=17,
                                                 name=f"blocks.{layer}.ffn.{name}",
                                                 residual_scale=0.25 if name == "down" else 1.0)
                        assert digest(expected.state_dict()) == digest(module.state_dict())
        evidence[form] = {"ffn_parameters": mc.unique_ffn_parameters, "total_parameters": mc.total_parameters,
                          "initial_weights": digest(model.state_dict()), "shared_weights": digest(shared)}
    assert evidence["plain"]["initial_weights"] == evidence["plain_calibrated"]["initial_weights"]
    record("counts_initialization", {"forms": evidence, "optimizer_updates": 0})


def test_optimizer_coverage_and_scale_identities(models):
    evidence = {}
    for form, (model, mc, tc) in models.items():
        groups = s.parameter_groups(model, tc)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        lookup = {id(p): g for g in groups for p in g["params"]}
        entries = s.calibration(model)
        if form not in s.NEW_FORMS:
            assert group_summary(groups) == group_summary(parameter_groups(model, tc))
        else:
            assert len(entries) == (48 if form == "plain_calibrated" else 72 if form == "blast_swiglu" else 48)
            for name, p in model.named_parameters():
                group = lookup[id(p)]
                if name in entries:
                    e = entries[name]
                    k, scale = e["factors"], e["lr_scale"]
                    assert group["lr_scale"] == scale and group["lr"] == tc.learning_rate * scale
                    assert math.isclose(scale / e["initial_rms"], 1 / (k * e["dense_sigma"]), rel_tol=1e-15)
                    assert math.isclose(group["weight_decay"] * scale * k, tc.weight_decay, rel_tol=1e-15)
                    a = tc.learning_rate * tc.weight_decay
                    expected_difference = a*a/4 if k == 2 else a*a/3-a*a*a/27
                    assert abs(((1-a/k)**k - (1-a)) - expected_difference) < 5e-16
                else:
                    assert group["lr_scale"] == 1 and group["weight_decay"] == (0.1 if p.ndim >= 2 else 0)
        evidence[form] = {"groups": group_summary(groups), "calibration": entries}
    record("optimizer", {"forms": evidence, "optimizer_updates": 0})


@pytest.mark.parametrize("form", s.NEW_FORMS)
@pytest.mark.parametrize("device,precision", (("cpu", "fp32"), ("cuda", "bf16")))
def test_full_model_observer_checkpoint_fidelity(form, device, precision, tmp_path):
    batch, length = (2, 16) if device == "cpu" else (16, 128)
    raw = torch.from_numpy(np.load(s.CACHE / "train.npy")[:batch*(length+1)].astype("int64"))
    raw = raw.reshape(batch, length+1).to(device)
    x, y = raw[:, :-1], raw[:, 1:]
    mc, tc = s.configuration(form, 0.0006)
    native = s.make_model(mc, 17, "none").to(device).train()
    initialize_dense_width(native, tc)
    observer = s.Observer(tmp_path, tc, s.MODES[form])
    observed = observer.constructor(mc, 17).to(device).train()
    observer.initialize(observed, tc)
    assert digest(native.state_dict()) == digest(observed.state_dict())
    opts = [torch.optim.AdamW(s.parameter_groups(m, tc), lr=tc.learning_rate,
                             betas=(0.9, 0.95), eps=1e-8) for m in (native, observed)]
    rows = []
    for step in range(2):
        losses = []
        for model, opt in zip((native, observed), opts):
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group["lr"] = s.trainer.learning_rate(step, tc) * group["lr_scale"]
            with s.benchmark.autocast(device, precision):
                loss = model.loss(x, y)
            loss.backward()
            losses.append(loss.item())
        assert losses[0] == losses[1]
        gradient_pairs = 0
        for a, b in zip(native.parameters(), observed.parameters()):
            assert a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad)
            gradient_pairs += 1
        norms = [s.previous.BASE_CLIP(native.parameters(), 1, error_if_nonfinite=True),
                 observer.clip(observed.parameters(), 1, error_if_nonfinite=True)]
        assert torch.equal(norms[0], norms[1])
        for opt in opts:
            opt.step()
        assert digest(native.state_dict()) == digest(observed.state_dict())
        assert digest(opts[0].state_dict()) == digest(opts[1].state_dict())
        assert finite_tree(native.state_dict()) and finite_tree(opts[0].state_dict())
        rows.append({"loss": losses[0], "preclip_norm": norms[0].item(),
                     "gradient_pairs_exact": gradient_pairs, "weights": digest(native.state_dict()),
                     "optimizer": digest(opts[0].state_dict())})
    assert observer.pending is None and len(observer.steps) == 2
    record(f"fidelity_{device}_{form}", {"device": device, "precision": precision, "steps": rows,
                                        "optimizer_updates": 4, "training_targets": 4 * batch * length})


def test_selection_and_failure_gates():
    rows = {cell: {"status": "SCREENED", "all_finite": True, "validation_loss":
                  5.8 if spec["form"] in s.NEW_FORMS else 5.9 if spec["form"].startswith("narrow_") else 6.0,
                  "ffn_reduction_percent": 70.3125, "peak_allocated_vram_bytes": 100}
            for cell, spec in s.CELLS.items()}
    selected, gates = s.select_and_gates(rows)
    assert all(c.endswith("lr300") for c in selected.values())
    assert all(all(g.values()) for g in gates.values())
    for cell, spec in s.CELLS.items():
        if spec["form"] == "blast_gelu":
            rows[cell]["validation_loss"] = 6.1
        if spec["form"] == "blast_swiglu":
            rows[cell]["peak_allocated_vram_bytes"] = 111
    _, gates = s.select_and_gates(rows)
    assert not gates["blast_gelu"]["nll_within_one_percent_full_gelu"]
    assert not gates["blast_swiglu"]["memory_within_ten_percent_full_swiglu"]
    rows["full_gelu_lr300"]["all_finite"] = False
    assert not any(g["all_cells_complete_finite"] for g in s.select_and_gates(rows)[1].values())
    rows.pop("plain_lr300")
    with pytest.raises(ValueError):
        s.select_and_gates(rows)
    record("selection", {"optimizer_updates": 0, "allocated_cells": len(s.CELLS)})
