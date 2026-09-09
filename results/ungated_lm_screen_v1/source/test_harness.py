"""Five H076 integration checks; only the explicit narrow fidelity check updates weights."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from results.ungated_lm_screen_v1.source import study as s
from src.core.config import ModelConfig
from src.core.native_recompute_audit import digest, read
from src.core.optimization import initialize_dense_width, parameter_groups


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(4)
    result = {}
    for form in s.FORMS:
        mc, tc = s.configuration(form, 0.0006)
        model = s.make_model(mc, 17, s.MODES[form])
        initialize_dense_width(model, tc)
        result[form] = (model, mc, tc)
    return result


def test_config_counts_and_operation_accounting(models):
    previous = read("results/ungated_resource_recovery_v1/result.json")
    assert s.MODES == previous["selected_modes"]
    for form, (model, mc, tc) in models.items():
        count = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
        assert count == mc.unique_ffn_parameters
        assert sum(p.numel() for p in model.parameters()) == mc.total_parameters
        assert s.operation_counts(mc)["ffn_matrix_forward_flops_per_token"] == 2 * count
        assert (
            tc.steps == 200
            and tc.batch_size == 16
            and tc.eval_batches == 158
            and tc.log_every == 50
        )
        assert all(b.recompute_scope == "block" for b in model.blocks)
    with pytest.raises(ValueError):
        ModelConfig(variant="blockshuffle_gelu").validate()


def test_initialization_shared_and_once(models):
    previous = read("results/ungated_resource_recovery_v1/result.json")
    common = None
    for form, (model, mc, tc) in models.items():
        origin = Path(previous["origins"][form + "_none"]["root"])
        assert (
            digest(model.state_dict())
            == read(origin / "workers" / (form + "_none") / "initial_signature.json")["weights"]
        )
        weights = {n: p for n, p in model.state_dict().items() if ".ffn." not in n}
        if common is None:
            common = weights
        assert all(torch.equal(p, common[n]) for n, p in weights.items())
        if form.startswith("narrow_"):
            raw = s.make_model(mc, 17, "none")
            ratio = (1024 if "swiglu" in form else 1536) / mc.ffn_width
            assert all(
                torch.equal(b.ffn.down.weight, a.ffn.down.weight * math.sqrt(ratio))
                for a, b in zip(raw.blocks, model.blocks)
            )
    assert all(
        torch.equal(p, models["plain"][0].state_dict()[n])
        for n, p in models["gelu_same"][0].state_dict().items()
    )


def test_optimizer_coverage_and_narrow_effective_decay(models):
    for form, (model, mc, tc) in models.items():
        groups = parameter_groups(model, tc)
        lookup = {id(p): g for g in groups for p in g["params"]}
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        if form.startswith("narrow_"):
            assert tc.ffn_decay_mode == "product"
            ratio = (1024 if "swiglu" in form else 1536) / mc.ffn_width
            for b in model.blocks:
                group = lookup[id(b.ffn.down.weight)]
                assert group["lr_scale"] == ratio and math.isclose(
                    group["weight_decay"] * ratio, 0.1, rel_tol=1e-15
                )
                assert group["lr"] == tc.learning_rate * ratio
        elif mc.variant.startswith("blockshuffle"):
            assert tc.ffn_decay_mode == "parameter" and tc.ffn_lr_mode == "fan_in"
            for b in model.blocks:
                assert lookup[id(b.ffn.up.first.weight)]["lr_scale"] == 4
                assert lookup[id(b.ffn.down.second.weight)]["lr_scale"] == mc.ffn_width / 96
                assert lookup[id(b.ffn.down.second.weight)]["weight_decay"] == 0.1


def test_observer_and_changed_narrow_update_fidelity(tmp_path):
    raw = torch.from_numpy(np.load(s.CACHE / "train.npy")[:34].astype("int64")).reshape(2, 17)
    x, y = raw[:, :-1], raw[:, 1:]
    updates = 0
    for form in ("narrow_swiglu", "narrow_gelu"):
        mc, tc = s.configuration(form, 0.0006)
        native = s.make_model(mc, 17, "none").train()
        initialize_dense_width(native, tc)
        folder = tmp_path / form
        folder.mkdir()
        observer = s.Observer(folder, tc, s.MODES[form])
        observed = observer.constructor(mc, 17).train()
        observer.initialize(observed, tc)
        assert digest(native.state_dict()) == digest(observed.state_dict())
        opts = [
            torch.optim.AdamW(
                parameter_groups(m, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
            )
            for m in (native, observed)
        ]
        for step in range(2):
            losses = []
            for model, opt in zip((native, observed), opts):
                for g in opt.param_groups:
                    g["lr"] = s.trainer.learning_rate(step, tc) * g["lr_scale"]
                opt.zero_grad(set_to_none=True)
                loss = model.loss(x, y)
                loss.backward()
                losses.append(loss.item())
            assert losses[0] == losses[1]
            assert all(
                torch.equal(a.grad, b.grad)
                for a, b in zip(native.parameters(), observed.parameters())
            )
            norms = [
                s.BASE_CLIP(native.parameters(), 1, error_if_nonfinite=True),
                observer.clip(observed.parameters(), 1, error_if_nonfinite=True),
            ]
            assert torch.equal(norms[0], norms[1])
            for opt in opts:
                opt.step()
                updates += 1
            assert digest(native.state_dict()) == digest(observed.state_dict())
            assert digest(opts[0].state_dict()) == digest(opts[1].state_dict())
        assert len(observer.steps) == 2 and observer.pending is None
    assert updates == 8


def test_selection_and_gate_failures():
    rows = {
        cell: {
            "status": "SCREENED",
            "all_finite": True,
            "validation_loss": 5.8
            if spec["form"].startswith("gelu_")
            else 5.9
            if spec["form"].startswith("narrow_")
            else 6.0,
            "ffn_reduction_percent": 70.3125,
            "peak_allocated_vram_bytes": 100,
        }
        for cell, spec in s.CELLS.items()
    }
    selected, gates = s.select_and_gates(rows)
    assert all(c.endswith("lr300") for c in selected.values()) and all(
        all(g.values()) for g in gates.values()
    )
    for cell, spec in s.CELLS.items():
        if spec["form"] == "gelu_same":
            rows[cell]["validation_loss"] = 6.1
        if spec["form"] == "gelu_matched":
            rows[cell]["peak_allocated_vram_bytes"] = 111
    _, gates = s.select_and_gates(rows)
    assert not gates["gelu_same"]["nll_within_one_percent_full_gelu"]
    assert not gates["gelu_same"]["nll_beats_narrow_gelu"]
    assert not gates["gelu_matched"]["memory_within_ten_percent_full_swiglu"]
    rows["full_gelu_lr300"]["all_finite"] = False
    assert not any(g["all_cells_complete_finite"] for g in s.select_and_gates(rows)[1].values())
    rows.pop("plain_lr300")
    with pytest.raises(ValueError):
        s.select_and_gates(rows)
