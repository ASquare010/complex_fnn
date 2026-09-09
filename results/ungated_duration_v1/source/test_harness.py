"""Five H078 preflight checks; no model forwards or optimizer updates."""

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from results.ungated_duration_v1.source import study as s
from src.core.data import TokenData
from src.core.native_recompute_audit import digest, read, tensor_record, write_new
from src.core.optimization import initialize_dense_width, parameter_groups


def test_recipes_initialization_counts_and_calibration():
    torch.set_num_threads(4)
    prior = read("results/ungated_lm_replication_v1/result.json")
    assert prior["replicated_primary_pass"] and len(s.CELLS) == 6
    initials, common = {}, None
    for cell, spec in s.CELLS.items():
        mc, tc = s.configuration(spec["form"], 17)
        old = prior["rows"][cell]
        assert asdict(mc) == old["model"]
        assert asdict(tc) == {**old["training"], "steps": 3200, "log_every": 800}
        assert s.MODES[spec["form"]] == old["execution_mode"]
        model = s.short.base.make_model(mc, 17, s.MODES[spec["form"]])
        metadata = initialize_dense_width(model, tc)
        record = {
            "weights": digest(model.state_dict()),
            "parameters": {n: tensor_record(v) for n, v in model.named_parameters()},
            "width_initialization": metadata,
            "execution_mode": s.MODES[spec["form"]],
        }
        assert record == read(
            Path("results/ungated_lm_replication_v1/runs") / cell / "initial_state_signature.json"
        )
        shared = {n: v for n, v in record["parameters"].items() if ".ffn." not in n}
        if common is None:
            common = shared
        assert shared == common
        count = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
        assert count == mc.unique_ffn_parameters == old["ffn_parameters"]
        assert (
            sum(p.numel() for p in model.parameters())
            == mc.total_parameters
            == old["total_parameters"]
        )
        assert s.operation_counts(mc)["ffn_matrix_forward_flops_per_token"] == 2 * count
        groups = parameter_groups(model, tc)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        gs = [
            {
                "weight_decay": g["weight_decay"],
                "lr_scale": g["lr_scale"],
                "parameter_count": sum(p.numel() for p in g["params"]),
            }
            for g in groups
        ]
        assert gs == old["optimizer_parameter_groups"]
        if spec["form"].startswith("narrow_"):
            raw = s.short.base.make_model(mc, 17, "none")
            ratio = (1024 if spec["form"] == "narrow_swiglu" else 1536) / mc.ffn_width
            lookup = {id(p): g for g in groups for p in g["params"]}
            for a, b in zip(raw.blocks, model.blocks):
                assert torch.equal(b.ffn.down.weight, a.ffn.down.weight * math.sqrt(ratio))
                g = lookup[id(b.ffn.down.weight)]
                assert g["lr_scale"] == ratio and math.isclose(
                    g["weight_decay"] * ratio, 0.1, rel_tol=1e-15
                )
            del raw
        initials[cell] = record
        del model, groups
    with pytest.raises(ValueError):
        s.configuration("gelu_matched", 29)
    with pytest.raises(ValueError):
        s.configuration("gelu_same", 17)
    write_new(s.ROOT / "qualification/initial_signatures.json", initials)


def test_readonly_sampler_and_validation_order():
    torch.set_num_threads(4)
    data = TokenData(s.CACHE, "cuda", 10017)
    old = read("results/ungated_lm_replication_v1/qualification/samplers.json")
    independent = torch.Generator(device="cuda").manual_seed(10017)
    record = {}
    for step in range(3200):
        starts = torch.randint(len(data.train) - 128, (16,), device="cuda", generator=independent)
        x, y = data.batch(16, 128)
        if step == 0:
            indices = starts[:, None] + torch.arange(128, device="cuda")
            assert torch.equal(x, data.train[indices]) and torch.equal(y, data.train[indices + 1])
            record.update(
                first_starts=tensor_record(starts),
                first_x=tensor_record(x),
                first_y=tensor_record(y),
            )
            assert all(
                record[k] == old["seeds"]["17"][k] for k in ("first_starts", "first_x", "first_y")
            )
        if step in (799, 3199):
            assert torch.equal(data.generator.get_state(), independent.get_state())
            record[f"state_after_{step + 1}"] = tensor_record(data.generator.get_state())
        if step == 799:
            assert record["state_after_800"] == old["seeds"]["17"]["state_after_800"]
            for cell in s.CELLS:
                q = read(
                    Path("results/ungated_lm_replication_v1/runs") / cell / "qualification.json"
                )
                assert record["state_after_800"] == q["sampling_rng"]
    batches = [y.detach().cpu() for _, y in data.validation(16, 128, 158)]
    actual = torch.cat([y.flatten() for y in batches])
    expected = torch.from_numpy(np.load(s.CACHE / "valid.npy").astype("int64"))[1:322689]
    assert torch.equal(actual, expected) and actual.numel() == 322688
    assert len(batches) == 158 and list(batches[-1].shape) == [9, 128]
    assert tensor_record(actual) == old["validation"]["ordered_targets"]
    write_new(
        s.ROOT / "qualification/samplers.json",
        {
            "seeds": {"17": record},
            "validation": old["validation"],
            "model_forwards": 0,
            "optimizer_updates": 0,
            "actual_batch_calls": 3200,
            "independent_index_draw_calls": 3200,
            "unscored_sampled_target_elements": 6553600,
        },
    )


def test_full_duration_schedule():
    for form in s.FORMS:
        _, tc = s.configuration(form, 17)
        for index in range(3200):
            expected = (
                tc.learning_rate * (index + 1) / 320
                if index < 320
                else tc.learning_rate
                * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ((index - 320) / 2879))))
            )
            assert s.trainer.learning_rate(index, tc) == expected
        assert s.trainer.learning_rate(0, tc) == tc.learning_rate / 320
        assert math.isclose(
            s.trainer.learning_rate(3199, tc), 0.1 * tc.learning_rate, rel_tol=1e-15
        )


def fake_rows():
    return {
        cell: {
            "status": "SCREENED",
            "all_finite": True,
            "validation_loss": 5.94
            if spec["form"] == "gelu_matched"
            else 5.95
            if spec["form"].startswith("narrow_")
            else 6.0,
            "ffn_parameters": 9437184 if spec["form"].startswith("full_") else 2801664,
            "peak_allocated_vram_bytes": 100,
            "late_nll_change_percent": -0.1,
            "final_above_best_percent": 0,
        }
        for cell, spec in s.CELLS.items()
    }


def test_primary_boundary_tie_and_incomplete_behavior():
    rows = fake_rows()
    assert s.decisions(rows)["primary_pass"]
    rows["gelu_matched_seed17"]["validation_loss"] = 5.95
    assert not s.decisions(rows)["gates"]["nll_beats_narrow_gelu"]
    rows["gelu_matched_seed17"]["validation_loss"] = 1.01 * 6
    assert s.decisions(rows)["gates"]["nll_within_one_percent_full_gelu"]
    rows["gelu_matched_seed17"]["validation_loss"] = math.nextafter(1.01 * 6, math.inf)
    assert not s.decisions(rows)["gates"]["nll_within_one_percent_full_gelu"]
    rows = fake_rows()
    rows["gelu_matched_seed17"]["peak_allocated_vram_bytes"] = 110
    assert s.decisions(rows)["primary_pass"]
    rows["gelu_matched_seed17"]["peak_allocated_vram_bytes"] = 111
    assert not s.decisions(rows)["primary_pass"]
    rows = fake_rows()
    rows["gelu_matched_seed17"]["ffn_parameters"] = 3000000
    assert not s.decisions(rows)["gates"]["at_least_70_percent_fewer_ffn_weights"]
    rows = fake_rows()
    rows["plain_seed17"]["all_finite"] = False
    assert not s.decisions(rows)["primary_pass"]
    rows.pop("plain_seed17")
    with pytest.raises(ValueError):
        s.decisions(rows)


def test_descriptive_plateau_and_narrow_margin_are_separate():
    rows = fake_rows()
    assert s.decisions(rows)["primary_pass"]
    assert not s.decisions(rows)["descriptive_point_two_percent_narrow_margin"]["narrow_gelu"]
    rows["plain_seed17"]["late_nll_change_percent"] = 0.2
    rows["plain_seed17"]["final_above_best_percent"] = 0.2
    assert s.decisions(rows)["all_forms_late_plateau"]
    rows["plain_seed17"]["late_nll_change_percent"] = math.nextafter(0.2, math.inf)
    assert not s.decisions(rows)["all_forms_late_plateau"]
    assert s.decisions(rows)["primary_pass"]
