"""Five H077 checks; initialization and sampler checks perform zero model updates."""

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from results.ungated_lm_replication_v1.source import study as s
from src.core.data import TokenData
from src.core.native_recompute_audit import digest, read, tensor_record, write_new
from src.core.optimization import initialize_dense_width, parameter_groups


def test_frozen_rates_and_exact_recipe_differences():
    previous = read("results/ungated_lm_screen_v1/result.json")
    assert previous["earns_longer_comparison"] == {"gelu_same": False, "gelu_matched": True}
    assert len(s.CELLS) == 18 and "gelu_same" not in s.FORMS
    assert [v["form"] for v in s.CELLS.values() if v["seed"] == 29] == list(s.FORMS[::-1])
    for spec in s.CELLS.values():
        form, seed = spec["form"], spec["seed"]
        chosen = previous["rows"][previous["selected_cells"][form]]
        mc, tc = s.configuration(form, seed)
        assert s.RATES[form] == chosen["training"]["learning_rate"]
        assert asdict(mc) == chosen["model"]
        assert asdict(tc) == {**chosen["training"], "steps": 800, "log_every": 200, "seed": seed}
        assert s.MODES[form] == chosen["execution_mode"]
    with pytest.raises(ValueError):
        s.configuration("gelu_same", 17)
    with pytest.raises(ValueError):
        s.configuration("gelu_matched", 99)


def test_initial_states_counts_and_optimizer_calibration():
    torch.set_num_threads(4)
    previous = read("results/ungated_lm_screen_v1/result.json")
    initials, common = {}, {}
    for cell, spec in s.CELLS.items():
        form, seed = spec["form"], spec["seed"]
        mc, tc = s.configuration(form, seed)
        torch.manual_seed(seed)
        model = s.base.make_model(mc, seed, s.MODES[form])
        metadata = initialize_dense_width(model, tc)
        weights = {n: tensor_record(v) for n, v in model.named_parameters()}
        shared = {n: v for n, v in weights.items() if ".ffn." not in n}
        if seed not in common:
            common[seed] = shared
        assert shared == common[seed]
        count = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
        assert (
            count == mc.unique_ffn_parameters == (9437184 if form.startswith("full_") else 2801664)
        )
        assert sum(p.numel() for p in model.parameters()) == mc.total_parameters == count + 6297984
        assert s.operation_counts(mc)["ffn_matrix_forward_flops_per_token"] == 2 * count
        initials[cell] = {
            "weights": digest(model.state_dict()),
            "parameters": weights,
            "width_initialization": metadata,
            "execution_mode": s.MODES[form],
        }
        if seed == 17:
            old = read(
                Path("results/ungated_lm_screen_v1/runs")
                / previous["selected_cells"][form]
                / "initial_state_signature.json"
            )
            assert initials[cell] == old
        groups = parameter_groups(model, tc)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        group_summary = [
            {
                "weight_decay": g["weight_decay"],
                "lr_scale": g["lr_scale"],
                "parameter_count": sum(p.numel() for p in g["params"]),
            }
            for g in groups
        ]
        assert (
            group_summary
            == previous["rows"][previous["selected_cells"][form]]["optimizer_parameter_groups"]
        )
        if form.startswith("narrow_"):
            raw = s.base.make_model(mc, seed, "none")
            ratio = (1024 if form == "narrow_swiglu" else 1536) / mc.ffn_width
            lookup = {id(p): g for g in groups for p in g["params"]}
            for a, b in zip(raw.blocks, model.blocks):
                assert torch.equal(b.ffn.down.weight, a.ffn.down.weight * math.sqrt(ratio))
                g = lookup[id(b.ffn.down.weight)]
                assert g["lr_scale"] == ratio and math.isclose(
                    g["weight_decay"] * ratio, 0.1, rel_tol=1e-15
                )
            del raw
        del model, groups
    assert len(initials) == 18
    write_new(s.ROOT / "qualification/initial_signatures.json", initials)


def test_schedule_readonly_sampler_and_validation_order():
    torch.set_num_threads(4)
    for form in s.FORMS:
        _, tc = s.configuration(form, 17)
        for index in range(800):
            expected = (
                tc.learning_rate * (index + 1) / 80
                if index < 80
                else tc.learning_rate
                * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ((index - 80) / 719))))
            )
            assert s.trainer.learning_rate(index, tc) == expected
    prior = read("results/ungated_lm_screen_v1/result.json")
    seeds = {}
    for seed in s.SEEDS:
        data = TokenData(s.CACHE, "cuda", seed + 10000)
        independent = torch.Generator(device="cuda").manual_seed(seed + 10000)
        record = {}
        for step in range(800):
            starts = torch.randint(
                len(data.train) - 128, (16,), device="cuda", generator=independent
            )
            x, y = data.batch(16, 128)
            if step == 0:
                indices = starts[:, None] + torch.arange(128, device="cuda")
                assert torch.equal(x, data.train[indices]) and torch.equal(
                    y, data.train[indices + 1]
                )
                record.update(
                    first_starts=tensor_record(starts),
                    first_x=tensor_record(x),
                    first_y=tensor_record(y),
                )
            if step in (199, 799):
                assert torch.equal(data.generator.get_state(), independent.get_state())
                record[f"state_after_{step + 1}"] = tensor_record(data.generator.get_state())
            if step == 199 and seed == 17:
                for form in s.FORMS:
                    q = read(
                        Path("results/ungated_lm_screen_v1/runs")
                        / prior["selected_cells"][form]
                        / "qualification.json"
                    )
                    assert record["state_after_200"] == q["sampling_rng"]
        seeds[str(seed)] = record
        if seed == 17:
            batches = [y.detach().cpu() for _, y in data.validation(16, 128, 158)]
            actual = torch.cat([y.flatten() for y in batches])
            expected = torch.from_numpy(np.load(s.CACHE / "valid.npy").astype("int64"))[1:322689]
            assert torch.equal(actual, expected) and actual.numel() == 322688
            assert len(batches) == 158 and list(batches[-1].shape) == [9, 128]
            validation = {
                "targets": 322688,
                "batches": 158,
                "last_shape": [9, 128],
                "ordered_targets": tensor_record(actual),
            }
        del data
    write_new(
        s.ROOT / "qualification/samplers.json",
        {
            "seeds": seeds,
            "validation": validation,
            "model_forwards": 0,
            "optimizer_updates": 0,
            "actual_batch_calls": 2400,
            "independent_index_draw_calls": 2400,
            "unscored_sampled_target_elements": 4915200,
        },
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


def test_every_seed_gate_boundaries_and_no_mean_rescue():
    rows = fake_rows()
    assert s.decisions(rows)["replicated_primary_pass"]
    rows["gelu_matched_seed43"]["validation_loss"] = 5.96
    d = s.decisions(rows)
    assert d["nll_summary"]["gelu_matched"]["mean"] < d["nll_summary"]["narrow_gelu"]["mean"]
    assert (
        not d["replicated_primary_pass"] and not d["per_seed_gates"]["43"]["nll_beats_narrow_gelu"]
    )
    rows = fake_rows()
    rows["gelu_matched_seed17"]["peak_allocated_vram_bytes"] = 110
    assert s.decisions(rows)["replicated_primary_pass"]
    rows["gelu_matched_seed17"]["peak_allocated_vram_bytes"] = 111
    assert not s.decisions(rows)["replicated_primary_pass"]
    rows = fake_rows()
    rows["gelu_matched_seed17"]["ffn_parameters"] = 3000000
    assert not s.decisions(rows)["per_seed_gates"]["17"]["at_least_70_percent_fewer_ffn_weights"]
    rows = fake_rows()
    rows["gelu_matched_seed17"]["validation_loss"] = 1.01 * 6.0
    assert s.decisions(rows)["per_seed_gates"]["17"]["nll_within_one_percent_full_gelu"]
    rows["gelu_matched_seed17"]["validation_loss"] = math.nextafter(1.01 * 6.0, math.inf)
    assert not s.decisions(rows)["per_seed_gates"]["17"]["nll_within_one_percent_full_gelu"]
    rows = fake_rows()
    rows["plain_seed29"]["all_finite"] = False
    assert not s.decisions(rows)["replicated_primary_pass"]
    rows.pop("plain_seed29")
    with pytest.raises(ValueError):
        s.decisions(rows)


def test_statistical_and_plateau_diagnostics():
    constant = s.summary([2.0, 2.0, 2.0])
    assert constant == {"mean": 2, "sample_sd": 0, "exploratory_ci95": [2, 2]}
    values = s.summary([1.0, 2.0, 3.0])
    assert values["mean"] == 2 and values["sample_sd"] == 1
    assert math.isclose(
        0.5 + s.T_CRITICAL / (2 * math.sqrt(s.T_CRITICAL**2 + 2)), 0.975, rel_tol=1e-15
    )
    with pytest.raises(ValueError):
        s.summary([1, 2])
    rows = fake_rows()
    rows["plain_seed17"]["late_nll_change_percent"] = 0.2
    rows["plain_seed17"]["final_above_best_percent"] = 0.2
    assert s.decisions(rows)["all_forms_seeds_late_plateau"]
    rows["plain_seed17"]["late_nll_change_percent"] = math.nextafter(0.2, math.inf)
    assert not s.decisions(rows)["all_forms_seeds_late_plateau"]
    assert s.decisions(rows)["replicated_primary_pass"]
