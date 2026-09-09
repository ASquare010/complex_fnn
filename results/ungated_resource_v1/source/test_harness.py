"""H074 isolated integration and decision tests, before any resource outcome."""

import copy

import pytest
import torch

from results.ungated_resource_v1.source import study as s
from src.core import transformer
from src.core.config import ModelConfig
from src.core.optimization import group_summary, parameter_groups

ORIGINAL_FACTORY = transformer.make_ffn


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(4)
    return {f: s.make_model(f) for f in s.FORMS}


def test_actual_counts_and_configuration(models):
    for form, (model, mc, tc) in models.items():
        expected = (
            9437184 if form.startswith("full_") else 1867776 if form == "gelu_same" else 2801664
        )
        assert (
            sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
            == mc.unique_ffn_parameters
            == expected
        )
        assert (
            sum(p.numel() for p in model.parameters()) == mc.total_parameters == expected + 6297984
        )
        mc.validate()
        assert all((b.ffn.gate is None) == ("gelu" in form) for b in model.blocks)
    with pytest.raises(ValueError):
        ModelConfig(variant="blockshuffle_gelu").validate()
    assert transformer.make_ffn is ORIGINAL_FACTORY
    assert models["gelu_matched"][1].variant == "blockshuffle_gelu"


def test_shared_initialization(models):
    reference = {n: p for n, p in models["plain"][0].state_dict().items() if ".ffn." not in n}
    for model, _, _ in models.values():
        assert all(torch.equal(model.state_dict()[n], p) for n, p in reference.items())
    plain, same = models["plain"][0], models["gelu_same"][0]
    assert all(torch.equal(plain.state_dict()[n], p) for n, p in same.state_dict().items())


def test_optimizer_coverage_and_calibration(models):
    for form, (model, mc, tc) in models.items():
        groups = parameter_groups(model, tc)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        assert set(ids) == {id(p) for p in model.parameters()}
        assert sum(g["parameter_count"] for g in group_summary(groups)) == mc.total_parameters
        lookup = {id(p): g["lr_scale"] for g in groups for p in g["params"]}
        if form in ("plain", "gelu_same", "gelu_matched"):
            for block in model.blocks:
                assert lookup[id(block.ffn.up.first.weight)] == 4
                assert lookup[id(block.ffn.up.second.weight)] == 4
                assert lookup[id(block.ffn.down.first.weight)] == 4
                assert lookup[id(block.ffn.down.second.weight)] == mc.ffn_width / 96
        elif form.startswith("narrow_"):
            reference = 1024 if "swiglu" in form else 1536
            ratio = reference / mc.ffn_width
            assert all(lookup[id(b.ffn.down.weight)] == ratio for b in model.blocks)
            uncalibrated = transformer.Transformer(mc, 17)
            assert all(
                torch.equal(b.ffn.down.weight, u.ffn.down.weight * ratio**0.5)
                for b, u in zip(model.blocks, uncalibrated.blocks)
            )
        else:
            assert set(lookup.values()) == {1.0}


def test_adapter_identity_and_evaluation(models):
    x = torch.randint(0, 4096, (1, 5), generator=torch.Generator().manual_seed(77))
    for form in ("full_swiglu", "full_gelu", "plain", "gelu_same", "gelu_matched"):
        original, _, tc = models[form]
        model = copy.deepcopy(original).eval()
        optimizer = torch.optim.AdamW(parameter_groups(model, tc), betas=(0.9, 0.95))
        with torch.no_grad():
            expected = model(x)
        record = s.configure(model, optimizer, "block_inner")
        assert all(
            record[k]
            for k in (
                "parameter_objects_preserved",
                "optimizer_references_preserved",
                "state_preserved",
            )
        )
        with torch.no_grad():
            assert torch.equal(expected, model(x))
            model.train()
            assert torch.equal(expected, model(x))


def test_selection_and_rejection_gates():
    rows = {
        cell: {
            "status": "PASS",
            "all_finite": True,
            "peak_allocated_bytes": 100,
            "median_step_ms": 10,
            "ffn_reduction_percent": 70.3125,
        }
        for cell in s.CELLS
    }
    selected, gates = s.derive_gates(rows, True)
    assert set(selected.values()) == {"none"} and all(all(g.values()) for g in gates.values())
    rows["gelu_matched_block"].update(peak_allocated_bytes=99, median_step_ms=13)
    selected, gates = s.derive_gates(rows, True)
    assert (
        selected["gelu_matched"] == "block"
        and not gates["gelu_matched"]["time_within_25_percent_plain_same_mode"]
    )
    for mode in s.MODES:
        rows[f"gelu_same_{mode}"]["peak_allocated_bytes"] = 111
    assert not s.derive_gates(rows, True)[1]["gelu_same"]["memory_within_ten_percent_full_gelu"]
    assert not any(
        g["all_workers_and_fidelity_exact"] for g in s.derive_gates(rows, False)[1].values()
    )
