"""Protect equal-budget checkpoint selection and both full-reference gates."""

from copy import deepcopy

import pytest

from src.core.wikitext_screen import promotion, select_best


def trial(loss, rate, status="SCREENED"):
    return {"status": status, "validation_loss": loss, "training": {"learning_rate": rate}}


def test_selection_uses_final_loss_then_lower_rate_and_ignores_failed_trials():
    a, b = trial(3.0, 0.0012), trial(3.0, 0.0006)
    assert select_best([a, trial(2.0, 0.0003, "FAILED"), b]) is b
    c = trial(2.99, 0.0012)
    assert select_best([b, c]) is c


def test_no_complete_finite_trial_cannot_win():
    with pytest.raises(ValueError):
        select_best([trial(float("nan"), 0.0003), trial(2.0, 0.0006, "FAILED")])


def selected():
    return {
        "blockshuffle": {
            "validation_loss": 3.0,
            "ffn_reduction_percent": 70.3125,
            "peak_allocated_vram_bytes": 105,
        },
        "calibrated_narrow": {"validation_loss": 3.1},
        "full_swiglu": {"validation_loss": 3.0, "peak_allocated_vram_bytes": 100},
        "full_gelu": {"validation_loss": 3.0, "peak_allocated_vram_bytes": 100},
    }


def test_either_full_reference_can_reject_quality_or_memory():
    rows = selected()
    assert all(promotion(rows).values())
    for key in ("full_swiglu", "full_gelu"):
        case = deepcopy(rows)
        case[key]["validation_loss"] = 2.96
        assert not promotion(case)[f"within_one_percent_{key}"]
        case = deepcopy(rows)
        case[key]["peak_allocated_vram_bytes"] = 90
        assert not promotion(case)[f"memory_within_ten_percent_{key}"]


def test_parameter_reduction_and_strict_narrow_advantage_are_required():
    rows = selected()
    rows["blockshuffle"]["ffn_reduction_percent"] = 69.9
    rows["calibrated_narrow"]["validation_loss"] = 3.0
    gates = promotion(rows)
    assert not gates["at_least_70_percent_fewer_ffn_weights"]
    assert not gates["beats_calibrated_narrow"]
