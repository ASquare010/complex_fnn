"""Frozen language-screen selection and independent gate boundaries."""

import math

import pytest

from src.core.additive_screen import RATES, configuration, decision, select_best


def row(loss, rate=0.0006, memory=100, reduction=70.3125):
    return {
        "validation_loss": loss,
        "training": {"learning_rate": rate},
        "peak_allocated_vram_bytes": memory,
        "ffn_reduction_percent": reduction,
    }


def test_selection_uses_final_finite_loss_and_lower_rate_tie_break():
    a, b, c = row(5.0, 0.0006), row(5.0, 0.0003), row(5.1, 0.0012)
    assert select_best([a, c, b]) is b
    for values in ([], [row(math.nan)], [a, row(math.inf)]):
        with pytest.raises(ValueError):
            select_best(values)


def test_quality_and_memory_require_both_full_controls_and_strict_narrow_win():
    references = {
        "full_swiglu": row(5.0),
        "full_gelu": row(5.0),
        "calibrated_narrow": row(5.04),
        "blockshuffle": row(5.06),
    }
    assert decision(row(5.03, memory=110), references)["earns_separate_longer_comparison"]
    assert not decision(row(5.04), references)["gates"]["beats_calibrated_narrow"]
    for full in ("full_swiglu", "full_gelu"):
        altered = {**references, full: row(4.9)}
        assert not decision(row(5.0), altered)["gates"][f"within_one_percent_{full}"]
        altered = {**references, full: row(5.0, memory=90)}
        assert not decision(row(5.0), altered)["gates"][f"memory_within_ten_percent_{full}"]
    assert not decision(row(5.0, reduction=69.99), references)["earns_separate_longer_comparison"]


def test_only_the_three_frozen_rates_and_qualified_policy_are_available():
    for rate in RATES:
        model, training = configuration(rate)
        assert model.unique_ffn_parameters == 2801664 and model.total_parameters == 9099648
        assert model.ffn_width == 1024
        assert training.learning_rate == rate and training.steps == 200
        assert training.batch_size * model.context * training.steps == 409600
        assert training.ffn_lr_mode == "perturbation" and training.ffn_decay_mode == "parameter"
        assert training.recompute_gate and training.gate_recompute_method == "native"
    with pytest.raises(ValueError):
        configuration(0.0007)
