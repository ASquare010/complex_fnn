"""Guard the H056 configuration and original quality/memory decision boundaries."""

from dataclasses import replace

import pytest

from src.core.overcomplete_screen import RATES, configuration, decision


def test_frozen_overcomplete_screen_matches_retained_budget():
    configurations = [configuration(rate) for rate in RATES]
    mc, tc = configurations[0]
    assert (mc.width, mc.layers, mc.heads, mc.context, mc.groups, mc.hidden) == (
        384,
        8,
        6,
        128,
        8,
        200,
    )
    assert (mc.unique_ffn_parameters, mc.total_parameters) == (2801664, 9099648)
    assert all(
        c == mc and replace(t, learning_rate=tc.learning_rate) == tc for c, t in configurations
    )
    assert tc.steps * tc.batch_size * mc.context == 409600
    assert tc.eval_batches == 158 and tc.seed == 17 and tc.precision == "bf16"
    assert tc.ffn_lr_mode == "fan_in" and tc.ffn_decay_mode == "parameter"
    assert tc.recompute_gate and tc.gate_recompute_method == "native"
    with pytest.raises(ValueError, match="outside"):
        configuration(0.0024)


def test_promotion_requires_both_full_controls_and_strict_narrow_win():
    references = {
        "full_swiglu": {"validation_loss": 5.0, "peak_allocated_vram_bytes": 100},
        "full_gelu": {"validation_loss": 4.99, "peak_allocated_vram_bytes": 90},
        "calibrated_narrow": {"validation_loss": 5.03},
        "blockshuffle": {"validation_loss": 4.9},
    }
    candidate = {
        "validation_loss": 5.02,
        "peak_allocated_vram_bytes": 99,
        "ffn_reduction_percent": 70.3125,
    }
    assert decision(candidate, references)["earns_separate_longer_comparison"]
    # Ordinary strict win is required; the separate 0.2% margin is descriptive.
    assert not decision(candidate, references)["diagnostic_point_two_percent_narrow_margin"]
    for change in (
        {"validation_loss": 5.03},
        {"peak_allocated_vram_bytes": 100},
        {"ffn_reduction_percent": 69.9},
    ):
        assert not decision({**candidate, **change}, references)["earns_separate_longer_comparison"]
    stricter_gelu = {
        **references,
        "full_gelu": {"validation_loss": 4.96, "peak_allocated_vram_bytes": 90},
    }
    assert not decision(candidate, stricter_gelu)["earns_separate_longer_comparison"]
