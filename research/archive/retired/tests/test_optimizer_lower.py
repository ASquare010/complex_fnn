"""Contracts for completing the lower-rate boundary without changing frozen H048."""

from dataclasses import replace

from src.core.affine_longer import configuration as original_configuration
from src.core.optimizer_lower import (
    NEW_RECIPES,
    RATES,
    RECIPES,
    configuration,
    decisions,
    same_rates,
)


def test_three_missing_cells_change_only_global_rate():
    assert NEW_RECIPES == ("full_swiglu", "calibrated_narrow", "full_gelu")
    assert RATES == (0.0006, 0.0012, 0.0024, 0.0048)
    for r in RECIPES:
        mc, tc = configuration(r)
        old_mc, old_tc = original_configuration(r)
        assert mc == old_mc and tc == replace(old_tc, learning_rate=0.0006)
        assert tc.steps * tc.batch_size * mc.context == 1638400


def test_new_lower_control_can_remove_the_selected_quality_pass():
    matrix = {r: [] for r in RECIPES}
    values = {
        "full_swiglu": [4.3, 4.9, 5.0, 5.1],
        "full_gelu": [4.3, 4.9, 5.0, 5.1],
        "calibrated_narrow": [4.4, 4.9, 5.0, 5.1],
        "blockshuffle": [4.8, 4.7, 5.0, 5.2],
    }
    for r in RECIPES:
        matrix[r] = [
            {
                "run": f"{r}_{i}",
                "validation_loss": loss,
                "training": {"learning_rate": rate},
                "ffn_reduction_percent": 70.3 if r == "blockshuffle" else 0,
                "peak_allocated_vram_bytes": 100,
            }
            for i, (rate, loss) in enumerate(zip(RATES, values[r]))
        ]
    d = decisions(matrix)
    assert not d["quality_passes"] and not d["full_promotion_passes"]
    assert d["grid_positions"]["blockshuffle"] == "interior"
    assert d["grid_positions"]["full_swiglu"] == "lower_boundary"
    assert same_rates(matrix)["600"]["full_swiglu"] > 0
    assert same_rates(matrix)["1200"]["full_swiglu"] < 0
