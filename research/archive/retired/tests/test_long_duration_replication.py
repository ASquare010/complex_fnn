"""Guard independent-seed scheduling and prevent a mean from hiding failed seeds."""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.core.long_duration import configuration as seed17_configuration
from src.core.long_duration_replication import ORDER, RECIPES, SEEDS, configuration, summarize


def test_replication_changes_only_seed_and_never_repeats_seed17():
    assert len(ORDER) == len(set(ORDER)) == 8
    assert {s for s, _ in ORDER} == {29, 43}
    for seed, recipe in ORDER:
        old_mc, old_tc = seed17_configuration(recipe)
        mc, tc = configuration(recipe, seed)
        assert mc == old_mc and tc == replace(old_tc, seed=seed)
        assert tc.steps * tc.batch_size * mc.context == 6553600
    with pytest.raises(ValueError, match="outside"):
        configuration("blockshuffle", 71)


def test_passing_mean_cannot_hide_failed_or_missing_seed():
    base = {
        r: {
            "validation_loss": 4.0 if r != "blockshuffle" else 3.95,
            "peak_allocated_vram_bytes": 100,
            "ffn_reduction_percent": 70.3125,
        }
        for r in RECIPES
    }
    rows = {s: deepcopy(base) for s in SEEDS}
    assert summarize(rows)["all_seeds_pass_primary_gates"]
    rows[43]["blockshuffle"]["validation_loss"] = 4.05
    result = summarize(rows)
    assert result["mean_nll"]["blockshuffle"] < result["mean_nll"]["full_swiglu"]
    assert not result["all_seeds_pass_primary_gates"]
    assert not result["per_seed"]["43"]["gates"]["within_one_percent_full_swiglu"]
    del rows[43]["full_gelu"]
    result = summarize(rows)
    assert not result["all_twelve_complete"] and not result["all_seeds_pass_primary_gates"]
    assert "mean_nll" not in result
