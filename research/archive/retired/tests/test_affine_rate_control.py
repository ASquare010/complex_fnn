"""Same-rate comparisons must not substitute the selected lower-rate base."""

from dataclasses import replace

from src.core.affine_longer import configuration as original
from src.core.affine_rate_control import CASES, RATES, configuration, same_rate_comparisons


def test_rate_intervention_changes_only_global_peak_learning_rate():
    for recipe in CASES:
        mc, tc = configuration(recipe)
        base_mc, base_tc = original(recipe)
        assert mc == base_mc and tc == replace(base_tc, learning_rate=RATES[recipe])


def test_same_rate_comparison_detects_rate_confound():
    def pair(base, affine):
        return {
            "blockshuffle": {"validation_loss": base},
            "blockshuffle_affine": {"validation_loss": affine},
        }

    comparisons = same_rate_comparisons({"600": pair(5.0, 4.9), "1200": pair(4.7, 4.8)})
    assert comparisons["600"]["at_least_point_two_percent_better"]
    assert not comparisons["1200"]["at_least_point_two_percent_better"]
    assert comparisons["1200"]["relative_affine_change_percent"] > 0
