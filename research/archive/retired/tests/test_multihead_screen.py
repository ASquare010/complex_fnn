"""Keep quality-investigation and full promotion distinct under memory failure."""

from src.core.multihead_screen import RATES, RECIPES, candidate_gates, configuration
from src.core.wikitext_screen import configuration as control_configuration


def test_multihead_screen_keeps_shared_full_training_settings():
    for rate in RATES:
        _, full = control_configuration("full_swiglu", rate)
        for recipe in RECIPES:
            model, training = configuration(recipe, rate)
            assert training == full and model.unique_ffn_parameters == 2807808
            assert model.width == 384 and model.groups == 48 and model.hidden == 24


def test_memory_failure_does_not_erase_a_quality_lead():
    references = {
        k: {"validation_loss": 5.0, "peak_allocated_vram_bytes": 100}
        for k in ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")
    }
    candidate = {
        "validation_loss": 4.95,
        "peak_allocated_vram_bytes": 120,
        "ffn_reduction_percent": 70.2,
    }
    result = candidate_gates(candidate, references)
    assert result["quality_investigation_passes"] and not result["full_promotion_passes"]
    assert references["blockshuffle"]["validation_loss"] == 5.0
