"""Frozen optimizer comparison contracts, including failure classification."""

import json
from dataclasses import replace

from src.core.affine_longer import configuration as original_configuration
from src.core.optimizer_bracket import (
    NEW_RATES,
    RATES,
    RECIPES,
    configuration,
    numerical_failure,
    same_rate_comparisons,
    selected_decision,
)


def test_only_global_rate_changes_in_all_eight_new_trials():
    assert len(RECIPES) * len(NEW_RATES) == 8
    for recipe in RECIPES:
        expected_model, expected_training = original_configuration(recipe)
        for rate in RATES:
            model, training = configuration(recipe, rate)
            assert model == expected_model
            assert training == replace(expected_training, learning_rate=rate)
            assert training.steps * training.batch_size * model.context == 1638400
            assert training.eval_batches == 158


def test_selected_and_same_rate_comparisons_keep_distinct_questions():
    matrix = {}
    losses = {
        "full_swiglu": [5.0, 4.5, 4.6],
        "full_gelu": [5.0, 4.5, 4.6],
        "calibrated_narrow": [5.1, 4.7, 4.8],
        "blockshuffle": [4.9, 4.8, 4.55],
    }
    for recipe in RECIPES:
        matrix[recipe] = [
            {
                "run": f"{recipe}_{i}",
                "validation_loss": loss,
                "training": {"learning_rate": rate},
                "ffn_reduction_percent": 70.3 if recipe == "blockshuffle" else 0,
                "peak_allocated_vram_bytes": 100,
            }
            for i, (rate, loss) in enumerate(zip(RATES, losses[recipe]))
        ]
    result = selected_decision(matrix)
    # Candidate wins every same-rate high-rate control but misses selected full quality.
    assert same_rate_comparisons(matrix)["4800"]["full_swiglu"] < 0
    assert not result["quality_passes"] and not result["full_promotion_passes"]
    assert result["grid_positions"]["full_swiglu"] == "interior"
    assert result["grid_positions"]["blockshuffle"] == "upper_boundary"
    matrix["blockshuffle"].pop()
    assert same_rate_comparisons(matrix)["4800"]["full_swiglu"] is None


def test_import_failure_is_never_classified_as_numerical_divergence(tmp_path):
    assert not numerical_failure(tmp_path)
    p = tmp_path / "failure.json"
    p.write_text(
        json.dumps(
            {
                "status": "FAILED",
                "error": "access violation importing torch",
                "traceback": "importlib",
            }
        )
    )
    assert not numerical_failure(tmp_path)
    p.write_text(
        json.dumps(
            {
                "status": "FAILED",
                "error": "gradient total norm is non-finite",
                "traceback": "clip_grad_norm_",
            }
        )
    )
    assert numerical_failure(tmp_path)
    (tmp_path / "metrics.json").write_text("{}")
    assert not numerical_failure(tmp_path)
