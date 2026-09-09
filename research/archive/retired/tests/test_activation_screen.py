"""A tiny or cheap gain cannot bypass the frozen activation promotion gates."""

from copy import deepcopy

from src.core.activation_screen import promotion


def case():
    references = {
        "calibrated_narrow": {"validation_loss": 3.0},
        "blockshuffle": {"validation_loss": 2.99},
        "full_swiglu": {"validation_loss": 2.99, "peak_allocated_vram_bytes": 100},
        "full_gelu": {"validation_loss": 2.99, "peak_allocated_vram_bytes": 100},
    }
    candidate = {
        "validation_loss": 2.98,
        "ffn_reduction_percent": 70.3,
        "peak_allocated_vram_bytes": 105,
    }
    return candidate, references


def test_requires_material_gain_over_correct_unmodified_base():
    candidate, refs = case()
    assert all(promotion(candidate, "blockshuffle", refs).values())
    candidate["validation_loss"] = 2.989
    assert not promotion(candidate, "blockshuffle", refs)[
        "at_least_point_two_percent_better_than_own_base"
    ]
    assert promotion(candidate, "calibrated_narrow", refs)[
        "at_least_point_two_percent_better_than_own_base"
    ]


def test_either_full_reference_can_veto_memory_or_quality():
    candidate, refs = case()
    for ref in ("full_swiglu", "full_gelu"):
        changed = deepcopy(refs)
        changed[ref]["validation_loss"] = 2.9
        assert not promotion(candidate, "blockshuffle", changed)[f"within_one_percent_{ref}"]
        changed = deepcopy(refs)
        changed[ref]["peak_allocated_vram_bytes"] = 90
        assert not promotion(candidate, "blockshuffle", changed)[f"memory_within_ten_percent_{ref}"]


def test_parameter_target_and_narrow_control_still_required():
    candidate, refs = case()
    candidate["ffn_reduction_percent"] = 69.999
    assert not promotion(candidate, "blockshuffle", refs)["at_least_70_percent_fewer_ffn_weights"]
    candidate["validation_loss"] = 3.0
    assert not promotion(candidate, "blockshuffle", refs)["beats_calibrated_narrow"]
