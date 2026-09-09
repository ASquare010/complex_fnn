"""Promotion must preserve the unmodified base and both full-reference gates."""

from copy import deepcopy

from src.core.affine_longer import promotion


def rows():
    return {
        "full_gelu": {"validation_loss": 3.0, "peak_allocated_vram_bytes": 100},
        "full_swiglu": {"validation_loss": 3.0, "peak_allocated_vram_bytes": 100},
        "calibrated_narrow": {"validation_loss": 3.02},
        "blockshuffle": {"validation_loss": 3.03},
        "blockshuffle_affine": {
            "validation_loss": 3.01,
            "peak_allocated_vram_bytes": 109,
            "ffn_reduction_percent": 70.3,
        },
    }


def test_gate_uses_unmodified_blockshuffle_after_candidate_substitution():
    source = rows()
    before = deepcopy(source)
    assert all(promotion(source).values())
    assert source == before
    source["blockshuffle"]["validation_loss"] = 3.015
    assert not promotion(source)["at_least_point_two_percent_better_than_unmodified_blockshuffle"]


def test_either_full_reference_and_narrow_can_veto():
    for reference in ("full_gelu", "full_swiglu"):
        source = rows()
        source[reference]["peak_allocated_vram_bytes"] = 98
        assert not promotion(source)[f"memory_within_ten_percent_{reference}"]
        source[reference]["validation_loss"] = 2.9
        assert not promotion(source)[f"within_one_percent_{reference}"]
    source = rows()
    source["calibrated_narrow"]["validation_loss"] = 3.01
    assert not promotion(source)["beats_calibrated_narrow"]
