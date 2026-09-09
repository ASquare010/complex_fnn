"""Reject joint claims when either quality region or kernel fidelity fails."""

from copy import deepcopy

import pytest

from src.core.joint_conditioning import compare_seed


def fixture_records():
    original = {
        "worst_down_condition": 500.0,
        "numerical_checks_pass": True,
        "finite_gradients": True,
        "native": {"prefix_nll": 2.8, "tail_nll": 2.9},
    }
    floored = deepcopy(original)
    floored["worst_down_condition"] = 20.0
    floored["compiled"] = deepcopy(floored["native"])
    controls = {
        "full_swiglu": {"prefix_nll": 2.8, "tail_nll": 2.9},
        "full_gelu": {"prefix_nll": 2.9, "tail_nll": 3.0},
        "calibrated_narrow": {"prefix_nll": 2.85, "tail_nll": 2.95},
    }
    return original, floored, controls


def test_both_regions_and_execution_paths_are_required():
    original, floored, controls = fixture_records()
    assert all(compare_seed(original, floored, controls).values())
    floored["compiled"]["tail_nll"] = 2.91
    gates = compare_seed(original, floored, controls)
    assert gates["prefix_compiled_floor_cost_at_most_point_one_percent"]
    assert gates["tail_native_floor_cost_at_most_point_one_percent"]
    assert not gates["tail_compiled_floor_cost_at_most_point_one_percent"]
    assert gates["tail_within_one_percent_full_swiglu"]


@pytest.mark.parametrize("field", ["numerical_checks_pass", "finite_gradients"])
def test_numerics_and_finiteness_cannot_be_hidden_by_good_quality(field):
    original, floored, controls = fixture_records()
    floored[field] = False
    assert not all(compare_seed(original, floored, controls).values())


def test_equal_narrow_loss_and_insufficient_condition_improvement_reject():
    original, floored, controls = fixture_records()
    floored["worst_down_condition"] = 51.0
    controls["calibrated_narrow"]["prefix_nll"] = 2.8
    gates = compare_seed(original, floored, controls)
    assert not gates["condition_improves_at_least_tenfold"]
    assert not gates["prefix_beats_calibrated_narrow"]
