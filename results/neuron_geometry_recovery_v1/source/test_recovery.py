"""Repeat the original27 cases with one declared precision-region change."""

import pytest

from results.neuron_geometry_recovery_v1.source.recovery import bindings
from results.neuron_geometry_v1.source.test_qualification import (  # noqa: F401
    test_counts,
    test_cuda_ffn,
    test_formula_and_gradcheck,
    test_geometry,
    test_groupsort,
    test_initial_functions,
    test_scalar_slopes,
)


@pytest.fixture(scope="session", autouse=True)
def declared_precision_region():
    with bindings():
        yield
