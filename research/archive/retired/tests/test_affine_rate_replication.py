"""Replicating the stronger plain control changes only its peak rate."""

from dataclasses import replace

from src.core.affine_longer import configuration as original
from src.core.affine_rate_replication import SEEDS, configuration


def test_same_rate_replication_preserves_each_seed_recipe():
    for seed in SEEDS:
        mc, tc = configuration(seed)
        base_mc, base_tc = original("blockshuffle", seed)
        assert mc == base_mc and tc == replace(base_tc, learning_rate=0.0012)
        assert tc.gate_recompute_method == "native" and tc.recompute_gate
