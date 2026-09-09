"""Frozen treatment isolation and promotion require material, verified quality."""

from copy import deepcopy

from src.core.duration_decay import configuration, decision


def test_only_decay_changes_and_reference_is_not_mutated():
    old = {
        "model": {"width": 384},
        "training": {"ffn_decay_mode": "parameter", "steps": 3200, "seed": 17},
    }
    saved = deepcopy(old)
    new = configuration(old)
    assert old == saved
    new["training"]["ffn_decay_mode"] = "parameter"
    assert new == old


def test_tiny_gain_missing_evidence_or_memory_failure_cannot_promote():
    refs = {
        r: {"validation_loss": 4.0, "ffn_parameters": 100, "peak_allocated_vram_bytes": 100}
        for r in ("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")
    }
    good = {"validation_loss": 3.99, "ffn_parameters": 30, "peak_allocated_vram_bytes": 105}
    assert decision(good, refs, True)["earns_replication"]
    assert not decision(good, refs, False)["earns_replication"]
    assert not decision(None, refs, False)["earns_replication"]
    tiny = {**good, "validation_loss": 3.999}
    assert decision(tiny, refs, True)["local_gates_pass"]
    assert not decision(tiny, refs, True)["earns_replication"]
    assert not decision({**good, "peak_allocated_vram_bytes": 111}, refs, True)["earns_replication"]
