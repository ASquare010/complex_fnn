"""Allocator lifetimes include rounded blocks and delayed reuse, not just requested bytes."""

import pytest

from src.core.allocation_trace import allocation_origin, replay_allocations


def snapshot(blocks, events, *, kind="small", address=4096):
    return {
        "segments": [
            {
                "device": 0,
                "address": address,
                "total_size": sum(b[1] for b in blocks),
                "segment_type": kind,
                "blocks": [
                    {"address": a, "size": s, "requested_size": r, "state": state}
                    for a, s, r, state in blocks
                ],
            }
        ],
        "device_traces": [events],
    }


def test_delayed_free_pointer_reuse_and_phase_peak():
    before = snapshot(
        [(4096, 512, 400, "active_allocated"), (4608, 1536, 0, "inactive")],
        [{"action": "snapshot"}],
    )
    events = [
        {"action": "snapshot"},
        {"action": "alloc", "addr": 4608, "size": 700},
        {"action": "free_requested", "addr": 4608, "size": 700},
        {"action": "free_completed", "addr": 4608, "size": 700},
        {"action": "alloc", "addr": 4608, "size": 513},
        {"action": "alloc", "addr": 5632, "size": 1},
        {"action": "free_requested", "addr": 4608, "size": 513},
        {"action": "snapshot"},
    ]
    after = snapshot(
        [
            (4096, 512, 400, "active_allocated"),
            (4608, 1024, 513, "active_awaiting_free"),
            (5632, 512, 1, "active_allocated"),
        ],
        events,
    )
    row = replay_allocations(before, after, forward_end=4)
    assert row["initial_bytes"] == 512 and row["final_bytes"] == 1024
    assert row["peak_bytes"] == 2048 and row["peak_event"] == 5 and row["peak_phase"] == "backward"


def test_large_pool_keeps_small_remainder_as_allocated_padding():
    size = 2**21
    requested = 3 * 2**19
    before = snapshot([(4096, size, 0, "inactive")], [], kind="large")
    events = [{"action": "alloc", "addr": 4096, "size": requested}]
    after = snapshot([(4096, size, requested, "active_allocated")], events, kind="large")
    row = replay_allocations(before, after, forward_end=1)
    assert row["peak_bytes"] == size and row["peak_allocations"][0]["requested_size"] == requested


def test_replay_rejects_missing_history_terminal_mismatch_and_ring_loss():
    before = snapshot([(4096, 512, 400, "active_allocated")], [])
    with pytest.raises(ValueError, match="unknown"):
        replay_allocations(
            before,
            snapshot(
                [(4096, 512, 400, "active_allocated")],
                [{"action": "free_requested", "addr": 9, "size": 20}],
            ),
            forward_end=0,
        )
    with pytest.raises(ValueError, match="terminal"):
        replay_allocations(before, snapshot([(4096, 512, 0, "inactive")], []), forward_end=0)
    with pytest.raises(ValueError, match="Truncated"):
        replay_allocations(
            before,
            snapshot([(4096, 512, 400, "active_allocated")], [{"action": "snapshot"}]),
            forward_end=0,
            cap=1,
        )


def test_source_regions_do_not_attribute_projection_to_outer_rational_wrapper():
    regions = [
        {
            "path": "src/rational_blockshuffle_ffn/__init__.py",
            "start": 40,
            "end": 65,
            "category": "rational_pointwise",
            "function": "residual",
        }
    ]
    block = {
        "preexisting": False,
        "addr": 1,
        "frames": [
            {"filename": "src/rational_blockshuffle_ffn/__init__.py", "line": 90, "name": "forward"}
        ],
    }
    assert allocation_origin(block, {}, regions)["category"] == "other_runtime"
    block["frames"].insert(
        0,
        {
            "filename": "C:/repo/src/rational_blockshuffle_ffn/__init__.py",
            "line": 60,
            "name": "residual",
        },
    )
    assert allocation_origin(block, {}, regions)["category"] == "rational_pointwise"
