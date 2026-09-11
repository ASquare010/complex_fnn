"""Derive timing-only edits from the unchanged H120 continuation source."""

from pathlib import Path

ROOT = Path("results/paired_complete_training_v1")
source = Path("results/optimizer_memory_v1/source/profile.py").read_text()
edits = [
    (
        "        started = time.perf_counter()",
        "        started_ns = time.time_ns()\n        started = time.perf_counter()",
    ),
    ('        mark(f"forward_{step}", [x, y, loss])\n', ""),
    ('        mark(f"backward_{step}", [x, y, loss])\n', ""),
    ('        mark(f"clipping_{step}", [x, y, loss, norm])\n', ""),
    (
        '        mark(f"optimizer_{step}", [x, y, loss, norm])\n        wall_ms = 1000 * (time.perf_counter() - started)',
        '        events[7].synchronize()\n        wall_ms = 1000 * (time.perf_counter() - started)\n        until_ns = time.time_ns()\n        event_complete_ms = events[0].elapsed_time(events[7])\n        mark(f"optimizer_{step}", [x, y, loss, norm])',
    ),
    (
        "            wall_ms=wall_ms,",
        "            wall_ms=wall_ms,\n            event_complete_ms=event_complete_ms,\n            at_ns=started_ns,\n            until_ns=until_ns,",
    ),
    (
        'for k in ("wall_ms", "event_sum_ms")}',
        'for k in ("wall_ms", "event_sum_ms", "event_complete_ms")}',
    ),
]
for before, after in edits:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
(ROOT / "loop.py").write_text(source, encoding="utf-8", newline="\n")
