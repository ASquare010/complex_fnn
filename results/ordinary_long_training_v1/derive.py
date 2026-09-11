"""Derive only update timestamps from the frozen long-training loop."""

from pathlib import Path

ROOT = Path("results/ordinary_long_training_v1")
source = Path("results/fp32_training_replication_recovery_v1/source/study.py").read_text()
edits = [
    (
        "        start = time.perf_counter()",
        "        at_ns = time.time_ns()\n        start = time.perf_counter()",
    ),
    (
        "        end = time.perf_counter()",
        "        end = time.perf_counter()\n        until_ns = time.time_ns()",
    ),
    (
        "            warmup=step <= 20,",
        "            warmup=step <= 20,\n            at_ns=at_ns,\n            until_ns=until_ns,",
    ),
]
for before, after in edits:
    assert source.count(before) == 1
    source = source.replace(before, after)
(ROOT / "loop.py").write_text(source, encoding="utf-8", newline="\n")
