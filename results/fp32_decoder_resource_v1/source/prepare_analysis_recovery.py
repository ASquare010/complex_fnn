"""Preserve the terminal analysis failure before a fresh CPU-only retry."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
assert (ROOT / "analyze_exit.txt").read_text().strip() == "3221225477"
assert "access violation" in (ROOT / "analyze.log").read_text()
assert not (ROOT / "summary.json").exists()
assert not (ROOT / "metrics.csv.gz").exists()
assert not (ROOT / "updates.csv.gz").exists()
assert not (ROOT / "analysis_recovery_protocol.json").exists()
paths = [
    ROOT / name
    for name in (
        "source/analyze.py",
        "source/analyze_recovery.py",
        "source/prepare_analysis_recovery.py",
        "source/launch.py",
        "result.json",
        "audit.json",
        "protocol.json",
        "analyze.log",
        "analyze_exit.txt",
    )
]
hashes = {}
for path in paths:
    with path.open("rb") as stream:
        hashes[path.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
(ROOT / "analysis_recovery_protocol.json").write_text(
    json.dumps(
        dict(
            files=hashes,
            prior_raw_exit=3221225477,
            observed_command_exit=-1,
            failure="Windows fatal access violation in the stdlib-only analyzer at line116; cause unresolved",
            missing_exports_before_retry=True,
            scientific_training_repeated=False,
            additional_backward_passes=0,
            additional_optimizer_updates=0,
            action="Fresh process imports and calls unchanged analyzer, with a 45-second traceback watchdog; no scientific retry or tolerance change",
        ),
        indent=2,
    )
    + "\n"
)
print("Preserved the analysis failure; only unchanged CPU analysis is allocated")
