"""Freeze the terminal Matplotlib import failure before one unchanged-code retry."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
assert (ROOT / "plot_exit.txt").read_text().strip() == "3221225477"
assert "Windows fatal exception: access violation" in (ROOT / "plot.log").read_text()
assert not Path("research/figures/fp32_training_replication.png").exists()
assert not (ROOT / "plot_recovery_protocol.json").exists()
for mode in ("study", "audit", "analyze"):
    assert (ROOT / f"{mode}_exit.txt").read_text().strip() == "0"
paths = [
    ROOT / name
    for name in (
        "source/plot.py",
        "source/plot_recovery.py",
        "source/prepare_plot_recovery.py",
        "source/launch.py",
        "protocol.json",
        "audit_protocol.json",
        "result.json",
        "audit.json",
        "summary.json",
        "metrics.csv.gz",
        "curves.csv.gz",
        "updates.csv.gz",
        "plot.log",
        "plot_exit.txt",
    )
]
hashes = {}
for path in paths:
    with path.open("rb") as stream:
        hashes[path.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
(ROOT / "plot_recovery_protocol.json").write_text(
    json.dumps(
        dict(
            files=hashes,
            prior_raw_exit=3221225477,
            observed_command_exit=-1,
            failure="Windows access violation while importing matplotlib.transforms; cause unresolved",
            missing_figure_before_retry=True,
            scientific_training_repeated=False,
            scientific_audit_repeated=False,
            additional_backward_passes=0,
            additional_optimizer_updates=0,
            action="One fresh CPU process imports the unchanged plot, with a 45-second traceback watchdog; inputs, results and gates are unchanged",
        ),
        indent=2,
    )
    + "\n"
)
print("Frozen failed plot and scientific inputs; one CPU-only unchanged-code retry allocated.")
