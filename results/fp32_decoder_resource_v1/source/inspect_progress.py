"""Read progress and the last log lines without touching the scientific runtime."""

from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
for name in (
    "progress.json",
    "study_exit.txt",
    "failure.json",
    "audit_exit.txt",
    "audit_failure.json",
):
    path = ROOT / name
    if path.exists():
        print(name, path.read_text())
for name in ("study.log", "audit.log"):
    path = ROOT / name
    if path.exists():
        print(name, "\n".join(path.read_text().splitlines()[-4:]))
