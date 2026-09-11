"""Read stage completion and recent logs without touching CUDA."""

from pathlib import Path

ROOT = Path("results/training_variability_v1")
for name in (
    "progress.json",
    "study_exit.txt",
    "audit_exit.txt",
    "failure.json",
    "audit_failure.json",
):
    path = ROOT / name
    if path.exists():
        print(name, path.read_text())
for name in ("study.log", "audit.log"):
    path = ROOT / name
    if path.exists():
        print(name, "\n".join(path.read_text().splitlines()[-4:]))
