"""Read logs and completion records without importing scientific dependencies."""

from pathlib import Path

ROOT = Path("results/fp32_training_replication_v1")
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
        print(name, "\n".join(path.read_text().splitlines()[-5:]))
