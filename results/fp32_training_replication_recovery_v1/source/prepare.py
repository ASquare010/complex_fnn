"""Freeze the diagnosed startup recovery; preserve original failure and budget."""

import hashlib
import json
from pathlib import Path

OLD = Path("results/fp32_training_replication_v1")
ROOT = Path("results/fp32_training_replication_recovery_v1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert not (ROOT / "protocol.json").exists()
original = read(OLD / "protocol.json")
for path, digest in original["sources"].items():
    assert sha(path) == digest, path
assert (OLD / "study_exit.txt").read_text().strip() == "1"
assert "assert not torch.cuda.is_initialized()" in read(OLD / "failure.json")["traceback"]
qualification = read(OLD / "qualification.json")
assert qualification["passed"] and qualification["qualification_backward_passes"] == 24
assert not list((OLD / "runs").iterdir())
assert not list((OLD / "initial").glob("**/*.pt"))
assert not (OLD / "initializations.json").exists()
assert not (OLD / "result.json").exists()
fixtures = []
for fixture in original["fixtures"]:
    item = fixture.copy()
    item["checkpoint"] = (ROOT / "initial" / item["label"] / "step0.pt").as_posix()
    fixtures.append(item)
sources = {
    **original["sources"],
    **{
        p.as_posix(): sha(p)
        for p in (
            *ROOT.glob("source/*.py"),
            Path("research/fp32_training_replication_recovery_plan.md"),
        )
    },
}
protocol = {
    **original,
    "sources": sources,
    "fixtures": fixtures,
    "original_failed_attempt": {
        "root": OLD.as_posix(),
        "completed_updates": 0,
        "qualification_backward_passes": 24,
        "reason": "CUDA environment metadata queried before CPU initialization guard",
    },
    "original_evidence": {
        p.as_posix(): sha(p)
        for p in (
            OLD / "protocol.json",
            OLD / "failure.json",
            OLD / "environment.json",
            OLD / "qualification.json",
            OLD / "study.log",
            OLD / "study_exit.txt",
        )
    },
    "training_repeated": False,
    "total_planned_backward_passes_across_attempts": 14478,
    "recovery_plan": "research/fp32_training_replication_recovery_plan.md",
}
(ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
(ROOT / "runs").mkdir(exist_ok=False)
print(f"Frozen recovery with {len(sources)} sources; original training count verified zero")
