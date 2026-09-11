"""Preserve terminal H112 failure and freeze the declared process-isolation change."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/whole_job_memory_continuation_v1")
OLD = Path("results/whole_job_memory_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT / "protocol.json").exists()
assert json.loads((OLD / "coordinator_status.json").read_text())["label"] == "qualification"
assert (OLD / "qualification_exit.txt").read_text().strip() == "3221225477"
assert not any(OLD.glob("*/runs/*/history.jsonl"))
protocol = json.loads((OLD / "protocol.json").read_text())
for path, digest in protocol["sources"].items():
    assert sha(path) == digest, path
preserved = {
    p.as_posix(): sha(p) for p in OLD.rglob("*") if p.is_file() and "__pycache__" not in p.parts
}
extra = [*ROOT.glob("source/*.py"), Path("research/whole_job_memory_recovery_plan.md")]
protocol["sources"].update({p.as_posix(): sha(p) for p in extra})
protocol["runtime_recovery"] = {
    "before": preserved,
    "previous_qualification_completed": False,
    "previous_optimizer_updates": 0,
    "same_process_fresh_models": True,
    "root_cause_proved": False,
}
text = json.dumps(protocol, indent=2) + "\n"
(ROOT / "protocol.json").write_text(text)
for dataset in protocol["datasets"]:
    path = ROOT / dataset
    path.mkdir(exist_ok=False)
    (path / "protocol.json").write_text(text)
print(f"Preserved {len(preserved)} failure files; frozen {len(protocol['sources'])} sources")
