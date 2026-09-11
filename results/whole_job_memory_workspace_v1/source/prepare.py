"""Freeze the workspace correction and preserve every preceding failed attempt."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/whole_job_memory_workspace_v1")
OLD = Path("results/whole_job_memory_continuation_v1")
DIAG = Path("results/cublas_boundary_diagnosis_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT / "protocol.json").exists()
assert (OLD / "study_exit.txt").read_text().strip() == "1"
assert not any(OLD.glob("*/runs/*/history.jsonl"))
assert json.loads((OLD / "qualification.json").read_text())["passed"]
diagnosis = json.loads((DIAG / "result.json").read_text())
assert len(diagnosis["records"]) == 2
assert all(
    r["after_gc_empty_cache_bytes"] == 17039360 and r["after_cublas_clear_bytes"] == 0
    for r in diagnosis["records"]
)
protocol = json.loads((OLD / "protocol.json").read_text())
for path, digest in protocol["sources"].items():
    assert sha(path) == digest, path
preserved = {
    p.as_posix(): sha(p)
    for root in (OLD, DIAG)
    for p in root.rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
}
extra = [*ROOT.glob("source/*.py"), Path("research/whole_job_memory_workspace_plan.md")]
protocol["sources"].update({p.as_posix(): sha(p) for p in extra})
protocol["workspace_recovery"] = {
    "before": preserved,
    "previous_optimizer_updates": 0,
    "zero_allocation_gate_unchanged": True,
    "workspace_bytes_subtracted_from_measurements": 0,
}
text = json.dumps(protocol, indent=2) + "\n"
(ROOT / "protocol.json").write_text(text)
for dataset in protocol["datasets"]:
    path = ROOT / dataset
    path.mkdir(exist_ok=False)
    (path / "protocol.json").write_text(text)
print(
    f"Preserved {len(preserved)} files; frozen {len(protocol['sources'])} sources; zero training repeated"
)
