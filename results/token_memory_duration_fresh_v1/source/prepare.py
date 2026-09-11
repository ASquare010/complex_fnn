"""Record and freeze the explicit second recovery before its scientific work."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/token_memory_duration_fresh_v1")
OLD = Path("results/token_memory_duration_v1")
RECOVERY = Path("results/token_memory_duration_recovery_v1")
DIAGNOSIS = Path("results/runtime_import_diagnosis_v1")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert not (ROOT / "before.json").exists()
assert read(OLD / "audit.json")["passed"]
assert len(read(OLD / "audit.json")["scores"]) == 5
for root in (OLD, RECOVERY):
    assert read(root / "coordinator_status.json")["status"] == "WORKER_FAILED"
    assert (root / "b16_t128_s17_loss_chunks_exit.txt").read_text().strip() == "3221225477"
for path, digest in read(RECOVERY / "before.json")["files"].items():
    assert sha(Path(path)) == digest, path
for path, digest in read(OLD / "protocol.json")["sources"].items():
    assert sha(Path(path)) == digest, path
files = set()
for root in (OLD, RECOVERY, DIAGNOSIS):
    files.update(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
files.update(ROOT.glob("source/*.py"))
files.add(Path("research/token_memory_duration_recovery_plan.md"))
manifest = {
    "before": {p.as_posix(): sha(p) for p in sorted(files)},
    "original_control_updates_reused": 800,
    "candidate_updates_before_continuation": 0,
    "planned_new_trials": 11,
    "stop_on_first_runtime_failure": True,
    "original_frozen_scientific_sources_unchanged": True,
    "runtime_root_cause_proved": False,
    "change": "Original UV Python3.12.9; fresh bytecode prefix, no bytecode writes, CPU dependency preload",
}
(ROOT / "protocol.json").write_bytes((OLD / "protocol.json").read_bytes())
(ROOT / "before.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Frozen {len(files)} evidence/source files before continuation")
