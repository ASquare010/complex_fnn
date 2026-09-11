"""Preserve the original terminal failure and all completed evidence before recovery."""

import hashlib
import json
from pathlib import Path

old = Path("results/token_memory_duration_v1")
root = Path("results/token_memory_duration_recovery_v1")
assert not (root / "protocol.json").exists()
assert json.loads((old / "coordinator_status.json").read_text())["status"] == "WORKER_FAILED"
failed = old / "runs/b16_t128_s17_loss_chunks"
assert failed.is_dir() and not any(failed.iterdir())
assert (old / "study_exit.txt").read_text().strip() == "1"
assert (old / "b16_t128_s17_loss_chunks_exit.txt").read_text().strip() == "3221225477"
files = [
    old / n
    for n in (
        "protocol.json",
        "source.zip",
        "qualification.json",
        "study.log",
        "study_exit.txt",
        "study_failure.json",
        "coordinator_status.json",
        "b16_t128_s17_block.log",
        "b16_t128_s17_block_exit.txt",
        "b16_t128_s17_loss_chunks.log",
        "b16_t128_s17_loss_chunks_exit.txt",
    )
]
files += sorted((old / "runs/b16_t128_s17_block").glob("*"))
before = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root / "protocol.json").write_bytes((old / "protocol.json").read_bytes())
(root / "before.json").write_text(
    json.dumps(
        {
            "files": before,
            "original_completed_updates": 800,
            "failed_candidate_updates": 0,
            "failed_candidate_directory_empty": True,
            "change": "Preload sympy and torch._dynamo on CPU before invoking unchanged frozen worker in a new root",
            "algorithm_precision_optimizer_gates_changed": False,
            "native_root_cause_proved": False,
        },
        indent=2,
    )
    + "\n"
)
print("Preserved original failure and completed control; no original file changed")
