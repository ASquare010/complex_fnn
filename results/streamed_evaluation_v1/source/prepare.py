"""Freeze H111 scientific files and existing checkpoint evidence before scoring."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/streamed_evaluation_v1")
OLD = Path("results/token_memory_duration_fresh_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT / "protocol.json").exists()
receipt = json.loads(Path("results/verification/token_memory_duration_final_v1.json").read_text())
assert receipt["status"] == "PASS"
for path, digest in receipt["files"].items():
    assert sha(path) == digest, path
previous = json.loads((OLD / "protocol.json").read_text())
sources = dict(previous["provenance"]["source_files"])
for path, digest in sources.items():
    assert sha(path) == digest, path
for p in [*ROOT.glob("source/*.py"), Path("research/streamed_evaluation_plan.md")]:
    sources[p.as_posix()] = sha(p)
result = json.loads((OLD / "result.json").read_text())
checkpoints = {c["path"]: c["sha256"] for row in result["cases"] for c in row["checkpoints"]}
assert len(checkpoints) == 24
for path, digest in checkpoints.items():
    assert sha(path) == digest, path
protocol = {
    "sources": sources,
    "checkpoint_hashes": checkpoints,
    "previous_receipt_sha256": sha("results/verification/token_memory_duration_final_v1.json"),
    "previous_result_sha256": sha(OLD / "result.json"),
    "states": 24,
    "evaluation_policies": 3,
    "optimizer_updates": 0,
    "maintained_files": previous["provenance"]["source_files"],
}
(ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
print(f"Frozen {len(sources)} files and 24 checkpoints before qualification/scoring")
