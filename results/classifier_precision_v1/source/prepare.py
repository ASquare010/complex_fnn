"""Verify prior completed science and freeze H113 before any GPU qualification."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/classifier_precision_v1")
OLD = Path("results/whole_job_memory_workspace_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT / "protocol.json").exists()
receipt_path = Path("results/verification/whole_job_memory_final_v1.json")
receipt = json.loads(receipt_path.read_text())
assert receipt["status"] == "PASS" and receipt["fresh_trials"] == 12
for name, digest in receipt["files"].items():
    assert sha(name) == digest, name
protocol = json.loads((OLD / "protocol.json").read_text())
for name, digest in protocol["sources"].items():
    assert sha(name) == digest, name
old_result = json.loads((OLD / "result.json").read_text())
states = {
    c["path"]: c["sha256"]
    for r in old_result["cases"]
    for c in r["checkpoints"]
    if c["step"] in (100, 800)
}
assert len(states) == 24
for name, digest in states.items():
    assert sha(name) == digest, name
before = ROOT / "before_documents"
before.mkdir(exist_ok=False)
documents = {}
for name in (
    "README.md",
    "research/CURRENT_STATE.md",
    "research/PROGRESS_OVERVIEW.md",
    "research/idea_bank.md",
    "research/ARTIFACTS.md",
    "research/literature.md",
):
    target = before / name.replace("/", "__")
    target.write_bytes(Path(name).read_bytes())
    documents[name] = {"sha256": sha(name), "preserved_path": target.as_posix()}
new = [*ROOT.glob("source/*.py"), Path("research/classifier_precision_plan.md")]
sources = {**protocol["sources"], **{p.as_posix(): sha(p) for p in new}}
result = dict(
    sources=sources,
    maintained_files=protocol["maintained_files"],
    datasets=protocol["datasets"],
    checkpoint_hashes=states,
    previous_receipt_sha256=sha(receipt_path),
    previous_results={
        p.as_posix(): sha(p)
        for p in (OLD / "result.json", OLD / "summary.json", OLD / "audit.json")
    },
    before_documents=documents,
    fixtures=24,
    methods=5,
    classifier_backward_passes=504,
    optimizer_updates=0,
    previous_goal_turn="progress",
)
(ROOT / "protocol.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for name in ("captures", "gradients"):
    (ROOT / name).mkdir(exist_ok=False)
print(f"Frozen {len(sources)} sources and {len(states)} checkpoints; prior receipt verified")
