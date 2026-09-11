"""Freeze H112 only after the independent H111 qualification is earned."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/whole_job_memory_v1")
EVAL = Path("results/streamed_evaluation_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


assert not (ROOT / "protocol.json").exists()
summary, audit = read(EVAL / "summary.json"), read(EVAL / "audit.json")
assert audit["passed"] and audit["native_rescores"] == 24 and audit["policy_comparisons"] == 72
assert audit["all_gates_and_policy_selections_independently_verified"]
group = next(g for g in summary["groups"] if g["context"] == 512)
policy = group["selected_policy"]
assert policy is not None and group["policies"][policy]["eligible"]
sources = dict(read(EVAL / "protocol.json")["sources"])
sources.update(read("results/token_memory_duration_v1/protocol.json")["sources"])
for path, digest in sources.items():
    assert sha(path) == digest, path
extra = [
    *ROOT.glob("source/*.py"),
    Path("research/whole_job_memory_plan.md"),
    EVAL / "summary.json",
    EVAL / "audit.json",
    EVAL / "result.json",
    EVAL / "source/analyze.py",
    EVAL / "source/audit.py",
]
sources.update({p.as_posix(): sha(p) for p in extra})
datasets = {}
for name in ("wikitext2", "tinystories"):
    path = Path(f"data/{name}_v1")
    manifest = read(path / "manifest.json")
    assert manifest["vocab_size"] == 4096
    for file, digest in manifest["files"].items():
        assert sha(path / file) == digest, file
    datasets[name] = {
        "path": path.as_posix(),
        "manifest_sha256": sha(path / "manifest.json"),
        "manifest": manifest,
    }
    (ROOT / name).mkdir(exist_ok=False)
protocol = {
    "sources": sources,
    "evaluation_policy": policy,
    "datasets": datasets,
    "seeds": [61, 73, 89],
    "trials": 12,
    "steps_per_trial": 800,
    "primary_metric": "independently audited native full-validation NLL",
    "fidelity_study_reopened": False,
    "maintained_files": read(EVAL / "protocol.json")["maintained_files"],
}
text = json.dumps(protocol, indent=2) + "\n"
(ROOT / "protocol.json").write_text(text)
for name in datasets:
    (ROOT / name / "protocol.json").write_text(text)
print(f"Frozen {len(sources)} files and both datasets; evaluation={policy}; 12 fresh trials")
