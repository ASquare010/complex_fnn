"""Freeze the workspace-capacity-only test and verify the failed predecessor."""

import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/blas_workspace_mid_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/blas_workspace_v1/receipt.json")["files"])
prior = read("results/blas_workspace_v1/protocol.json")
for key in ("sources", "input_hashes", "maintained_files"):
    hashes(prior[key])
p = {
    k: v
    for k, v in prior.items()
    if k not in ("sources", "input_hashes", "current_state_before", "readme_before")
}
p.update(
    study="H132",
    current_state_before=sha("research/CURRENT_STATE.md"),
    readme_before=sha("README.md"),
)
files = [*ROOT.glob("*.py"), Path("research/blas_workspace_mid_plan.md")]
p["sources"] = {f.as_posix(): sha(f) for f in files}
inputs = [
    "results/blas_workspace_v1/receipt.json",
    "results/blas_workspace_v1/protocol.json",
    "results/blas_workspace_v1/worker.py",
    "results/blas_workspace_v1/audit.py",
    "results/blas_workspace_v1/common.py",
]
p["input_hashes"] = {f: sha(f) for f in inputs}
for name, path in (("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")):
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
for mode in ("high", "low"):
    for arm in ("native", "reuse"):
        (ROOT / mode / arm / "runs").mkdir(parents=True)
(ROOT / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
