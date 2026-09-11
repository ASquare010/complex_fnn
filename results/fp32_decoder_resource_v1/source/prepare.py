"""Verify the complete H115 evidence and freeze H116 before GPU execution."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")
OLD = Path("results/decoder_gradient_transport_v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert not (ROOT / "protocol.json").exists()
prior_path = Path("results/verification/decoder_gradient_transport_final_v1.json")
receipt, previous = read(prior_path), read(OLD / "protocol.json")
assert receipt["status"] == "PASS" and not receipt["broad_goal_achieved"]
assert receipt["resource_screen_candidates"] == ["fp32_default_block", "fp32_math_block"]
for field in ("files", "tensor_hashes", "local_metric_hashes"):
    for path, digest in receipt[field].items():
        assert sha(path) == digest, path
for path, value in receipt["packed"].items():
    assert sha(path) == value["sha256"], path
for field in ("sources", "maintained_files"):
    for path, digest in previous[field].items():
        assert sha(path) == digest, path
policies = [
    "bf16_default_native",
    "fp32_default_native",
    "fp32_default_chunks",
    "fp32_math_native",
    "fp32_math_chunks",
]
fixtures, checkpoints = [], {}
for index, original in enumerate(previous["fixtures"]):
    fixture = original.copy()
    order = policies[index % 5 :] + policies[: index % 5]
    if index % 2:
        order.reverse()
    fixture["policy_order"] = order
    fixtures.append(fixture)
    path = fixture["checkpoint"]
    assert sha(path) == previous["input_hashes"][path]
    checkpoints[path] = sha(path)
assert len(fixtures) == len(checkpoints) == 6
documents = {}
before = ROOT / "before_documents"
before.mkdir(exist_ok=False)
for name in previous["before_documents"]:
    target = before / (name.replace("/", "__") if name != ".gitignore" else "gitignore")
    target.write_bytes(Path(name).read_bytes())
    documents[name] = dict(sha256=sha(name), preserved_path=target.as_posix())
sources = {
    **previous["sources"],
    **{
        p.as_posix(): sha(p)
        for p in (*ROOT.glob("source/*.py"), Path("research/fp32_decoder_resource_plan.md"))
    },
}
protocol = dict(
    study="H116",
    previous_goal_turn="progress",
    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    sources=sources,
    maintained_files=previous["maintained_files"],
    previous_receipt_sha256=sha(prior_path),
    previous_scientific_records={
        p.as_posix(): sha(p)
        for p in (OLD / "result.json", OLD / "summary.json", OLD / "audit.json")
    },
    before_documents=documents,
    datasets=previous["datasets"],
    fixtures=fixtures,
    checkpoint_hashes=checkpoints,
    policies=policies,
    cases=30,
    steps_per_case=50,
    warmup_steps=20,
    optimizer_updates=1500,
    profile_backward_passes=1530,
    qualification_backward_passes=24,
    planned_audit_backward_passes=24,
    fresh_training_runs=0,
    broad_goal_achieved=False,
)
(ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
(ROOT / "runs").mkdir(exist_ok=False)
print(f"Verified H115; frozen {len(sources)} sources, six checkpoints and 30 H116 cases")
