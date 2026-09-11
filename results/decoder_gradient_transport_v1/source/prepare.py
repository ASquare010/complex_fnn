"""Verify H114 and freeze H115 before any scientific execution."""

import hashlib
import json
from pathlib import Path

ROOT = Path("results/decoder_gradient_transport_v1")
OLD = Path("results/fp32_classifier_profile_v1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT / "protocol.json").exists()
receipt_path = Path("results/verification/fp32_classifier_profile_final_v1.json")
receipt, previous = read(receipt_path), read(OLD / "protocol.json")
assert receipt["status"] == "PASS" and receipt["candidate_qualification"] == "FAILED"
for field in ("files", "tensor_hashes"):
    for path, digest in receipt[field].items():
        assert sha(path) == digest, path
for path, digest in previous["sources"].items():
    assert sha(path) == digest, path
fixtures = [
    f.copy() for f in previous["fixtures"] if f["seed"] == 61 or f["scope"].startswith("full")
]
assert len(fixtures) == 6
cases = read(OLD / "result.json")["cases"]
inputs = {}
for fixture in fixtures:
    path = fixture["checkpoint"]
    assert sha(path) == previous["checkpoint_hashes"][path]
    inputs[path] = sha(path)
    peers = [c for c in cases if c["fixture"]["label"] == fixture["label"]]
    assert len(peers) == 4
    fixture["probe_tokens_sha256"] = peers[0]["initial_probe"]["tokens_hash"]
    fixture["probe_targets_sha256"] = peers[0]["initial_probe"]["targets_hash"]
    fixture["h114_gradient_probes"] = {}
    for case in peers:
        if case["policy"] in ("block_fp32", "chunks_fp32"):
            path, digest = case["initial_probe"]["path"], case["initial_probe"]["sha256"]
            assert sha(path) == digest
            inputs[path] = digest
            fixture["h114_gradient_probes"][case["policy"]] = path
before = ROOT / "before_documents"
before.mkdir(exist_ok=False)
documents = {}
for name in previous["before_documents"]:
    target = before / (name.replace("/", "__") if name != ".gitignore" else "gitignore")
    target.write_bytes(Path(name).read_bytes())
    documents[name] = dict(sha256=sha(name), preserved_path=target.as_posix())
sources = {
    **previous["sources"],
    **{
        p.as_posix(): sha(p)
        for p in (*ROOT.glob("source/*.py"), Path("research/decoder_gradient_transport_plan.md"))
    },
}
protocol = dict(
    study="H115",
    previous_goal_turn="progress",
    sources=sources,
    maintained_files=previous["maintained_files"],
    datasets=previous["datasets"],
    input_hashes=inputs,
    fixtures=fixtures,
    previous_receipt_sha256=sha(receipt_path),
    previous_scientific_records={
        p.as_posix(): sha(p)
        for p in (OLD / "result.json", OLD / "summary.json", OLD / "completion_audit.json")
    },
    before_documents=documents,
    conditions=30,
    classifier_backward_passes=60,
    decoder_backward_passes=240,
    qualification_backward_passes=20,
    optimizer_updates=0,
    broad_goal_achieved=False,
)
(ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
(ROOT / "conditions").mkdir(exist_ok=False)
print(f"Frozen {len(sources)} sources, six model fixtures and 30 diagnostic conditions")
