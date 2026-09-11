"""Publish a bounded finding and verify evidence without another GPU run."""

import gzip
import json
from pathlib import Path

from results.backward_allocation_v1.analyze import ROOT, read
from results.backward_allocation_v1.payload import analyze
from results.checkpoint_input_offload_v1.source.prepare import hashes, sha

p, r, s = [read(ROOT / n) for n in ("protocol.json", "result.json", "payload_summary.json")]
for field in ("sources", "inputs", "maintained_files"):
    hashes(p[field])
hashes(read(ROOT / "reanalysis_protocol.json")["files"])
assert [analyze(c) for c in r["cases"]] == s["rows"]
assert len(r["cases"]) == 4 and sum(c["backwards"] for c in r["cases"]) == 12
assert all(c["training_updates"] == 0 and c["state_unchanged"] for c in r["cases"])
assert all(
    c["gradient_global"] <= 1e-5 and c["gradient_max_tensor"] <= 1e-4 and c["loss_error"] <= 1e-6
    for c in r["cases"]
)
assert len(r["boundaries"]) == 5 and all(
    b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"]
)
assert s["original_gate"] == "FAIL_ALLOCATED_ACCOUNTING"
for mode in ("prepare", "trace", "inspect_sizes", "payload"):
    assert (ROOT / (mode + "_exit.txt")).read_text().strip() == "0"
assert (ROOT / "analyze_exit.txt").read_text().strip() != "0"
assert (ROOT / "inspect_events_exit.txt").read_text().strip() != "0"
prior = read(Path("results/verification/checkpoint_input_offload_final_v1.json"))
for path, digest in prior["files"].items():
    assert (
        sha(ROOT / "CURRENT_STATE.before.md" if path == "research/CURRENT_STATE.md" else path)
        == digest
    )
for field in ("tensor_hashes", "local_metric_hashes"):
    hashes(prior[field])
for path, info in prior["packed"].items():
    assert sha(path) == info["sha256"]
table = []
for c, row in zip(r["cases"], s["rows"], strict=True):
    interval = (
        "after block 0 recomputation" if c["mode"] == "offload" else "after block 7 recomputation"
    )
    assert row["previous"] == ("block_0_exit" if c["mode"] == "offload" else "block_7_exit")
    table.append(
        f"| {c['dataset']} | {c['mode']} | {row['requested_peak'] / 2**20:.3f} | {row['allocated_peak'] / 2**20:.3f} | {interval} |"
    )
maximum = max(c["gradient_global"] for c in r["cases"])
max_tensor = max(c["gradient_max_tensor"] for c in r["cases"])
report = f"""# H122: backward allocation diagnostic

The **requested-payload peak moves from block 7 backward to block 0 backward**
when checkpoint inputs are offloaded, in both corpora. Blocks are indexed 0–7.
This is evidence of a changing memory bottleneck, not proof of the exact
allocated-VRAM peak's location. The original exact-allocation gate **failed**.

| Corpus | Input storage | Requested peak MiB | Allocated peak MiB | Requested-peak interval |
|---|---|---:|---:|---|
{chr(10).join(table)}

Native peaks occur after block 7 recomputation and before block 6 begins.
Offloaded peaks occur after block 0 recomputation and before backward finishes.
The trigger's Python stack does not resolve the native backward operator.
Do not assign it to a specific attention, FFN or gradient tensor.

The original analyzer incorrectly treated event requested bytes as allocated
block sizes. Rounded blocks and unsplit remainders make them differ. Snapshot
marker indices also needed +1: a returned snapshot excludes its own history
event. The frozen analyzer and failure log remain intact. Separate offline
reanalysis exactly reconciles requested bytes with final active requested
sizes, validates all 76 markers and records site groups at the requested peak.
It does **not** rescue the original allocated-peak attribution gate. Some
grouped allocation sites remain unidentified; that category includes persistent
storage and must not be called wholly temporary backward memory.

Four cases used the same two trained H121 checkpoints, chunked FP32 loss,
resident Adam moments/data and native versus offloaded checkpoint inputs.
Two warmup backwards plus one traced backward each: **12 backwards, zero
optimizer updates, 49,152 diagnostic targets**. No training-quality or timing
claim. All four gradients match H121's independently audited references:
maximum global relative L2 {maximum:.3e}; maximum tensor relative L2
{max_tensor:.3e}. Model/moment states are unchanged and all five GPU boundaries
are zero. H121's original four-fixture gate remains failed.

A CPU inspection encountered a transient JSON integer-decoding error; a fresh
read of all four traces succeeded. Both logs remain. No completed GPU case
was repeated. This does not diagnose the recurrent runtime failures. The first report pass had a path-type error before writing output; it was corrected without rerunning any scientific stage. Its failure log remains.

**Next hypothesis:** completed parameter gradients may contribute to the late
backward peak after input offloading. A separate storage inventory can test
that before attempting gradient offload or altered scheduling. Do not infer
their contribution from the unidentified stack bucket. No maintained code or
default changes; the broader VRAM, parameter-efficiency and quality goal is open.

[Plan](backward_allocation_plan.md), [offline reanalysis](backward_allocation_reanalysis.md),
[evidence](../results/backward_allocation_v1/receipt.json).
The trace uses the existing [PyTorch memory-history API](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html);
this is diagnostic work, not algorithmic novelty.
"""
Path("research/backward_allocation_results.md").write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
old = current.read_text(encoding="utf-8")
head, rest = old.split("\n\n", 1)
note = """## Latest: backward payload peak shifts after checkpoint-input offload

[H122](backward_allocation_results.md) completes 12 backwards, zero updates.
Requested-payload peaks move from block 7 backward to block 0 backward on both
corpora after input offloading. All four gradient comparisons pass. Exact
allocated-peak attribution remains **FAILED**: event requested sizes differ
from allocator block sizes. Offline requested-byte accounting reconciles;
it does not rescue that gate. No GPU case was repeated or default changed.
Next test: inventory completed parameter gradients at the late backward peak
before designing another storage change. The broad research goal remains open.

"""
current.write_text(
    head + "\n\n" + note + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
for path in [*ROOT.glob("*.log"), ROOT / "result.json"]:
    path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.gz"),
    *ROOT.glob("*_exit.txt"),
    ROOT / "CURRENT_STATE.before.md",
    *ROOT.glob("runs/*/*"),
    current,
    *[
        Path("research") / n
        for n in (
            "backward_allocation_plan.md",
            "backward_allocation_reanalysis.md",
            "backward_allocation_results.md",
        )
    ],
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    scientific_gate="FAIL_ALLOCATED_ACCOUNTING",
    requested_endpoint_accounting=True,
    gradient_checks_passed=True,
    backwards=12,
    optimizer_updates=0,
    diagnostic_targets=49152,
    gradient_artifacts=4,
    gpu_case_repeats=0,
    maintained_hashes_verified=61,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
