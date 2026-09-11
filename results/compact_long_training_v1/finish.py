"""Summarize all long-run gates and seal the full evidence."""

import gzip
import json
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha

ROOT = Path("results/compact_long_training_v1")
p, r, a, s = [
    read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json", "summary.json")
]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
previous = read("results/compact_training_v1/receipt.json")
for path, digest in previous["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else path
    )
    assert sha(target) == digest
for dataset in p["datasets"].values():
    folder = Path(dataset["path"])
    assert sha(folder / "manifest.json") == dataset["manifest_sha256"]
    hashes(
        {
            (folder / name).as_posix(): digest
            for name, digest in dataset["manifest"]["files"].items()
        }
    )
for stage in ("prepare", "study", "prepare_audit", "audit", "analyze", "plot"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
assert a["passed"]
rows = [
    f"| {c['dataset']} | {c['arm']} | {c['nll']:.6f} | {c['peak_mib']:.3f} | {c['event_ms']:.3f} | {c['wall_ms']:.3f} |"
    for c in s["metrics"]
]
comparisons = []
for c in s["comparisons"]:
    failed = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    ratios = c["ratios"]
    comparisons.append(
        f"| {c['dataset']} | {c['control']} | {100 * (ratios['nll'] - 1):+.3f}% | {100 * (1 - ratios['peak_mib']):.2f}% | {ratios['event_ms']:.4f}x | {ratios['wall_ms']:.4f}x | {failed} |"
    )
status = (
    "PASS: all fixed long-run gates"
    if s["passed"]
    else "FAIL: the candidate does not pass every fixed long-run gate"
)
next_step = (
    "A separate multiseed/scale replication is earned; this one-seed study is insufficient for a broad claim."
    if s["passed"]
    else "Expansion to more seeds is not earned under this protocol. Investigate the failed dimensions before allocating another training matrix. Memory savings remain scoped observations, not a rescued quality/runtime gate."
)
ge = max(v["check"]["replay"]["error"]["global_relative_l2"] for v in a["cases"])
score = max(v["relative_error"] for row in a["cases"] for v in row["check"]["checkpoint_scores"])
phases = sum(len(row["measurement"]["memory_phases"]) for row in r["cases"])
report = f"""# H126: long training with native and chunked controls

**{status}.** These are fresh-from-initialization runs of one deliberately
selected seed per corpus, including the historical difficult WikiText seed.
All independent initialization, gradient, data and scoring audits pass.

| Corpus | Arm | Final validation NLL | Complete-job peak MiB | Median event ms | Median wall ms |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Candidate comparisons (positive NLL change means worse):

| Corpus | Control | NLL change | VRAM saved | Event ratio | Wall ratio | Failed gates |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

The combined arm saves 38.94%/39.77% CUDA allocation against native on
WikiText/TinyStories, and 11.46%/12.44% against chunks. TinyStories passes both
comparisons. WikiText exceeds the native quality limit: +1.071% NLL against
a predeclared +1% maximum. Its observed native-relative event overhead is
30.21%, above 15%, and the native arm's timing stability is 1.170, above 1.15.
That instability limits runtime interpretation and fails both WikiText
comparisons under the corpus-wide stability rule; it does not erase the
separate quality failure. This single seed cannot establish a systematic
quality penalty or isolate which numerical change caused it.

![Validation trajectories and resources](figures/compact_long_training.png)

The native arm uses ordinary RMSNorm, native FP32 classifier loss and standard
whole-block checkpointing. The chunks arm uses chunked FP32 loss and CPU
checkpoint-input storage. The combined arm additionally uses compact RMSNorm
and gradient staging/restoration. Every gradient returns to CUDA before the
unchanged global clipping/default AdamW update. BF16 evaluation uses ordinary
RMSNorm. There are no parameter reductions: all models have 9,099,648 parameters.

## Evidence and limits

Six 800-update runs use the original two step-zero states, identical optimizer
settings and identical batch order per corpus. Arm order reverses across
corpora. Total **4,800 updates, 19,660,800 training targets and 4,812 backwards**:
4,800 training, six initial probes, six independent replays. Initial-probe and
replay targets are diagnostics, not extra training exposure. No run was repeated.

The audit regenerates both original initializations exactly, reconstructs all
4,800 training batches, verifies all 18 model/optimizer states at steps 200/400/800,
and checks 20 native validation scores. Maximum gradient replay relative L2 is
{ge:.3e}; maximum trained-checkpoint score discrepancy is {score:.3e}. All states
and gradients are finite. The 24 tensor artifacts are six initial gradients and
18 trained states. Sources, datasets and all 61 maintained file hashes verify.

Complete-job peaks include warmup, initial probes, all transfers/updates,
evaluation, diagnostics and serialization. All {phases} memory intervals and
16 zero GPU allocator boundaries are recorded. Host pinned active/allocated
statistics are separate from CUDA allocation; pinned memory is additional RAM.
Pinned allocated peak reaches 112.03 MiB. The host allocator cache persists
across cases, so later control peaks can include buffers cached by earlier
arms; these are process peaks, not isolated per-arm host ownership.
CUDA allocator figures exclude driver/context memory and other processes.
Runtime uses steps 21–800, with mean, median, variance and block stability saved.
The figure shows all three trained validation checkpoints; initial scores are
also retained in machine-readable results.

The old H117 training and audit functions and H125 adapter are reused without
editing their implementation. H125's short continuation pass is preserved;
it cannot substitute for these longer-run gates. H117/H119 failures are not
retroactively resolved. No maintained defaults change and no algorithmic
novelty or architectural breakthrough is claimed.

## Next decision

{next_step}
The broader VRAM/quality and parameter-efficient FFN research goal remains open.

[Prospective plan](compact_long_training_plan.md),
[summary](../results/compact_long_training_v1/summary.json),
[evidence receipt](../results/compact_long_training_v1/receipt.json).
"""
Path("research/compact_long_training_results.md").write_text(report, encoding="utf-8", newline="\n")
current = Path("research/CURRENT_STATE.md")
assert sha(current) == p["current_state_before"]
head, rest = current.read_text(encoding="utf-8").split("\n\n", 1)
note = f"""## Latest: long compact-training comparison

[H126](compact_long_training_results.md): **{status}.** Six matched runs from
original initialization, 800 updates each, cover native/chunked/combined arms
on both corpora. Independent audits pass. Total 4,800 updates/4,812 backwards;
24 tensor artifacts, no repeated runs, all 61 maintained file hashes unchanged.

{next_step}
One seed per corpus does not establish broad quality or architectural success.
The full research goal remains open. See the report for every comparison.

"""
current.write_text(
    head + "\n\n" + note + rest.replace("## Latest:", "## Previous:", 1),
    encoding="utf-8",
    newline="\n",
)
readme = Path("README.md")
assert sha(readme) == p["readme_before"]
head, rest = readme.read_text(encoding="utf-8").split("\n\n", 1)
intro = f"""Latest: [long-training validation](research/compact_long_training_results.md).
**{status}.** Six 800-update runs and independent audits are complete.
See [current research state](research/CURRENT_STATE.md) for the next decision.

"""
readme.write_text(
    head + "\n\n" + intro + rest.replace("Latest evidence:", "Earlier short-run evidence:", 1),
    encoding="utf-8",
    newline="\n",
)
for path in ROOT.glob("*.log"):
    path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
for path in ROOT.glob("*/runs/*/*"):
    if path.suffix in (".json", ".jsonl"):
        path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
files = [
    *ROOT.glob("*.py"),
    *ROOT.glob("*.json"),
    *ROOT.glob("*.gz"),
    *ROOT.glob("*_exit.txt"),
    ROOT / ".gitignore",
    ROOT / "CURRENT_STATE.before.md",
    ROOT / "README.before.md",
    *ROOT.glob("*/runs/*/*"),
    current,
    readme,
    Path("research/compact_long_training_plan.md"),
    Path("research/compact_long_training_results.md"),
    Path("research/figures/compact_long_training.png"),
]
receipt = dict(
    status="EVIDENCE_VERIFIED",
    long_gate_passed=s["passed"],
    independent_audit_passed=True,
    training_updates=4800,
    training_targets=19660800,
    backwards=4812,
    native_audit_scores=20,
    tensor_artifacts=24,
    maintained_hashes_verified=61,
    repeated_runs=0,
    broad_goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files if f.is_file()},
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
hashes(receipt["files"])
print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
