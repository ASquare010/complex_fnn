"""Publish fresh batch16 long-training evidence with unchanged per-seed gates."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/batch_scale_long_v1")
p, s, a, r = [
    read(ROOT / name) for name in ("protocol.json", "summary.json", "audit.json", "result.json")
]
assert not (ROOT / "receipt.json").exists()
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
for values in p["base_verified"].values():
    hashes(values)
hashes(read(ROOT / "audit_protocol.json")["files"])
assert sha("results/augmentation_barrier_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/augmentation_barrier_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
for stage in ("prepare", "prepare_audit", "audit", "analyze", "plot"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
assert all(c["fixture"]["batch"] == 16 for c in r["cases"])
assert sum(c["measurement"]["training_targets"] for c in r["cases"]) == 78643200
status = (
    "PASS fresh batch16 long-training gate"
    if s["passed"]
    else "FAIL fresh batch16 long-training gate"
)
comparisons = []
for c in s["comparisons"]:
    v = c["ratios"]
    failed = ", ".join(k for k, ok in c["gates"].items() if not ok) or "none"
    comparisons.append(
        f"| {c['dataset']} | {c['seed']} | {100 * (1 - v['peak_mib']):.2f}% | {v['event_ms']:.4f} | {v['wall_ms']:.4f} | {v['nll']:.6f} | {failed} |"
    )
metrics = []
for m in s["metrics"]:
    c = next(v["measurement"] for v in r["cases"] if v["index"] == m["index"])
    metrics.append(
        f"| {m['dataset']} | {m['seed']} | {m['arm']} | {m['peak_mib']:.2f} | {m['host_mib']:.2f} | {m['event_ms']:.2f} | {m['nll']:.5f} | {m['stability']:.4f} | {c['case_wall_seconds']:.1f} |"
    )
aggregates = []
for g in s["aggregates"]:
    v = g["stats"]
    aggregates.append(
        f"| {g['dataset']} | {g['arm']} | {v['nll']['mean']:.6f} | {v['nll']['median']:.6f} | {v['nll']['variance']:.3e} | {v['peak_mib']['mean']:.2f} | {v['event_ms']['mean']:.2f} |"
    )
next_step = (
    "The maintained opt-in helper now has fresh 800-update evidence at batch16 across both corpora and all three seeds. Keep the scope explicit; test a larger model or another precision/workload before generalizing. This is a memory optimization and does not meet the separate architectural parameter-reduction target."
    if s["passed"]
    else "Do not claim long-run qualification. Keep every failed seed and identify whether quality, timing stability, transfer cost or memory broke the gate before allocating another run. H142 remains a short-run result; this experiment does not rewrite H138 or change acceptance limits."
)
report = f"""# H156: maintained memory helper under fresh batch16 long training

**{status}.** Independent initialization, native-gradient and checkpoint audit:
**{a["passed"]}**. Completed all 12 continuous 800-update runs. Each seed must pass;
aggregate means cannot rescue a failure. The full VRAM/quality research goal remains
open, and no new activation, parameter reduction or SOTA claim is made.

| Corpus | Seed | GPU allocation saved | Complete CUDA ratio | Wall ratio | Final NLL ratio | Failed gates |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

![All-seed convergence and resources](figures/batch_scale_long.png)

## What this resolves

H142 found roughly 24% less allocated VRAM with roughly 5% update-time overhead
in short batch16 continuations. That did not establish fresh long training.
H156 starts both arms from the original empty-Adam step-zero states and runs
800 updates continuously at batch16. H138's earlier batch8 failure is unchanged.
These are the same two small text corpora, not a new domain or independent
benchmark selection. The candidate was fixed before these trajectories.

The ordinary arm uses native loss and keeps checkpoint inputs on GPU. The
candidate calls maintained buffer_model_loss and offload_checkpoint_inputs with
last_n_blocks=4, wrapping blocks 4-7. Both use the same 9,099,648-parameter
Transformer: d384, eight checkpointed blocks, context512 and tied classifier.
FP32, disabled TF32, ordinary attention, default 8.125 MiB cuBLAS workspace,
four CPU threads, AdamW LR0.0006/betas(0.9,0.95), unchanged decay groups and
clipping at 1. Parameters and maintained source code are unchanged.

Both training and validation batches are 16. H142 validated with batch8, so its
scores and timings are not direct paired baselines for this study. All present
quality comparisons use identical validation batches and splits. Each fixture's
arm order alternates, giving three ordinary-first and three candidate-first pairs.
No completed run was reordered, discarded, selected or repeated for a better score.

## Per-run resources and quality

| Corpus | Seed | Arm | GPU peak MiB | Pinned host peak MiB | CUDA update ms | Final NLL | Timing stability ratio | Whole-case seconds |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

Whole-case time includes setup, validation, saving and diagnostics, but excludes
process startup and the following cleanup boundary. Primary update timers include
forward/backward, clipping, Adam, transfers and recomputation. They exclude sampling,
gradient clearing, serialization, scoring and logging. All such phases remain in
whole-job CUDA peak accounting. Allocated/reserved peaks exclude driver/context
and other processes. Pinned-host allocation is reported separately, not total RSS.
These are training resource measurements, not isolated inference-memory results.

## All-seed statistics and frozen gates

| Corpus | Arm | Mean final NLL | Median final NLL | Sample variance | Mean GPU peak MiB | Mean CUDA update ms |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(aggregates)}

Every pair must pass allocated memory ratio <=0.90, complete CUDA and wall update
median ratios <=1.15, final NLL ratio <=1.01 and candidate pinned-host peak <=128 MiB.
Each run must also pass max/min <=1.15 across its three 260-update wall-mean blocks
after 20 warmups. Telemetry coverage and full numerical audits are mandatory.
All gates match the prior protocol; clocks and times are not normalized. Passive
200ms telemetry is retained, missing sensors remain missing, and no power or
clock settings were changed. Timing is a measured property of this hardware/run,
not an architecture-independent guarantee. Raw phase timings include forward,
backward and optimizer components.

## Independent verification

9,600 updates and 78,643,200 targets; 12 initial gradient probes and 12 native
replays give 9,624 backwards. There are 48 study validation scores and 42 independent
native scores, covering initial states and all 200/400/800 checkpoints. All 48
tensor artifacts, 19,332 memory intervals and 43 zero allocator boundaries verify.
The independent audit regenerates all six original initial states exactly,
checks every training batch and sampler state, validates model/Adam hashes and
finite moments/counters, and scores every saved checkpoint without the custom loss
or offload path. It does not rerun all optimizer updates independently.

Initial numerical caps remain global gradient relative L2 <=1e-5, per-tensor
relative L2 <=1e-4, and relative loss/score error <=1e-6. Ordinary attention is
nondeterministic, so passing is bounded numerical agreement, not bitwise trajectory
identity. All training losses, gradient summaries and layer diagnostics remain in
the artifacts. Source, input, library, checkpoint and maintained-code hashes verify.
UV-managed Python and the installed CUDA PyTorch environment are recorded per run.

## Decision

{next_step}

A separate [checkpoint-compression design note](checkpoint_compression_note.md)
records a possible follow-up; it was not implemented or tested in this experiment.

This is one model scale and 800 updates on two text corpora, not final convergence
or cross-domain generality. The reusable API and its restrictions are documented
in [training_memory_usage.md](training_memory_usage.md). All historical failures
remain intact, and default model/trainer behavior is unchanged.

[Prospective plan](batch_scale_long_plan.md),
[machine-readable summary](../results/batch_scale_long_v1/summary.json),
[receipt](../results/batch_scale_long_v1/receipt.json).
"""
report_path = Path("research/batch_scale_long_results.md")
report_path.write_text(report, encoding="utf-8")
for path, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = path.read_text(encoding="utf-8")
    assert old.startswith(title)
    if path.name == "README.md":
        assert sha(path) == p["readme_before"]
        rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H156 fresh batch16 long training](research/batch_scale_long_results.md):\n**{status}.** 12 continuous runs, three seeds and two corpora; independent audit {a['passed']}.\nMemory optimization only; the broad FFN research goal remains open.\n\n"
    else:
        assert sha(path) == p["current_state_before"]
        rest = old[len(title) :].lstrip().replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: fresh batch16 memory-helper validation\n\n[H156](batch_scale_long_results.md): **{status}**.\n12 fresh runs / 9,600 updates / 78,643,200 targets. All saved endpoints and\ninitial native gradients audited: {a['passed']}. Maintained helper source unchanged.\n{next_step}\nBroad research goal stays open.\n\n"
    path.write_text(title + "\n\n" + intro + rest, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    report_path,
    Path("research/batch_scale_long_plan.md"),
    Path("research/checkpoint_compression_note.md"),
    Path("research/figures/batch_scale_long.png"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
receipt = dict(
    study="H156",
    status="EVIDENCE_VERIFIED",
    gate=status,
    goal_achieved=False,
    training_updates=9600,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
