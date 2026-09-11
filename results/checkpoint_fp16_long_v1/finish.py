"""Publish the fresh FP16 learning qualification and preserve its limits."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/checkpoint_fp16_long_v1")
p, s, a, r = [
    read(ROOT / n) for n in ("protocol.json", "summary.json", "audit.json", "result.json")
]
pre = read(ROOT / "preflight.json")
assert not (ROOT / "receipt.json").exists()
for stage in ("prepare", "preflight", "prepare_audit", "audit", "analyze", "plot"):
    assert (ROOT / (stage + "_exit.txt")).read_text().strip() == "0"
for field in ("sources", "input_hashes", "maintained_files", "checkpoint_hashes"):
    hashes(p[field])
hashes(read(ROOT / "audit_protocol.json")["files"])
assert sha("results/checkpoint_fp16_timing_v1/receipt.json") == p["prior_receipt"]
hashes(read("results/checkpoint_fp16_timing_v1/receipt.json")["files"])
assert pre["passed"] and len(pre["artifacts"]) == 12 and len(pre["boundaries"]) == 13
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in pre["boundaries"])
for item in pre["artifacts"]:
    assert sha(item["path"]) == item["sha256"]
    c = next(
        row["measurement"]
        for row in r["cases"]
        if (row["fixture"]["dataset"], row["fixture"]["seed"], row["arm"])
        == (item["dataset"], item["seed"], "fp16" if item["half"] else "ordinary")
    )
    assert (
        item["tokens_hash"] == c["initial_probe"]["tokens_hash"]
        and item["targets_hash"] == c["initial_probe"]["targets_hash"]
    )
    assert abs(item["loss"] / c["initial_probe"]["loss"] - 1) <= 1e-6
status = (
    "PASS fresh 800-update FP16 checkpoint-storage qualification"
    if s["passed"]
    else "FAIL fresh FP16 long-training gate"
)
rows = []
for c in s["comparisons"]:
    v = c["ratios"]
    failed = ", ".join(k for k, ok in c["gates"].items() if not ok) or "none"
    rows.append(
        f"| {c['dataset']} | {c['seed']} | {100 * (1 - v['peak_mib']):.2f}% | {v['event_ms']:.4f} | {v['wall_ms']:.4f} | {v['nll']:.6f} | {failed} |"
    )
metrics = []
for m in s["metrics"]:
    metrics.append(
        f"| {m['dataset']} | {m['seed']} | {m['arm']} | {m['peak_mib']:.3f} | {m['host_mib'] * 2**20:.0f} | {m['event_ms']:.3f} | {m['nll']:.6f} | {m['stability']:.4f} |"
    )
aggregates = []
for g in s["aggregates"]:
    v = g["stats"]
    aggregates.append(
        f"| {g['dataset']} | {g['arm']} | {v['nll']['mean']:.6f} | {v['nll']['median']:.6f} | {v['nll']['variance']:.3e} | {v['event_ms']['mean']:.3f} |"
    )
exact_diagnostic_failures = sum(
    not v["check"]["replay"]["passed"] for v in a["cases"] if v["arm"] == "fp16"
)
report = f"""# H160: FP16 checkpoint storage from initialization

**{status}.** Twelve continuous fresh runs, 9,600 optimizer updates and
78,643,200 training targets. Independent initialization, checkpoint, native-score
and separately qualified gradient audit: **{a["passed"]}**. The broader goal
remains open; this is one model scale and an 800-update validation result.

| Corpus | Seed | GPU allocation saved | Complete CUDA ratio | Wall ratio | Final NLL ratio | Failed gates |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

![All-seed convergence and resources](figures/checkpoint_fp16_long.png)

## What was tested

H159's short continuations passed timing and resource gates from mature models.
Here both arms start from the same original H117 step-zero weights and empty
Adam state, with fresh native controls. The candidate remains buffered classifier
loss plus eight FP16 GPU checkpoint inputs, restored to FP32 for recomputation.
The original forward is FP32 and the backward is approximate. The H157 codec and
long-training loop are imported unchanged. No parameters or activation families
are added, and maintained model/trainer defaults remain unchanged.

Two small text corpora, seeds 101/113/127, 9,099,648 parameters, width 384, eight
blocks, context 512, training and validation batch 16. FP32, TF32 off, ordinary
attention, four CPU threads, default 8.125MiB cuBLAS workspace, AdamW LR0.0006,
betas0.9/0.95, original weight decay and clipping1. Each run gets an isolated
process; the first arm alternates by fixture. No completed run is repeated.

## Absolute resources and outcomes

| Corpus | Seed | Arm | GPU peak MiB | Pinned host peak bytes | CUDA update ms | Final NLL | Timing stability |
|---|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(metrics)}

Whole-job allocation includes model/moments/data, construction, evaluation,
training, gradient copies, checkpoint saving and diagnostics. Driver/context,
unrelated processes and total CPU RSS are excluded. Complete-update timers
include forward, backward, FP16 casts/restoration, recomputation, clipping and
Adam; sampling, gradient clearing, logging, evaluation and saving are outside
those timers. This does not measure isolated inference memory.

## All-seed statistics and prospective gates

| Corpus | Arm | Mean final NLL | Median | Sample variance | Mean CUDA update ms |
|---|---|---:|---:|---:|---:|
{chr(10).join(aggregates)}

Every seed must satisfy allocation ratio<=0.90, median complete CUDA and wall
ratios<=1.15, final validation NLL ratio<=1.01 and pinned-host peak<=128MiB.
After 20 warmups, each run's three 260-update wall-time means must have max/min
<=1.15. Passive 200ms telemetry must cover timed training. No clock adjustment,
normalization, discarded seed or aggregate rescue. The plot shows all seeds;
its mean+/-SD bands are not confidence intervals. Intermediate checkpoints show
development of differences; only the frozen final-NLL gate decides qualification.

## Independent evidence and gradient semantics

Before long training, all six native/FP16 initial-gradient pairs passed NumPy
FP64 checks: global relative L2<=.002, maximum tensor<=.02, cosine>=.99999 and
loss relative error<=1e-6. Their 12 saved gradients, batch hashes and losses match
the subsequent initial probes. This checks initialization scale, not every later
gradient approximation. Each candidate saves/unpacks 6,408 inputs: 801 backwards
including the initial probe, eight blocks each, with no full input snapshots.

The independent audit regenerates all six initial states bitwise, verifies
all 9,600 sampled batches and every model/Adam/sampler hash at 200/400/800, finite
states and counters, and scores all saved endpoints through native code. There
are 48 study and 42 independent native validation scores, 60 tensor artifacts,
19,332 training memory phases and 56 zero-allocation/reservation GPU boundaries
including preflight. The 9,600 training backwards plus 12 training initial probes,
12 preflight backwards and 12 native audit replays total 9,636; the audit performs
no optimizer updates. It does not independently repeat all 9,600 optimizer steps.

Native controls retain exact replay limits 1e-5 global/1e-4 tensor/1e-6 loss.
For FP16, the old exact replay result remains visible ({exact_diagnostic_failures}
of six fail that exact-gradient diagnostic). A separate NumPy comparison uses
native as the error denominator and applies the declared approximate caps and
cosine. A qualified approximate replay never converts that exact diagnostic into
a pass. Scoring/state audit code remains unchanged. The implementation is not
claimed to compute exact gradients or universally avoid overflow or vanishing
and exploding gradients. All studied losses, norms and states were checked finite.

## Decision and limits

{"This qualifies the fixed recipe over 800 fresh updates on both corpora and all three seeds. It warrants testing another workload/model scale and considering a carefully scoped reusable codec, with its approximate-gradient semantics explicit." if s["passed"] else "Do not promote the recipe as quality-preserving long training. Retain every failing gate and seed, then distinguish numerical failure, learning degradation and timing instability before redesigning."}

The same two language corpora have informed development, so this is not an
untouched final-test or unrelated-domain result. Eight hundred updates do not
prove terminal convergence. There is no parameter reduction, new scalar activation,
novelty or SOTA claim. Activation compression has prior art documented in H157;
the broad architecture/complex-pattern objective remains unresolved. H156's exact
offload result and H157's failed absolute-zero host gate remain unchanged.

[Prospective plan](checkpoint_fp16_long_plan.md),
[machine-readable summary](../results/checkpoint_fp16_long_v1/summary.json),
[independent audit](../results/checkpoint_fp16_long_v1/audit.json).
"""
path = Path("research/checkpoint_fp16_long_results.md")
assert not path.exists()
path.write_text(report, encoding="utf-8")
for target, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = target.read_text(encoding="utf-8")
    assert old.startswith(title)
    rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
    link = (
        "research/checkpoint_fp16_long_results.md"
        if target.name == "README.md"
        else "checkpoint_fp16_long_results.md"
    )
    intro = f"Latest: [H160 fresh FP16 training]({link}): **{status}.**\n12 fresh runs /9,600 updates, independent initialization and native endpoint audit.\nApproximate-gradient qualification is separate from exact replay. Broad goal open.\n\n"
    target.write_text(title + "\n\n" + intro + rest, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    path,
    Path("research/checkpoint_fp16_long_plan.md"),
    Path("research/figures/checkpoint_fp16_long.png"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
write_json(
    ROOT / "receipt.json",
    dict(
        study="H160",
        status="EVIDENCE_VERIFIED",
        gate=status,
        training_updates=9600,
        backwards=9636,
        training_targets=78643200,
        goal_achieved=False,
        files={f.as_posix(): sha(f) for f in files},
    ),
)
print(status)
