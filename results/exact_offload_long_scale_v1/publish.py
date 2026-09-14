"""Publish H163's sustained result without generalizing beyond the experiment."""

import statistics as st
from pathlib import Path

from results.exact_offload_long_scale_v1.study import read, sha, write

ROOT = Path("results/exact_offload_long_scale_v1")
for stage in ("run", "audit", "plot"):
    assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
p, s, q = [read(ROOT / name) for name in ("protocol.json", "summary.json", "qualification.json")]
for path, digest in p["hashes"].items():
    assert sha(path) == digest
rows, host, statistics, stats_rows = [], [], {}, []
for pair in s["pairs"]:
    r = pair["ratios"]
    gates = dict(pair["gates"])
    gates.update(next(v for v in q["pairs"] if v["seed"] == pair["seed"])["gates"])
    failed = ", ".join(k for k, v in gates.items() if not v) or "none"
    rows.append(
        f"| {pair['seed']} | {100 * (1 - r['memory']):.2f}% | {r['cuda']:.4f} | {r['wall']:.4f} | {r['nll']:.6f} | {failed} |"
    )
for run in s["runs"]:
    peak = next(v for v in q["peaks"] if v["index"] == run["index"])
    host.append(
        f"| {run['seed']} | {run['arm']} | {run['memory']['allocated'] / 2**20:.2f} | {peak['working_set'] / 2**20:.2f} | {peak['private_commit'] / 2**20:.2f} | {run['final_score']['nll']:.6f} | {run['stability']:.4f} |"
    )
for arm in ("ordinary", "helper"):
    values = [r["final_score"]["nll"] for r in s["runs"] if r["arm"] == arm]
    statistics[arm] = dict(
        mean=st.mean(values), median=st.median(values), sample_variance=st.variance(values), n=3
    )
    stats_rows.append(
        f"| {arm} | {st.mean(values):.8f} | {st.median(values):.8f} | {st.variance(values):.8g} |"
    )
write(ROOT / "seed_statistics.json", statistics)
status = (
    "PASS 800-update larger-scale qualification"
    if q["passed"]
    else "FAIL 800-update larger-scale qualification"
)
report = f"""# H163: sustained exact memory-helper qualification at larger scale

**{status}.** Six fresh800-update runs;4,800 optimizer updates/4,806 backwards.
Three WikiText2 seeds,22,163,968 parameters, FP32. The model and trainable parameter
count are identical between arms. This is a scoped memory optimization, not a
new activation, parameter-reduction result or breakthrough.

| Seed | Allocated GPU saved | CUDA update ratio | Wall ratio | Final NLL ratio | Failed gates |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

![All-seed convergence and resources](figures/exact_offload_long_scale.png)

## Fixed recipe and measurements

Width512, hidden608,12 layers/eight heads, batch16/context512, ordinary attention,
TF32off, four CPU threads, whole-block checkpointing. AdamW lr0.0006, beta(.9,.95),
eps1e-8, matrix decay.1/vector decay0, gradient clipping1, no learning-rate schedule.
Native classifier CE is compared with unchanged buffered CE and exact CPU offload
of the last four block inputs. Seeds401/409/419 use arm orderAB/BA/AB. Initialization
and all sampled batches are identical within each pair.

GPU peaks include construction, resident data, initial gradient probe, full FP32
chunked validation initially and at200/400/800, all complete updates and checkpoint
serialization. Windows lifetime peak working set and private commit include imports
and CPU serialization through worker completion; they are distinct from pinned RAM
and from whole-system RAM. See H162 for the validated native counter definitions.

| Seed | Arm | GPU allocated MiB | Host peak WS MiB | Host peak commit MiB | Final NLL | Timing block max/min |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(host)}

All-seed final validation NLL statistics:

| Arm | Mean | Median | Sample variance |
|---|---:|---:|---:|
{chr(10).join(stats_rows)}

The figure's training panel uses a25-update moving average, then mean±sampleSD
across all three seeds; the band is not a confidence interval. The validation
panel shows every seed at all four scoring points. No best-seed selection.
Raw per-update forward/backward/optimizer/wall times and their mean, median and
sample variance are retained in the summary. Passive GPU telemetry is retained.

## Prospective gates and audit

Each seed must save at least10% allocated GPU peak; complete CUDA/wall medians
must be<=1.15x native; final validation NLL<=1.01x native. Pinned allocation<=128MiB;
helper minus native host peak working set and private commit each<=256MiB, with
both arms' peaks<=8GiB. Discard20 timing warmups and compare three260-update
wall means: max/min<=1.15; max timed wall/median<=5. No clock normalization,
interruption trimming, aggregate rescue or post-result gate change is permitted.

Independent audit regenerates every initial model and all4,800 sampled batches,
checks sampler states at200/400/800, verifies all saved hashes and compares full
initial gradients in NumPy FP64 (global relativeL2<=1e-5, max tensor<=1e-4,
initial relative loss<=1e-6). Native full-classifier scoring checks all initial and
saved endpoint scores:24 audit scores, relative NLL tolerance1e-6. Each rolling
resume state is checked against final model/sampler hashes, finite optimizer
moments, optimizer settings and800-step counters. Every GPU boundary returns to0.
The audit does not independently repeat all4,800 optimizer steps.

Weights at200/400/800 and initial gradients remain immutable. The complete model,
Adam and sampler checkpoint is rolling: its earlier contents are replaced at the
next endpoint, avoiding redundant optimizer archives. The final resume.pt files
contain step800 state. This study had no execution retry.

## Decision and limits

{"The exact helper qualifies for800updates on this larger model and corpus across all three seeds. It remains opt-in and scoped to the tested FP32 recipe. Further evidence is needed for other workloads, precisions, domains and convergence endpoints." if q["passed"] else "Do not extend sustained qualification to this scale. Preserve the exact failed gates and seed outcomes before diagnosing or redesigning; short H161/H162 passes do not override this result."}

Eight hundred updates do not prove terminal convergence. WikiText2 has informed
prior development and is not an untouched final-test corpus. The same architecture
is used in both arms; richer-neuron parameter efficiency and new-architecture
capability remain unresolved. The broad research goal stays open.

[Prospective plan](exact_offload_long_scale_plan.md),
[per-run numerical/resource evidence](../results/exact_offload_long_scale_v1/summary.json),
[host and combined gates](../results/exact_offload_long_scale_v1/qualification.json).
"""
path = Path("research/exact_offload_long_scale_results.md")
assert not path.exists()
path.write_text(report, encoding="utf-8")
for target, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = target.read_text(encoding="utf-8")
    before = ROOT / (
        "README.before.md" if target.name == "README.md" else "CURRENT_STATE.before.md"
    )
    with before.open("x", encoding="utf-8") as f:
        f.write(old)
    assert old.startswith(title)
    rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
    link = (
        "research/exact_offload_long_scale_results.md"
        if target.name == "README.md"
        else "exact_offload_long_scale_results.md"
    )
    target.write_text(
        title
        + f"\n\nLatest: [H163 sustained larger-model test]({link}): **{status}.**\nSix fresh runs/4,800 updates; all endpoint scores, batches and resume states audited.\nThe architecture and broad research goal remain unresolved.\n\n"
        + rest,
        encoding="utf-8",
    )
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "publish.log", "publish_exit.txt")
]
files += [
    path,
    Path("research/exact_offload_long_scale_plan.md"),
    Path("research/figures/exact_offload_long_scale.png"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
write(
    ROOT / "receipt.json",
    dict(
        status=status,
        passed=q["passed"],
        optimizer_updates=4800,
        backwards=4806,
        goal_achieved=False,
        hashes={f.as_posix(): sha(f) for f in files},
    ),
)
print(status)
