"""Readable outcome with all cases, fixed failures and boundary-accounting limits."""

from pathlib import Path

from results.optimizer_memory_v1.source.prepare import ROOT, read


def name(f):
    return (
        ("WikiText" if f["dataset"] == "wikitext2" else "TinyStories")
        + " / "
        + ("chunks" if f["loss_policy"].endswith("chunks") else "native")
    )


def run():
    p, r, a, s = [
        read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json", "summary.json")
    ]
    table = []
    for fixture in p["fixtures"]:
        for mode in p["modes"]:
            c = next(c for c in r["cases"] if c["fixture"] == fixture and c["mode"] == mode)
            optimizer = (
                max(
                    m["peak_allocated_bytes"]
                    for m in c["phases"]
                    if m["phase"].startswith("optimizer_")
                )
                / 2**20
            )
            table.append(
                f"| {name(fixture)} | {mode} | {optimizer:.3f} | {c['peak_allocated_bytes'] / 2**20:.3f} | {c['peak_reserved_bytes'] / 2**20:.1f} | {c['after_score']['nll']:.8f} | {c['timing']['event_sum_ms']['median']:.3f} | {c['timing']['wall_ms']['median']:.3f} |"
            )
    comparisons = []
    for c in s["comparisons"]:
        f = next(f for f in p["fixtures"] if f["label"] == c["fixture"])
        failed = ", ".join(k for k, v in c["gates"].items() if not v)
        comparisons.append(
            f"| {name(f)} | {c['mode']} | {c['memory_ratio']:.6f} | {c['time_ratios']['event_sum_ms']:.4f} | {c['time_ratios']['wall_ms']:.4f} | {100 * (c['nll_ratio'] - 1):+.5f}% | {failed} |"
        )
    phases = []
    for fixture in p["fixtures"]:
        c = next(c for c in r["cases"] if c["fixture"] == fixture and c["mode"] == "default")
        maxima = [
            max(
                m["peak_allocated_bytes"] for m in c["phases"] if m["phase"].startswith(phase + "_")
            )
            / 2**20
            for phase in ("forward", "backward", "clipping", "optimizer")
        ]
        phases.append("| " + name(fixture) + " | " + " | ".join(f"{v:.3f}" for v in maxima) + " |")
    moment_error = max(e["distance"] for c in a["numerical"] for e in c["moment_errors"])
    doc = f"""# H120 — optimizer temporaries are not the complete-job bottleneck

**Decision: both per-tensor and fused AdamW are ELIMINATED for this memory
gate.** They lower the optimizer-phase peak, but backward sets the largest
allocation in every one of the 12 profiles. Neither produces the required
10% complete-job reduction. All numerical, data and scoring audits pass.

Fused AdamW removes roughly 35–36 MiB from the optimizer-phase peak and has
2.7–3.6% lower median summed phase-event time in this instrumented screen.
Complete-job allocation is actually 25 KiB higher. Per-tensor AdamW leaves the
job peak identical and is slower. These measurements prevent promoting an
optimizer-only saving as a model-memory improvement.

![Optimizer and complete-job peaks](figures/optimizer_memory.png)

## Controlled question and scope

The [prospective plan](optimizer_memory_plan.md) follows H119's sensitivity
diagnosis and returns to the primary VRAM objective. It compares three existing
native AdamW implementations: default (foreach/fused unspecified), single
(both false), and fused (fused true, foreach unspecified).
[PyTorch's AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
motivates the test by describing foreach's extra temporary storage. This is
an implementation/resource experiment, not a novel optimizer or FFN.

Two H117 FP32-chunked checkpoints at update800, WikiText-2 and TinyStories
seed101, supply the same saved model, moments and sampler to every arm within
each corpus. Both native and chunked FP32 classifier losses are tested. Mode
order rotates across the four corpus/loss fixtures. Twelve cases each continue
30 updates; 10 are warmup and 20 contribute timing statistics.

The common maintained model is width384/FFN456/eight layers/six heads, B8/T512,
vocabulary4096, **9,099,648 total / 2,801,664 FFN parameters**. Default SDPA,
whole-block checkpointing, FP32 model/gradients/moments, AdamW LR0.0006,
betas(0.9,0.95), epsilon1e-8, matrix decay0.1, no norm-weight decay, norm clip1,
four CPU threads and disabled TF32 stay fixed. The UV-managed Python3.12.9 /
PyTorch2.14.0+cu132 runtime and RTX4070 Laptop GPU are unchanged.

Only optimizer implementation flags differ within each comparison. They are
applied to an in-memory copy before native state loading; no checkpoint file
is edited. All arms use one common profiling loop, maintained data/model/group
code and the existing loss/evaluation adapters. These are correlated
continuations of one seed per corpus, not a multi-seed quality qualification.

## Every case

NLL is the complete streamed BF16 validation score after update830. Timing is
median over updates11–30. Summed CUDA phase-event time excludes gaps between
measured operations; wall time includes synchronization/inventory overhead.
Neither is an uninstrumented throughput benchmark.

| Fixture | Mode | Optimizer peak MiB | Complete-job peak MiB | Reserved MiB | Final NLL | Event sum ms | Wall ms |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Mean, median, sample variance, minimum, maximum and split-half timing stability
are recorded for every case in the [full results](../results/optimizer_memory_v1/result.json.gz).
The [360-row update table](../results/optimizer_memory_v1/updates.csv.gz) retains
each loss, norm, batch hash and individual phase timing.

## Fixed gates: keep the failed comparisons

All four fixtures must pass numerical checks, a 10% complete-job memory saving,
NLL<=1.01×default, event and wall medians<=1.10×default and split-half timing
ratios<=1.15. Every numerical, quality and stability gate passes. Every memory
gate fails. Some single-mode timing gates also fail.

| Fixture | Mode | Job memory ratio | Event-time ratio | Wall-time ratio | NLL change | Failed gates |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

No average or favorable optimizer-only peak overrides these per-fixture rules.
Neither mode earns the planned follow-on uninstrumented resource screen.
The small fused timing signal is retained as an observation, with no default
change or strong speed claim.

## Why the whole-job peak remains

Default-mode maximum allocated MiB by numerical phase:

| Fixture | Forward | Backward | Clipping | Optimizer |
|---|---:|---:|---:|---:|
{chr(10).join(phases)}

Let P be parameter bytes and M the two FP32 moments. Here P=36,398,592 bytes
and M=72,797,184 bytes (34.71 and 69.42 MiB). Persistent moment payload is the
same in all three modes. Removing optimizer temporary storage does not remove
these moments from the backward phase. The job peak is the maximum of phase
peaks, not their sum. Holding other phases fixed, reducing a nonmaximal phase
cannot reduce that maximum. The measured profiles exhibit exactly this case.

Every phase boundary records unique CUDA storage belonging to parameters,
buffers, gradients, moments, step counters, data cache and current batch/loss.
Unattributed live bytes are the allocator total minus that known storage;
they may include workspaces, saved activations and allocator rounding. This is
a **boundary inventory**, not a transient allocation trace: it cannot assign
all allocations inside backward to a particular tensor or operation.
See the [1,884-row phase table](../results/optimizer_memory_v1/phases.csv.gz).

Fused mode places 50 four-byte step counters on CUDA instead of CPU (200 bytes
of payload). The observed complete-job increase is 25,600 bytes, consistent
with 50 allocations rounded to 512-byte blocks. This is a storage/accounting
explanation consistent with the source and measurements; no allocator-address
trace was captured to isolate it further. Reserved memory is reported separately
and is not substituted for allocated memory.

## Independent audit and accounting

The first actual update of each case supplies its saved raw/clipped gradient
and update801 model/moments. A separate NumPy FP64 calculation evaluates
general-step AdamW from the incoming update800 moments, including clipping,
bias correction, decoupled decay and epsilon. All 12 checks pass the frozen
tolerances. Maximum observed errors:

- parameter relative L2: {max(c["parameter_error"]["distance"] for c in a["numerical"]):.6e};
- parameter absolute: {max(c["parameter_error"]["max_absolute"] for c in a["numerical"]):.6e};
- moment relative L2: {moment_error:.6e};
- clipping relative L2: {max(c["clip_error"]["distance"] for c in a["numerical"]):.6e};
- same-fixture first raw-gradient global relative L2: {max(c["global_relative"] for c in a["gradient_comparisons"]):.6e};
- same-fixture maximum tensor relative L2: {max(c["max_tensor_relative"] for c in a["gradient_comparisons"]):.6e};
- native versus streamed final-score relative error: {max(c["relative_error"] for c in a["scores"]):.6e}.

All first-step raw gradient norms are below one (about 0.819 for WikiText and
0.976 for TinyStories), so clipping is inactive in these saved first steps.
Their zero clipping error does not test an active clipping reduction or resolve
H119's failed active CPU clipping comparison.

All 36 new tensor artifact hashes, every first/final state and step counter,
all 360 sampled batches, and final sampler hashes verify. All states are finite.
Twelve independently evaluated native final scores match target counts/order
and pass the fixed1e-6 tolerance. Audit adds no optimizer update or backward.

Budget used: **360 training updates / 1,474,560 targets / 360 backwards**, plus
24 study validation scores and 12 native audit scores. All 1,884 memory intervals
include warmup, construction, validation, diagnostics and serialization. Study
and audit each have 13 zero-allocation/reservation boundaries. Profile wall time
is {r["wall_seconds"]:.3f}s, excluding interpreter startup and independent audit.
There was no failed scientific stage or repeated case. A later read-only
PowerShell display process had an internal CLR failure; its observation note
is retained, and no experiment was repeated.

## Decision and next question

Stop pursuing optimizer temporary storage as the VRAM solution for these
fixtures. H119's older CPU clipping failure and H117's older quality failure
remain unchanged; H120 does not requalify either. This result adds no new
parameter reduction, activation or architectural claim.

Backward is the next target. Whole-block checkpointing still retains block
inputs. Eight FP32 B8/T512/d384 input tensors contain 48 MiB in total, before
considering their actual lifetimes. A separate hypothesis should test selective
host storage of checkpoint inputs, while keeping arithmetic and optimizer
fixed. This is an upper-bound payload estimate, not an observed saving. It
requires actual saved-tensor/lifetime measurements, transfer-cost accounting
and gradient checks before language-quality promotion. Do not offload all
intermediates or alter the model without measuring the tradeoff.

All 152 frozen source hashes and 61 maintained source/config/test/lock hashes
are retained. The existing model tree and defaults remain unchanged. The earlier
116-test pass is historical; maintained tests were not rerun for this isolated
profile. New scripts pass lint. The full VRAM/quality and architectural research
goal remains open.

[Plan](optimizer_memory_plan.md) · [source guide](../results/optimizer_memory_v1/source/README.md) ·
[summary](../results/optimizer_memory_v1/summary.json) ·
[audit](../results/optimizer_memory_v1/audit.json.gz) ·
[verification receipt](../results/verification/optimizer_memory_final_v1.json).
"""
    Path("research/optimizer_memory_results.md").write_text(doc, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    run()
