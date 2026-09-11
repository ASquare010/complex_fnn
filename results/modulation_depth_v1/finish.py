"""Publish audited depth/recomputation measurements without changing model defaults."""

import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/modulation_depth_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "audit_exit.txt").read_text().strip() == "0"
assert a["status"] == "VERIFIED"
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/split_modulation_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/split_modulation_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
trace_rows = []
for t in p["traces"]:
    trace_rows.append(
        f"| {t['arm']} | {t['unique_bytes']['activation'] / 2**20:.1f} | {t['unique_bytes']['input'] / 2**20:.1f} |"
    )
rows = []
for g in a["aggregates"]:
    s = g["statistics"]
    rows.append(
        f"| {g['batch']} | {g['arm']} | {s['peak_mib']['mean']:.3f} | {s['event_ms']['mean']:.3f} | {s['wall_ms']['mean']:.3f} |"
    )
comp_rows = []
for arm, passed in a["candidates"].items():
    for b in p["batches"]:
        for control in ("full_gelu", "full_swiglu"):
            cs = [
                c
                for c in a["comparisons"]
                if c["arm"] == arm and c["batch"] == b and c["control"].split(":")[0] == control
            ]
            assert len(cs) == 3
            intervals = {
                k: f"{min(c['ratios'][k] for c in cs):.3f}–{max(c['ratios'][k] for c in cs):.3f}"
                for k in ("peak", "event_ms", "wall_ms")
            }
            failures = (
                ", ".join(
                    k
                    for k in ("parameters", "peak", "timing", "stability")
                    if any(not c["gates"][k] for c in cs)
                )
                or "none"
            )
            comp_rows.append(
                f"| {b} | {arm} | {control} | {intervals['peak']} | {intervals['event_ms']} | {intervals['wall_ms']} | {failures} |"
            )
within = []
within_rows = []
for b in p["batches"]:
    for base in ("full_gelu", "full_swiglu", "budget_gelu", "shared_gate", "split_gate"):
        ratios = []
        for seed in p["seeds"]:
            e = next(
                c
                for c in r["cases"]
                if (c["batch"], c["seed"], c["arm"]) == (b, seed, base + ":eager")
            )
            k = next(
                c
                for c in r["cases"]
                if (c["batch"], c["seed"], c["arm"]) == (b, seed, base + ":checkpoint")
            )
            ratios.append(
                dict(
                    seed=seed,
                    peak=k["peak_bytes"] / e["peak_bytes"],
                    event_ms=k["timing"]["event_ms"]["median"] / e["timing"]["event_ms"]["median"],
                    wall_ms=k["timing"]["wall_ms"]["median"] / e["timing"]["wall_ms"]["median"],
                )
            )
        within.append(dict(batch=b, arm=base, ratios=ratios))
        within_rows.append(
            f"| {b} | {base} | {st.mean(v['peak'] for v in ratios):.3f} | {st.mean(v['event_ms'] for v in ratios):.3f} | {st.mean(v['wall_ms'] for v in ratios):.3f} |"
        )
write_json(ROOT / "checkpoint_ratios.json", within)
passed = [k for k, v in a["candidates"].items() if v]
gate = (
    "RESOURCE PASS: " + ", ".join(passed)
    if passed
    else "REJECT tested depth/recomputation modulation recipes"
)
unstable = [
    dict(index=c["index"], arm=c["arm"], batch=c["batch"], seed=c["seed"], stability=c["stability"])
    for c in r["cases"]
    if max(c["stability"].values()) > 1.15
]
write_json(ROOT / "unstable_cases.json", unstable)
report = f"""# H146: depth, saved activations and fair recomputation controls

**{gate}.** This study tests actual VRAM and update cost for eight residual
FFNs, with the same recomputation option offered to every architecture. It does
not measure held-out quality or establish a new activation. H145's isolated eager
failure remains unchanged. The broader research goal remains open.

## What changed and why

Each step is x+FFN(x)/sqrt(8), repeated eight times with independently initialized
blocks. Full GELU uses hidden384, full SwiGLU256, budget GELU288, shared gate192,
and split modulation uses base192/gate96. The latter two use about24.8% fewer
parameters and25% fewer matrix MACs than both full controls. The exact module
implementation is H145's. Shared/split parameter totals are1,777,152/1,777,920;
full GELU/SwiGLU totals2,365,440/2,366,464. Budget GELU has1,774,848.

Eight layers test accumulation of saved intermediates. Each architecture runs
both ordinary autograd and non-reentrant checkpointing of each entire residual
step. Thus a checkpointed candidate must beat checkpointed controls. Comparing
only against unoptimized controls would not establish an architecture advantage.
This is an unnormalized residual MLP resource probe, not a Transformer benchmark.

[PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html) is a
known compute-for-memory tradeoff. RNG preservation is disabled here because
these modules contain no stochastic operations. CPU FP64 outputs and all input/
parameter gradients agree at rtol1e-10/atol1e-12 for all five architectures;
the gate weights are perturbed away from zero for these correctness checks.

## Saved-storage evidence

Batch2048, width384, depth8, FP32; CPU hooks deduplicate storage addresses to
avoid counting parameter transposes and activation views multiple times.
Parameter storage is classified separately and available in the raw traces.
The original input is separate from the activation column. This measures tensors
retained for backward after forward, not GPU peak or all live forward outputs.
The loss/output buffer and optimizer temporaries are outside this hook trace.
No hooks run during GPU resource measurements.

| Architecture:execution | Saved activation MiB | Original input MiB |
|---|---:|---:|
{chr(10).join(trace_rows)}

Eager split modulation retains81MiB versus full GELU69MiB; shared retains69MiB.
Both readouts in the shared candidate can alias one feature bank, whereas split
views retain the larger concatenated bank. Checkpointing leaves the same21MiB
of intermediate block inputs for every architecture, plus the original3MiB input.
This equal floor explains why eliminating internal saved tensors alone cannot
establish an architecture-specific memory advantage. GPU peaks additionally
include parameters, gradients, optimizer states and recomputation temporaries.

## GPU measurements

Means across three seeds. Timing entries average each run's median of eight
complete updates after four warmups. Gradient clearing is outside timing;
forward, backward, clipping and AdamW are inside. Input gradients are enabled.
The no-grad final diagnostic score is included in peak but outside timing.

| Batch | Architecture:execution | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|
{chr(10).join(rows)}

## Candidate gates against equally optimized full controls

Ranges across seeds, candidate/control. Required: parameters<=.80, peak<=.90,
CUDA and wall<=1.15, candidate and control half-window stability<=1.15 in every
fixture. Each comparison uses the same execution mode. Failures list any gate
failed by at least one seed; individual comparisons are in audit.json.

| Batch | Candidate | Same-mode control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---|---|---:|---:|---:|---|
{chr(10).join(comp_rows)}

There are{len(unstable)} unstable timing cases; their exact arm/seed/size and
half-window ratios are retained in unstable_cases.json. Eight timed updates
are a screen, not sustained-throughput proof. No favorable repeats were selected.
Memory failures remain valid independently of noisy timing comparisons.

## Checkpoint/eager comparison within each architecture

Means of three paired seed ratios. These are execution effects, not evidence of
novel architecture or preserved quality. Lower peak with slower updates is a
tradeoff and must satisfy the same prospective gate before promotion.

| Batch | Architecture | Peak ratio | CUDA ratio | Wall ratio |
|---:|---|---:|---:|---:|
{chr(10).join(within_rows)}

## Audit, scope and next decision

Sixty runs,720 optimizer updates/backwards,61 zero allocated/reserved CUDA
boundaries,60 final diagnostic scores plus60 independent CPU audit scores.
All six Gaussian/orthogonal-target datasets regenerate bitwise. Saved states and
Adam moments are finite, step counters equal12, parameter/optimizer counts and
all recorded timing summaries verify. First/final input and post-clipping parameter
gradients are finite; this does not prove freedom from deep gradient pathologies.

Independent CPU FP64 scoring uses explicit matrices, erf/sigmoid nonlinearities,
and batch257 rather than the trained model's forward function. Maximum relative
score error: {max(c["relative_error"] for c in a["checks"]):.3e}, below1e-5.
The audit performs no backward and initializes no CUDA. It verifies saved-state
scoring, not an independent replay of Adam. Mean, median and sample variance
across seeds, raw timings, losses and gradient diagnostics are machine-readable.
Losses are fixed-batch diagnostics: there is no validation/generalization claim.

Seeds83/97/109, batches128/2048, d384, eight residual blocks,12 AdamW updates,
LR.003, betas(.9,.95), zero decay, clip1, FP32/TF32off, four CPU threads,
ordinary eager kernels and8.125MiB default cuBLAS workspace. Arm order rotates
by fixture. RTX4070 Laptop GPU, driver610.62; PyTorch{a["torch_version"]},
CUDA build{a["cuda_build"]}. Hardware was queried during the run. No clock/power
changes or telemetry collection; clock/thermal causality is unknown.

Only resource-passing candidates may earn a separately frozen quality study.
Failures close these depth/execution settings; a kernel, precision, recomputation
schedule or architecture change requires a distinct hypothesis and protocol.
No maintained implementation/default changed; H145 receipt and source hashes
verify. This work supplies measured elimination evidence, not a breakthrough.

[Plan](modulation_depth_plan.md), [raw results](../results/modulation_depth_v1/result.json),
[audit](../results/modulation_depth_v1/audit.json),
[receipt](../results/modulation_depth_v1/receipt.json).
"""
reportpath = Path("research/modulation_depth_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: depth and checkpoint modulation screen\n\n[H146](modulation_depth_results.md): **{gate}**. Eight-layer stacks, both\neager and equally checkpointed controls;60 cases,720 updates, independent saved-state\nscoring verified. Checkpointing reduces saved internal activations to the same\n21MiB across architectures at batch2048. No quality claim or default changes;\nthe broader research goal remains open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H146 depth/recomputation screen](research/modulation_depth_results.md):\n{gate}. Audited resource evidence; research goal remains open."
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    reportpath,
    Path("research/modulation_depth_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H146",
    status="EVIDENCE_VERIFIED",
    gate=gate,
    training_updates=720,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
