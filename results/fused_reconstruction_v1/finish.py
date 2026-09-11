"""Publish the audited fused-inverse execution result."""

import ast
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/fused_reconstruction_v1")
p, r, a, k = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json", "checks.json")]
assert not (ROOT / "receipt.json").exists()
assert a["status"] == "VERIFIED" and k["passed"]
assert all(
    (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
    for stage in ("prepare", "check", "worker", "audit")
)
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/reversible_training_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/reversible_training_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
expected = (
    Path("results/reversible_training_v1/worker.py")
    .read_text()
    .replace("results.reversible_training_v1.model", "results.fused_reconstruction_v1.model")
    .replace("results/reversible_training_v1", "results/fused_reconstruction_v1")
)
expected = expected.replace(
    'p = read(ROOT / "protocol.json")',
    'p = read(ROOT / "protocol.json")\n    assert read(ROOT / "checks.json")["passed"]\n    assert (ROOT / "check_exit.txt").read_text().strip() == "0"',
)
assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((ROOT / "worker.py").read_text()))
passed = a["candidates"]["learned:fused"]
gate = (
    "PASS scoped fused-reconstruction resource gate"
    if passed
    else "REJECT tested fused-reconstruction resource recipe"
)
rows = []
for v in a["aggregates"]:
    s = v["statistics"]
    rows.append(
        f"| {v['batch']} | {v['arm']} | {s['peak_mib']['mean']:.3f} | {s['event_ms']['mean']:.3f} | {s['wall_ms']['mean']:.3f} |"
    )
comps = []
for c in a["comparisons"]:
    v = c["ratios"]
    failed = ", ".join(key for key, value in c["gates"].items() if not value) or "none"
    comps.append(
        f"| {c['batch']} | {c['seed']} | {c['control']} | {v['peak']:.4f} | {v['event_ms']:.3f} | {v['wall_ms']:.3f} | {failed} |"
    )
relative = []
for c in r["cases"]:
    if c["arm"] != "learned:fused":
        continue
    ref = next(
        v
        for v in r["cases"]
        if (v["batch"], v["seed"], v["arm"]) == (c["batch"], c["seed"], "learned:reconstruct")
    )
    relative.append(
        dict(
            batch=c["batch"],
            seed=c["seed"],
            peak_ratio=c["peak_bytes"] / ref["peak_bytes"],
            event_ratio=c["timing"]["event_ms"]["median"] / ref["timing"]["event_ms"]["median"],
            wall_ratio=c["timing"]["wall_ms"]["median"] / ref["timing"]["wall_ms"]["median"],
        )
    )
write_json(ROOT / "native_ratios.json", relative)
pairs = read(ROOT / "execution_pairs.json")
unstable = [
    dict(index=c["index"], arm=c["arm"], batch=c["batch"], seed=c["seed"], stability=c["stability"])
    for c in r["cases"]
    if max(c["stability"].values()) > 1.15
]
write_json(ROOT / "unstable_cases.json", unstable)
fused_pairs = [v for v in pairs if v["mode"] == "fused"]
local_error = max(v["relative"] for c in k["checks"] if "shape" in c for v in c["errors"])
stack_error = max(
    v for c in k["checks"] if "full_batch" in c for v in c["relative_gradient_errors"]
)
report = f"""# H149: fused inverse and local gradients

**{gate}.** The new execution backend fuses reconstruction and local derivative
work. It changes no forward equation or parameterization. The decision is based
on actual training allocation and complete update timing against checkpointed
GELU and SwiGLU. Any resource pass earns a separate quality study, not a claim
that the architecture learns equally well. The broad research goal remains open.

## Implementation and qualification

H148's analytical inverse created many full-sized temporary tensors. A fixed
64-row by32-channel Triton tile computes reconstructed z, z-b, input-side dz,
and partial theta/bias gradient sums together. Native PyTorch sums combine the
partials; native matrix products propagate the state and cotangent. The earliest
layer skips z-b because only its input gradient is needed. Four warps, no autotune,
no floating-operation fusion. Forward and fixed dense orthogonal buffers remain
exactly H148's. There are6,144 trainable scalars and4.5MiB of fixed matrices.

The kernel uses sqrt(B*B+4*abs(y)) with stable root branches, rather than hypot.
Qualification covers signed magnitudes1e-12..1e12 plus zero, not all FP32 values.
Shapes(1,1),(7,33),(128,384),(2048,384) check channel/row tails against independent
CPU FP64 formulas for reconstructed preactivation and all local gradients.
Maximum local relative error is{local_error:.3e}; all relative errors<=5e-5 and
scaled elementwise errors<=1e-4. Complete-stack checks at batches128/2048 compare
against H148's PyTorch reconstructed backward: maximum per-family relative error
{stack_error:.3e}<=1e-4 and identical forward outputs. These complement H148's
finite-difference checks; they are not an independent finite-difference test of
the compiled kernel. All checks pass before the resource worker is permitted.

Compilation and correctness work took{k["setup_and_checks_seconds"]:.2f}s in the
check process. Generated kernels are cached in this study's triton_cache directory;
cache artifacts are excluded from evidence hashes. Source/compiler hashes and
Triton{k["triton_version"]} are recorded. Cached loading and first-use overhead
in the worker may enter its four discarded warmups; setup is not claimed free.
Only contiguous CUDA FP32 first-order use is qualified; no autocast, higher-order,
compiler integration, arbitrary-magnitude or other-hardware guarantee is made.

## Complete GPU training measurements

Eight layers, width384, rho.25. Seeds223/239/251, batches128/2048. Means across
seeds of allocated peak and each run's median eight updates after four warmups.
Inputs require gradients. Gradient reset is outside timing; forward, backward,
clipping and AdamW are inside. Final no-grad scores enter peak but not timing.
QR setup is outside timing. All matrix buffers, gradients and optimizer states
are counted in the CUDA allocator peak; this does not measure total driver VRAM.

| Batch | Arm | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|
{chr(10).join(rows)}

The fixed-shape and affine controls retain their native H148 implementation;
this experiment does not establish fused nonlinearity beats a fused affine control.
Their inclusion tests how much runtime/parameter cost is attributable to shape
rather than treating invertibility itself as proof of useful nonlinear capacity.
Same-model native/fused allocation and timing ratios are in native_ratios.json.

## Every primary gate

Fused/control ratios. Required in every fixture: parameters<=.8, peak<=.9,
CUDA/wall<=1.15, candidate and control half-window timing stability<=1.15.
The parameter ratio is about.0026; fixed matrices remain included in memory.
Both full controls receive checkpointing, and their eager results are retained.

| Batch | Seed | Checkpointed control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---:|---:|---:|---|
{chr(10).join(comps)}

There are{len(unstable)} timing-unstable cases across all60 arms/fixtures; exact
identities and half-window ratios are retained in unstable_cases.json. No clocks,
power limits, selected reruns or thresholds changed. This brief screen is not
sustained-throughput or long-run convergence evidence. H148's failure is preserved.

## Audit and decision scope

All60 saved states pass independent CPU FP64 explicit-formula scoring with
batch257; max relative discrepancy{max(c["relative_error"] for c in a["checks"]):.3e},
below1e-5. Six datasets and all fixed matrices regenerate bitwise. Optimizer
counters equal12, exact parameter/buffer/optimizer bytes and timing summaries
verify, all states/moments/diagnostic gradients are finite. The audit uses the
forward equation without either custom backward. It is saved-state validation,
not replay of every optimizer step. Means/medians/sample variances are retained.

Fused versus native reconstructed execution starts identically. After12 updates,
maximum parameter relative difference is{max(v["parameter_relative_error"] for v in fused_pairs):.3e};
maximum final-loss relative difference is{max(v["loss_relative_error"] for v in fused_pairs):.3e}.
These descriptive checks do not guarantee identical later training trajectories.

There were720 optimizer/backward calls and four extra correctness VJPs, plus
four local-kernel qualification cases. Training has61 clean allocated/reserved
CUDA boundaries; the check process records seven more. Audit performs no GPU,
backward or optimizer work. The worker's AST is H148's with only model/root
replacement and mandatory successful-check guards. Four CPU threads, FP32/TF32off,
default8.125MiB cuBLAS workspace, AdamW LR.003 betas(.9,.95), decay0,clip1.
RTX4070 Laptop GPU; PyTorch{a["torch_version"]}, CUDA build{a["cuda_build"]}.
No telemetry was collected; thermal/clock effects are unassessed.

This uses one Gaussian batch and independent orthogonal target per fixture.
Losses are diagnostic only, with no validation, generalization or nonlinear
capacity result. The strict gate controls further allocation. A pass still needs
strong nonlinear tasks, fixed/affine controls, ordinary activation baselines,
multiple seeds and held-out quality at materially reduced resource cost. A fail
requires a distinct implementation or hypothesis before promotion. Maintained
modules/defaults are unchanged and previous receipt/source hashes verify.

Fusion and activation reconstruction are known techniques, not novelty claims:
[Triton fusion tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html),
[RevNets](https://arxiv.org/abs/1707.04585).

[Plan](fused_reconstruction_plan.md), [kernel](../results/fused_reconstruction_v1/kernel.py),
[raw results](../results/fused_reconstruction_v1/result.json),
[audit](../results/fused_reconstruction_v1/audit.json),
[receipt](../results/fused_reconstruction_v1/receipt.json).
"""
reportpath = Path("research/fused_reconstruction_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: fused reconstruction training screen\n\n[H149](fused_reconstruction_results.md): **{gate}**.\nFused inverse/local gradients qualified,60 audited resource runs/720 updates,\nfixed matrix buffers counted. Four extra numerical-check VJPs. No quality or\nnovelty claim; broad research goal remains open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H149 fused reconstruction](research/fused_reconstruction_results.md):\n{gate}. Resource evidence is separate from learning quality."
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and not {"unused_cache", "triton_cache"}.intersection(f.parts)
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    reportpath,
    Path("research/fused_reconstruction_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H149",
    status="EVIDENCE_VERIFIED",
    gate=gate,
    training_updates=720,
    numerical_check_backwards=4,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
