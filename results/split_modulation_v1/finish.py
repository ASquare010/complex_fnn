"""Publish the audited negative resource result and preserve prior evidence."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/split_modulation_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "audit_exit.txt").read_text().strip() == "0"
assert a["status"] == "VERIFIED" and not any(a["candidates"].values())
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/modulated_capacity_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/modulated_capacity_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
rows = []
for g in a["aggregates"]:
    s = g["statistics"]
    n = p["counts"][g["arm"]]["parameters"]
    rows.append(
        f"| {g['batch']} | {g['arm']} | {n:,} | {s['peak_mib']['mean']:.3f} | {s['event_ms']['mean']:.3f} | {s['wall_ms']['mean']:.3f} |"
    )
ratios = []
for c in a["comparisons"]:
    q = c["ratios"]
    failures = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    ratios.append(
        f"| {c['batch']} | {c['seed']} | {c['arm']} | {c['control']} | {q['peak']:.4f} | {q['event_ms']:.3f} | {q['wall_ms']:.3f} | {failures} |"
    )
max_error = max(v["relative_error"] for v in a["checks"])
unstable = [(c["index"], c["arm"]) for c in r["cases"] if max(c["stability"].values()) > 1.15]
report = f"""# H145: separate gate observations fail the eager VRAM screen

**Reject this resource recipe.** Both output-modulation candidates have about
24.8% fewer parameters and25% fewer matrix MACs than the full controls. Neither
achieves the required10% allocated-GPU-memory saving at either batch size.
At batch2048, both use more GPU memory than full GELU. No learning-quality
experiment is promoted from this screen.

## Hypothesis and equations

For dimension384, shared modulation uses z=GELU(Ux+a) with192 features and
F(x)=Bz+c+x*(Cz+e). Split modulation forms288 features in one projection,
feeds the first192 to B and the remaining96 to C. The split candidate and a
288-wide GELU with a learned diagonal input path have exactly222,240 parameters;
the shared candidate has222,144. All four budget arms use221,184 matrix MACs,
versus294,912 for full GELU384 and SwiGLU256. These MAC counts exclude elementwise
operations; lower matrix arithmetic does not guarantee lower runtime or memory.

The split design gives base and gate different observations, but their combined
linear observation still has rank at most288. [H144's bound](modulated_capacity_theory.md)
still applies to that combined observation; removing one skew-target obstruction
is not a universal approximation or learning guarantee. Gate readouts start at
zero; the active-gate FP64 input/all-parameter finite-difference tests pass.
Shared/split initial base weights are identical and outputs agree numerically.
The prospective plan's word "exactly" describes the algebraic function; the
preflight uses floating-point tolerances for outputs from different GEMM shapes.

## Measurements

Means across three seeds. Each timing is a run's median of eight complete
forward/backward/clipping/AdamW updates after four warmups. Inputs require
upstream gradients. Gradient clearing is outside timers. Peak includes model,
resident batch, gradients, optimizer, temporaries and the final diagnostic score.
One model is resident at a time. Reserved allocation is separately retained.

| Batch | Arm | Parameters | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

At batch128, both dynamic candidates save only about6.2% versus full GELU.
At batch2048, shared uses about0.8% more and split about2.7% more than full GELU.
The budget GELU control is also below the required10% saving. Parameter savings
are insufficient here because actual training allocation includes much more than
weights. Modulation adds full-width outputs, products and backward dependencies;
this screen measures their aggregate cost, not an allocator-attributed causal
breakdown. No activation-level attribution should be inferred without a trace.

## Every prospective candidate comparison

Ratios are candidate/control; smaller is better. Required: parameter ratio<=.80,
peak<=.90, CUDA/wall<=1.15, candidate half-window stability<=1.15. Controls' stability
is also retained, and all raw timings remain available. No thresholds were tuned.

| Batch | Seed | Candidate | Control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---|---:|---:|---:|---|
{chr(10).join(ratios)}

The short timing windows are a resource screen, not sustained-throughput proof.
Cases exceeding the15% half-window stability threshold: {unstable}.
Timing noise does not rescue either candidate's independently failed memory gate.
Seed-level means, medians and sample variances for peak, CUDA/wall timing and
final diagnostic loss are in [audit.json](../results/split_modulation_v1/audit.json).

## Reproducibility and audit

Frozen recipe: Gaussian input, independent orthogonal linear target, one fixed
batch per seed/size; seeds53/67/79; twelve updates; AdamW LR.003, betas(.9,.95),
zero decay, clip1; FP32, TF32off, four CPU threads, ordinary eager execution,
default8.125MiB cuBLAS workspace. Six-arm execution order rotates by fixture.
NVIDIA GeForce RTX4070 Laptop GPU, driver610.62 (queried after training);
PyTorch {a["torch_version"]}, CUDA build {a["cuda_build"]}. No clock/power changes.
Telemetry was not collected, so temperature or clock causality is unassessed.

All36 saved model/optimizer states are finite, optimizer counters equal12,
exact parameter/optimizer byte counts verify, and six datasets regenerate bitwise.
Independent CPU FP64 scoring uses explicit matrices, erf/sigmoid formulas and
batch257, without the model's forward method. All36 final scores agree within
1e-5 relative tolerance; maximum observed relative error is {max_error:.3e}.
This verifies saved-state scoring, not an independent replay of Adam updates.
First/final input-gradient and post-clipping parameter-gradient norms are finite;
this twelve-step check does not establish long-depth gradient stability.

There were432 optimizer updates/backwards,37 clean GPU allocator boundaries,
36 training-process final scores and36 independent CPU audit scores. The audit
initialized no CUDA and performed no backward calls. No hidden repeats or tuning.
All frozen source and prior receipt hashes verify; maintained code/defaults are
unchanged. Initial/final losses are fixed-batch diagnostics only: no validation,
generalization, expressivity gain or quality retention is claimed.

## Decision

Close these eager shared/split-gate settings before costly fitting. A future
resource hypothesis must change the execution or saved-tensor footprint under
a separately frozen protocol; it cannot silently reuse these timings as evidence
for a fused or recomputed implementation. Existing H141/H142 memory-helper gains
remain scoped evidence; they do not make this architecture successful.

Modulation and feature splitting have prior art, including
[FiLM](https://arxiv.org/abs/1709.07871). This screen claims no novel activation
or breakthrough. The broader VRAM-and-quality research goal remains open.

[Prospective plan](split_modulation_plan.md),
[prototype](../results/split_modulation_v1/model.py),
[raw measurements](../results/split_modulation_v1/result.json),
[evidence receipt](../results/split_modulation_v1/receipt.json).
"""
report_path = Path("research/split_modulation_results.md")
report_path.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = """## Latest: shared/split modulation resource screen rejected

[H145](split_modulation_results.md): both candidates fail the10% VRAM-saving gate
at batches128 and2048 despite about24.8% fewer parameters. At batch2048, shared
and split use0.8% and2.7% more memory than full GELU. All36 saved states/scores
are audited;432 updates,37 clean GPU boundaries. No learning-quality promotion,
model/default change or novelty claim. The broader research goal remains open."""
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = """Latest: [H145 resource screen](research/split_modulation_results.md)
rejects shared/split output modulation: fewer parameters did not yield enough
VRAM savings. Audited results; no model/default changes. Research goal remains open."""
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    report_path,
    Path("research/split_modulation_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H145",
    status="EVIDENCE_VERIFIED",
    gate="REJECT eager shared/split modulation resource recipe",
    training_updates=432,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
