"""Publish buffer-aware autograd reconstruction resource evidence."""

import ast
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/reversible_training_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "audit_exit.txt").read_text().strip() == "0" and a["status"] == "VERIFIED"
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/reversible_softsign_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/reversible_softsign_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
original = (
    Path("results/modulation_depth_v1/worker.py")
    .read_text()
    .replace("results.modulation_depth_v1.model", "results.reversible_training_v1.model")
    .replace("results/modulation_depth_v1", "results/reversible_training_v1")
)
assert ast.dump(ast.parse(original)) == ast.dump(ast.parse((ROOT / "worker.py").read_text()))
passed = a["candidates"]["learned:reconstruct"]
gate = (
    "PASS scoped reconstruction resource gate"
    if passed
    else "REJECT tested reconstruction resource recipe"
)
rows = []
for v in a["aggregates"]:
    s = v["statistics"]
    rows.append(
        f"| {v['batch']} | {v['arm']} | {s['peak_mib']['mean']:.3f} | {s['event_ms']['mean']:.3f} | {s['wall_ms']['mean']:.3f} |"
    )
comparisons = []
for c in a["comparisons"]:
    q = c["ratios"]
    fail = ", ".join(k for k, v in c["gates"].items() if not v) or "none"
    comparisons.append(
        f"| {c['batch']} | {c['seed']} | {c['control']} | {q['peak']:.4f} | {q['event_ms']:.3f} | {q['wall_ms']:.3f} | {fail} |"
    )
traces = []
for t in p["traces"]:
    v = t["unique_bytes"]
    traces.append(
        f"| {t['arm']} | {v['activation'] / 2**20:.3f} | {v['input'] / 2**20:.3f} | {v['buffer'] / 2**20:.3f} |"
    )
pairs = read(ROOT / "execution_pairs.json")
max_parameter = max(v["parameter_relative_error"] for v in pairs)
max_loss = max(v["loss_relative_error"] for v in pairs)
unstable = [
    dict(index=c["index"], arm=c["arm"], batch=c["batch"], seed=c["seed"], stability=c["stability"])
    for c in r["cases"]
    if max(c["stability"].values()) > 1.15
]
write_json(ROOT / "unstable_cases.json", unstable)
report = f"""# H148: activation reconstruction inside real training

**{gate}.** The scalar inverse now runs inside a custom autograd backward.
The experiment measures actual complete optimizer updates, including resident
fixed mixing matrices. Numerical correctness and resource eligibility are
separate: all saved states pass independent scoring, but the prospective resource
decision below controls whether this implementation earns a quality study.
No learning-quality or breakthrough claim follows from this fixed-batch screen.

## Mechanism and fair controls

Eight layers compute z=Qx+b and phi(z)=z+a*z/(1+abs(z)), a=.25*tanh(theta).
The reconstruction boundary saves final output, theta, bias and Q. It reconstructs
one layer at a time and returns analytical input/parameter gradients; it does
not retain interior forward activations. Q is immutable dense orthogonal mixing,
created by CPU FP64 QR and cast to FP32. [H147](reversible_softsign_results.md)
derives the inverse, input-Jacobian bounds and contraction-capacity limitation.

At d384/depth8 there are6,144 trainable scalars and1,179,648 fixed matrix entries.
The matrices consume4.5MiB FP32 and are counted in all GPU peaks. Fewer trainable
parameters do not mean the matrices cost no memory, construction or arithmetic.
The learned model has half the forward matrix MACs of the full residual FFN
stacks; analytical inverse and elementwise operations are additional work.

Controls include full GELU/SwiGLU in eager and checkpoint modes, budget GELU
checkpointing, the identical reversible model in eager/checkpoint modes,
frozen-shape reconstruction (3,072 trainable biases), and affine reconstruction.
Affine uses (1+a)*z with the same6,144 parameters and same Q/b initialization.
Fixed-shape and learned nonlinear controls start with identical functions.
Conventional controls are eight residual steps x+FFN(x)/sqrt(8); the reversible
stack is a different architecture whose representational adequacy remains untested.

## Retained-storage trace

CPU hooks at batch2048, deduplicated by storage. Parameter storage is classified
separately in protocol.json. The activation column includes the final output
saved by reconstruction. Original input is separate. Hook-saved buffers do not
exhaust resident buffers: checkpoint closures still hold the model's4.5MiB Q,
although those tensors are not returned by these hooks. All resident buffers are
included in measured GPU peak. Traces exclude loss and optimizer, and hooks are
absent during resource measurement.

| Arm | Saved activation MiB | Saved original input MiB | Hook-saved buffer MiB |
|---|---:|---:|---:|
{chr(10).join(traces)}

## Complete GPU training measurements

Means over seeds173/191/211. Each timing is the median of eight updates after
four warmups. Input gradients are enabled; gradient reset is outside timing,
forward/backward/clipping/AdamW inside. A final no-grad diagnostic score is in
job peak but outside timing. QR initialization/setup is outside timing.

| Batch | Arm | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|
{chr(10).join(rows)}

The reconstruction trace retains less than checkpointing, yet measured peak can
be higher because backward temporaries also contribute. These peaks establish
the total difference; exact per-operation causality needs an allocation trace.
The cheaper affine control prevents treating reconstruction efficiency as evidence
that learned nonlinear shape itself is useful. No quality comparison is made.

## Every primary comparison

Learned reconstruction divided by each checkpointed full control. Required in
every fixture: parameter ratio<=.8, peak<=.9, CUDA/wall<=1.15, and both candidate
and control half-window stability<=1.15. The parameter ratio is about.0026;
the much larger fixed matrices remain included in the peak column.

| Batch | Seed | Checkpointed control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---:|---:|---:|---|
{chr(10).join(comparisons)}

{len(unstable)} of60 cases exceed the15% half-window timing-stability threshold;
the exact cases are in unstable_cases.json. Short timing windows are a screen,
not sustained-throughput evidence. No clock/power changes, favorable repeats or
threshold changes were made. Finite runs were retained even when gates failed.

## Verification and limitations

CPU FP64 finite-difference checks cover inputs/theta/bias for nonlinear and affine
custom backward. Learned eager/checkpoint/reconstruction gradients agree within
rtol1e-10/atol1e-12 on exactly FP64-orthogonal small probes. The custom boundary
is first-order only: no higher derivatives, autocast, compiler or stochastic-module
support is claimed. An initial closure-binding lint issue was fixed before any
preflight or frozen experiment ran; no numerical retry was needed.

Sixty saved states independently score in CPU FP64 using explicit matrix and
erf/sigmoid/scalar formulas with batch257. Maximum relative score discrepancy is
{max(c["relative_error"] for c in a["checks"]):.3e}, below1e-5. Six datasets
regenerate bitwise, fixed Q and frozen theta match independent regeneration,
optimizer counters equal12, counts/byte totals verify, and model states, moments
and gradient diagnostics are finite. Per-seed means, medians, sample variances,
raw histories, memory and gradient diagnostics are retained. This is saved-state
verification, not independent replay of every Adam update.

For learned checkpoint/reconstruction versus eager from matched initialization,
maximum final parameter relative difference is{max_parameter:.3e} and maximum
final loss relative difference is{max_loss:.3e}. These are descriptive twelve-step
checks, not bitwise or long-trajectory equivalence guarantees.

There were720 optimizer updates/backwards,61 clean allocated/reserved CUDA
boundaries,60 final fixed-batch scores and60 CPU audit scores. Audit used no
GPU or backward. The worker's AST matches H146's exactly after replacing only
model import and output root. Recipe: FP32/TF32off, four CPU threads, ordinary
kernels/default8.125MiB cuBLAS workspace, AdamW LR.003, betas(.9,.95), decay0,
clip1, twelve steps. The initial plan mentions a diagnostic flag exception, but
no diagnostic logic changed. RTX4070 Laptop GPU; PyTorch{a["torch_version"]},
CUDA build{a["cuda_build"]}. No hardware telemetry was collected.

The input is one Gaussian batch and its independent orthogonal target. Final
losses are diagnostics, not validation or generalization. No maintained source
or model default changed; prior receipts and frozen hashes verify.

The resource gate alone decides whether the current implementation earns fitting.
A failed recipe requires a new execution hypothesis before promotion; a pass
would still need strong nonlinear targets and conventional activation baselines,
multiple seeds, reduced-width comparisons and held-out quality. The broader
VRAM-and-quality research goal remains open. [RevNets](https://arxiv.org/abs/1707.04585)
established activation reconstruction; this experiment claims no novelty.

[Plan](reversible_training_plan.md), [prototype](../results/reversible_training_v1/model.py),
[results](../results/reversible_training_v1/result.json),
[audit](../results/reversible_training_v1/audit.json),
[receipt](../results/reversible_training_v1/receipt.json).
"""
reportpath = Path("research/reversible_training_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: real autograd reconstruction resource screen\n\n[H148](reversible_training_results.md): **{gate}**. Custom backward,\n60 audited runs/720 updates with fixed matrices counted. Both checkpointed\nfull controls, learned eager/checkpoint and fixed-shape/affine controls included.\nNumerical verification is not quality evidence. Broad research goal remains open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H148 reconstruction training screen](research/reversible_training_results.md):\n{gate}. Fixed buffers counted; learning quality remains unproven."
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
    Path("research/reversible_training_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H148",
    status="EVIDENCE_VERIFIED",
    gate=gate,
    training_updates=720,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
