"""Publish the fully audited learning screen and retained negative/inconclusive results."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/readout_learning_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "audit_exit.txt").read_text().strip() == "0" and a["status"] == "VERIFIED"
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/fixed_tail_capacity_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/fixed_tail_capacity_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
gate = (
    "PASS small learning pilot; larger validation required"
    if a["promoted"]
    else "NO PROMOTION from the readout learning pilot"
)
rows = []
for c in a["selected"]:
    s = c["reporting"]
    rows.append(
        f"| {c['task']} | {c['arm']} | {c['parameters']:,} | {c['rate']:.3f} | {s['mean']:.5f} | {s['median']:.5f} | {s['sample_variance']:.3e} |"
    )
comparisons = []
for c in a["comparisons"]:
    v = c["ratios"]
    failed = ", ".join(k for k, passed in c["gates"].items() if not passed) or "none"
    comparisons.append(
        f"| {c['task']} | {c['seed']} | {c['qualified']} | {v['gelu']:.3f} | {v['swiglu']:.3f} | {v['budget_gelu']:.3f} | {v['fixed']:.3f} | {v['affine']:.3f} | {failed} |"
    )
qualifications = ", ".join(
    f"{task}: {'qualified' if passed else 'inconclusive (positive-control gate failed)'}"
    for task, passed in a["qualifications"].items()
)
counts = "\n".join(
    f"| {arm} | {v['parameters']:,} | {v['buffer_bytes']:,} |" for arm, v in p["counts"].items()
)
# A scientific plot of selected reporting results; all baselines remain visible.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
for ax, task in zip(axes, p["tasks"], strict=True):
    cs = [
        next(v for v in a["selected"] if v["task"] == task and v["arm"] == arm) for arm in p["arms"]
    ]
    colors = [
        "#c24a32" if arm == "learned" else "#8064a2" if arm in ("fixed", "affine") else "#527c99"
        for arm in p["arms"]
    ]
    ax.barh(range(len(cs)), [v["reporting"]["mean"] for v in cs], color=colors)
    ax.set_xscale("log")
    ax.set_title(task)
    ax.set_xlabel("Reporting MSE (log scale)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(p["arms"])), p["arms"])
axes[0].invert_yaxis()
fig.suptitle(
    "H152: readout extension versus standard activations\nMeans of three seeds; learning rates selected using validation only"
)
fig.tight_layout()
fig.savefig(ROOT / "selected_results.png", dpi=150)
plt.close(fig)
checks = read(ROOT / "checks.json")
report = f"""# H152: learned readout fixes linear capacity, but must earn nonlinear performance

**{gate}.** All198 fits completed their fixed budget. Rate selection uses
validation only; reporting results below are held out from that selection.
Task status: {qualifications}. Failure is scoped to this initialization,
optimizer/search and300-update budget. An inconclusive task does not establish
that every architecture failed in principle. The broad research goal remains open.

## What was tested

Add a learned32x32 readout and bias after four reversible scalar/orthogonal layers.
At zero shape/bias the core is A*x; W=M*A^-1 exactly represents any linear target.
The CPU FP64 witness passes with max absolute error
{checks["linear_witness_max_error"]:.3e}. Complete GPU core/readout/input gradients
agree with CPU FP64 native autograd within1e-4 per family (observed maximum
{max(checks["gradient_relative_errors"]):.3e}). This evades H151's fixed-tail bound,
but linear representability is not sufficient evidence for complex pattern learning.

The learned candidate uses the existing fused reconstruction backend. Fixed shape
and affine controls use the native reconstructed backend; inference equations and
matched initial core/readout tensors define their learning ablations. Core biases
start at zero, nonzero theta/Q are seeded identically; readout starts at A^T, which
approximately cancels rotation, not nonlinear shape. The fixed-shape control still
learns core biases and the final readout; only its scalar shape is frozen.

Standard controls are four residual FFN blocks x+FFN(x)/2, with hidden32 or
SwiGLU hidden21. Budget GELU hidden4 and a plain linear map test low-parameter
alternatives. ReLU, LeakyReLU(.01), layer-shared PReLU, GELU, SiLU and SwiGLU are
all included. Ungated standard models share seeded weight initialization; gate
shape changes make SwiGLU's initialization structurally different.

| Arm | Trainable parameters | Fixed buffer bytes |
|---|---:|---:|
{counts}

This small d32 pilot measures learnability. Its allocated peaks and fit times are
retained as diagnostics, not evidence of production-scale VRAM savings. Inputs
are fixed training data and do not require upstream gradients during fitting;
input-gradient correctness was checked separately in preflight. The earlier
full-scale resource failures are not reclassified by this study.

## Data and matched training

Three fixed target functions: orthogonal linear map; hidden128 GELU teacher;
cyclic pairwise products followed by orthogonal mixing. Functions remain fixed
across seeds263/277/293 while input data and model initialization vary.
Each task/seed has8192 Gaussian samples:4096 training,2048 validation,2048 reporting.
Target mean and global RMS use training data only. Every arm and rate sees the
same300 batches of128 training indices. Batch order is saved and reproducible.

All11 arms receive rates.001 and.003,300 AdamW updates, betas(.9,.95), zero decay,
clip1, FP32/TF32off and four CPU threads. No augmentation, early stopping,
rate-specific extra steps, tuning based on reporting data, or failed-fit replacement.
The chosen rate minimizes mean validation MSE across the three seeds for each
arm/task; ties choose the lower rate. Reporting seeds are model/data repetitions
within the same target function, not three independent target functions.

## Selected reporting results

![Selected reporting results](../results/readout_learning_v1/selected_results.png)

Means, medians and sample variances across three reporting seeds. MSE is measured
in training-normalized target units; the zero-predictor reporting loss is recorded
per dataset. All unselected rate results remain in result.json.

| Task | Arm | Parameters | Selected LR | Mean MSE | Median MSE | Sample variance |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Every candidate gate

A task qualifies only if both full GELU and SwiGLU beat zero-predictor reporting
MSE by at least20% in every seed. On qualified tasks the learned candidate must
be within1% of both full controls and budget GELU in every seed. On nonlinear
tasks it must additionally improve fixed-shape and affine controls by at least5%.
Ratios below are learned/control; lower is better. Inconclusive tasks cannot pass
promotion regardless of these diagnostic ratios.

| Task | Seed | Qualified | GELU ratio | SwiGLU ratio | Budget ratio | Fixed ratio | Affine ratio | Failed comparison gates |
|---|---:|---|---:|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

## Evidence and decision

Independent CPU FP64 scoring implements the equations directly with batches257,
without model.forward or the custom backward. All396 validation/reporting scores
pass relative tolerance1e-5; maximum observed discrepancy is
{max(v for c in a["checks"] for v in c["relative_errors"].values()):.3e}.
All nine datasets regenerate bitwise. Fixed Q/frozen theta, parameter and buffer
counts, optimizer hyperparameters/counters300 and finite states/moments/recorded
gradient diagnostics verify. Histories record losses/pre-clipping global gradient
norm every25 steps; they do not establish depth-independent gradient health or
identify which parameter family limits learning. Final shapes are in saved states.

There were59,400 training backward/optimizer updates, one GPU and one CPU preflight
backward,199 clean training CUDA boundaries and a clean preflight boundary.
The independent scoring audit initialized no CUDA and ran no backward. One model
was resident at a time. Training wall seconds across all fits sum to
{sum(c["wall_seconds"] for c in r["cases"]):.1f}; scoring, initialization, data
transfer/setup outside each timer and audit add further elapsed time.

This is a balanced but limited two-rate pilot, not exhaustive hyperparameter
optimization, a significance test, universal approximation proof, or a SOTA claim.
Do not allocate long language-model training on linear-capacity algebra alone.
The reported gates decide promotion for this recipe; deficits/inconclusive tasks
must guide a distinct hypothesis or a stronger positive-control benchmark, not
be concealed by averaging tasks or selecting a lucky seed. Prior resource and
capacity evidence, maintained source/defaults and receipt hashes remain intact.

[Plan](readout_learning_plan.md), [model](../results/readout_learning_v1/model.py),
[all runs](../results/readout_learning_v1/result.json),
[selection/audit](../results/readout_learning_v1/audit.json),
[receipt](../results/readout_learning_v1/receipt.json).
"""
reportpath = Path("research/readout_learning_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: readout learning pilot\n\n[H152](readout_learning_results.md): **{gate}**.\n198 fits/59,400 updates, eleven arms, three tasks/seeds and equal two-rate search.\n396 independent saved-state scores verified. Task status: {qualifications}.\nLinear capacity is repaired; general nonlinear performance must satisfy the\nreported gates. Prior failures remain intact; broad research goal stays open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H152 readout learning pilot](research/readout_learning_results.md):\n{gate}. Mandatory activations and fixed/affine controls included."
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
    Path("research/readout_learning_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H152",
    status="EVIDENCE_VERIFIED",
    gate=gate,
    training_updates=59400,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})

