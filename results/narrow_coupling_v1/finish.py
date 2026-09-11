"""Publish the audited narrow-coupling screen without upgrading synthetic evidence."""

import platform
import statistics as st
from pathlib import Path

import matplotlib
import torch

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("results/narrow_coupling_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "audit_exit.txt").read_text().strip() == "0" and a["status"] == "EVIDENCE_VERIFIED"
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/even_tangent_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/even_tangent_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest

gate = (
    "PASS small learning screen; larger validation required"
    if a["passed"]
    else "NO PROMOTION from narrow coupling"
)
qualifications = "; ".join(
    f"{t}: {'qualified' if v['qualified'] else 'inconclusive (positive-control gate failed)'}"
    for t, v in a["tasks"].items()
)
rows = []
resource_rows = []
for c in a["selected"]:
    s = c["reporting"]
    rows.append(
        f"| {c['task']} | {c['arm']} | {p['counts'][c['arm']]['parameters']:,} | {c['rate']:.3f} | {s['mean']:.5f} | {s['median']:.5f} | {s['sample_variance']:.3e} |"
    )
    cs = [r["cases"][i] for i in c["case_indices"]]
    resource_rows.append(
        f"| {c['task']} | {c['arm']} | {st.mean(v['training_peak_bytes'] for v in cs) / 2**20:.3f} | {st.mean(v['inference_peak_bytes'] for v in cs) / 2**20:.3f} | {st.mean(v['wall_seconds'] for v in cs):.2f} | {st.mean(v['history'][-1]['gradient_norm'] for v in cs):.3f} |"
    )
comparisons = []
for c in a["comparisons"]:
    v = c["ratios"]
    comparisons.append(
        f"| {c['task']} | {c['seed']} | {v['gelu']:.3f} | {v['swiglu']:.3f} | {v['budget_gelu']:.3f} | {v['fixed_coupling']:.3f} | {v['affine_coupling']:.3f} | {c['passed']} |"
    )
counts = "\n".join(
    f"| {arm} | {v['parameters']:,} | {v['buffer_bytes']:,} |" for arm, v in p["counts"].items()
)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, task in zip(axes, p["tasks"], strict=True):
    cs = [
        next(v for v in a["selected"] if v["task"] == task and v["arm"] == arm) for arm in p["arms"]
    ]
    colors = [
        "#bd4935" if arm == "coupling" else "#8872a5" if "coupling" in arm else "#527c99"
        for arm in p["arms"]
    ]
    ax.barh(range(len(cs)), [v["reporting"]["mean"] for v in cs], color=colors)
    for j, c in enumerate(cs):
        ax.scatter(
            [r["cases"][i]["reporting_mse"] for i in c["case_indices"]],
            [j] * 3,
            color="black",
            s=12,
            zorder=3,
        )
    ax.set_title(task)
    ax.set_xlabel("Held-out MSE (bars: mean; dots: seeds)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(p["arms"])), p["arms"])
axes[0].invert_yaxis()
fig.suptitle(
    "H154: narrow reversible coupling versus conventional FFNs\nLearning rates selected using validation only"
)
fig.tight_layout()
fig.savefig(ROOT / "selected_results.png", dpi=140)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, task in enumerate(p["tasks"]):
    for arm in ("gelu", "swiglu", "budget_gelu", "coupling", "fixed_coupling", "affine_coupling"):
        c = next(v for v in a["selected"] if v["task"] == task and v["arm"] == arm)
        cs = [r["cases"][i] for i in c["case_indices"]]
        steps = [v["step"] for v in cs[0]["history"]]
        for row, field in enumerate(("loss", "gradient_norm")):
            values = [st.mean(v["history"][j][field] for v in cs) for j in range(len(steps))]
            axes[row, col].plot(steps, values, label=arm)
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_yscale("log")
    axes[0, col].set_title(task)
    axes[1, col].set_xlabel("Optimizer updates")
axes[0, 0].set_ylabel("Minibatch MSE (seed mean)")
axes[1, 0].set_ylabel("Global preclip gradient norm")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3)
fig.suptitle(
    "H154: learning and gradient diagnostics\nLogged minibatches, not full-training-set loss"
)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(ROOT / "learning_curves.png", dpi=140)
plt.close(fig)
checks = read(ROOT / "checks.json")
structural = read(ROOT / "structural_check.json")
assert structural["passed"] and structural["source_sha256"] == sha(ROOT / "structural_check.py")
assert (ROOT / "structural_check_exit.txt").read_text().strip() == "0"
max_grad = max(max(v["relative_errors"]) for v in checks["gradient_checks"])
report = f"""# H154: learned narrow coupling must earn its compression

**{gate}.** Completed 120 fits and 72,000 optimizer updates on an RTX 4070
Laptop GPU using UV-managed Python. Task qualification: {qualifications}.
The broad VRAM/quality research goal remains open. No claim of architecture
novelty, universal dominance, or real-data validation follows from this screen.

## What changed and why

H153's even scalar controls mostly overlapped existing bias tangent directions.
This study learns projection directions inside a narrow, invertible additive
coupling core. Four blocks each permute 32 features into halves (a,b), apply

    a_new = a + (V GELU(U b + c) + e) / 2
    b_new = b

and undo the permutation. U maps 16 -> 8, V maps 8 -> 16. A full learned
32 -> 32 readout follows. Coupling has 2,176 trainable parameters, including
readout: 74.24% fewer than full GELU's 8,448. Narrow GELU has 2,208.
The fixed-hidden control freezes U and c while training V, e and readout.
The affine control replaces GELU by identity with the same parameterization.
Learned and fixed-hidden coupling have exactly the same initial function.

## Mathematical scope

Each coupling block is exactly invertible in real arithmetic: recover a by
subtracting the update computed from unchanged b. Its Jacobian, in permuted
coordinates, is [[I,K],[0,I]], with determinant one. An SVD of K reduces this
to independent two-dimensional shears; for each singular value k their singular
values are sqrt(1+k^2/4) +/- k/2. Thus k=||K|| gives valid extremal bounds.
Products of the block bounds apply to the core. Unconstrained learned matrices
provide no uniform depth-independent gradient bound; the readout can be singular.
No claim about optimizer gradients or stable inversion after arbitrary training
is established by the initialization checks.

Zeroing all down maps makes the core identity. Setting readout W=M and bias zero
then represents every linear map M exactly; the FP64 witness passes.
Nine FP64 inverse checks at input scales 0.1, 1 and 10 have maximum absolute
roundtrip error {max(checks["inverse_errors"]):.3e}. Small-model input and parameter
finite differences pass. Three GPU input/parameter-gradient comparisons against
CPU FP64 pass, maximum relative error {max_grad:.3e}. These are correctness
checks, not empirical proof of memory savings from reconstruction. Training uses
ordinary autograd and does not implement reconstructed backward.

Additive coupling is prior art: [NICE](https://arxiv.org/abs/1410.8516).
Activation reconstruction is also established: [RevNets](https://arxiv.org/abs/1707.04585).
The experiment tests this repository's narrow budget and learning bottleneck.
It differs from the earlier scalar-controlled rational twist and weight-shared
recurrence, but these differences do not establish novel priority.

## Post-hoc exact-representation obstruction

This argument was added during the run; it does not change the frozen gates or
explain the observed MSE quantitatively. Write the predictor as W R(x) + b with
R an invertible map from R^32 to R^32. If W is nonsingular, the predictor is
injective and cannot give identical outputs for x and -x. If W is singular, its
outputs lie in a proper affine subspace.

For T(x)_i = x_i x_(i+1), T(x)=T(-x). Also T(0)=0 and
T(e_i+e_(i+1))=e_i for all 32 cyclic indices. These outputs affinely span R^32.
Thus neither case for W can exactly represent this target, even on the finite
witness set containing zero and those signed pairs. Orthogonal output mixing,
translation and nonzero target scaling preserve both collision and affine-span
properties. A separate integer-arithmetic check verifies every witness.

This is **not a nonzero lower bound on attainable MSE**, nor proof that the
observed optimization gap is inevitable. It applies to the H152 square-readout
family too. It motivates testing an augmented reversible state with a rectangular
readout, or an explicitly noninvertible terminal map, rather than assuming that
any square learned readout removes every limitation of invertibility. The extra
state/readout must still earn its VRAM cost. Augmentation for representational
limitations is established prior art in
[Augmented Neural ODEs](https://arxiv.org/abs/1904.01681); no novel priority is claimed.

## Controlled protocol

Two nonlinear tasks from H152: a fixed random wide GELU teacher and an
orthogonally mixed cyclic-product target. New input/model seeds 347, 359, 373.
Functions are held fixed across seeds; this is not three independent teachers.
Each fixture has 4,096 training, 2,048 validation and 2,048 reporting examples.
Training-only target mean/RMS normalization. All arms share saved 600 x 128
batch indices, AdamW betas (0.9, 0.95), zero decay, global clipping at 1,
FP32 and disabled TF32. Rates 0.001 and 0.003 receive equal budgets. Biases start
at zero; coupling readout starts at identity. Structurally different arms cannot
share every tensor, but learned/fixed-hidden coupling initialization is matched.

A rate is selected per task/arm by mean validation MSE over all three seeds.
Reporting examples never select rates. Qualification requires both full GELU and
SwiGLU to beat zero prediction by at least 20% on every seed. Promotion requires
candidate reporting MSE within 1% of each full baseline and narrow GELU, and at
least 5% below both fixed-hidden and affine coupling, for every seed and both
qualified tasks. Failed qualification is inconclusive, not an impossibility result.
This 600-step budget is larger than H152's 300; cross-study numbers are not paired.

## All selected held-out results

| Task | Arm | Trainable parameters | LR | Mean MSE | Median MSE | Sample variance |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![All-arm reporting results](../results/narrow_coupling_v1/selected_results.png)

Candidate/comparator ratios below 1 favor coupling. The fixed/affine ablations
must be <= 0.95; all other ratios must be <= 1.01.

| Task | Seed | / GELU | / SwiGLU | / narrow GELU | / fixed-hidden | / affine | All ratio gates |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(comparisons)}

## Memory and optimization diagnostics

The following are means over selected seeds. Peak allocated memory includes
resident training inputs/targets and optimizer state. The evaluation peak is
measured after resetting peak counters, while training data, optimizer state and
final gradients remain resident. It is **not isolated deployment inference VRAM**.
Reserved peaks and all fit measurements are in result.json. Fit wall time includes
Python overhead and optimizer updates; no warmed paired timing, backward-only
latency, or equal-checkpoint resource gate was attempted at this small scale.
Do not extrapolate these numbers to d384 or claim reconstruction savings.

| Task | Arm | Training peak MiB | Evaluation peak MiB | 600-step seconds | Final preclip gradient norm |
|---|---|---:|---:|---:|---:|
{chr(10).join(resource_rows)}

![Learning diagnostics](../results/narrow_coupling_v1/learning_curves.png)

| Arm | Trainable parameters | Fixed buffer bytes |
|---|---:|---:|
{counts}

Core projection forward MACs are 4 * 2 * 16 * 8 = 1,024; readout adds 1,024,
so the candidate has 2,048 dense MACs/example versus full GELU's 8,192 and narrow
GELU's 2,048. These counts exclude activation, indexing, biases and launch costs;
GELU/shear composition depth and memory traffic differ. They are not measured FLOPs
or a speed prediction. Fixed-hidden parameters are counted as buffers, not free.

## Evidence and decision

Independent explicit CPU FP64 formulas re-score all 120 saved checkpoints on
validation and reporting splits (240 scores, batch 257, relative tolerance 1e-5).
All datasets regenerate bitwise; permutations and frozen weights match originals.
All 120 Adam states have step 600 and finite moments; all logged losses and gradient
norms are finite. There are 121 clean GPU allocation/reservation boundaries.
Source hashes, prior receipt, frozen protocol and maintained code hashes verify.
Raw checkpoints, per-fit histories, all rates and audit errors remain available.

This is a bounded screening decision for this architecture, initialization and
training recipe. {"A larger reconstructed-backward resource test is warranted before any promotion to real-data validation." if a["passed"] else "The candidate does not earn a reconstructed-backward implementation or a language-model run. Preserve this negative/inconclusive result and move beyond this exact narrow recipe."}

Reproduce with the UV Python path and launch.py stages prepare, worker, audit,
finish in that order in a fresh output directory after updating ROOT/module paths;
the existing directory intentionally refuses to overwrite its evidence. Hardware:
{(ROOT / "hardware.txt").read_text().strip()}. Python {platform.python_version()},
PyTorch {torch.__version__}. The frozen plan contains the full budget and gates.
"""
report_path = Path("research/narrow_coupling_results.md")
report_path.write_text(report, encoding="utf-8")
for path, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = path.read_text(encoding="utf-8")
    assert old.startswith(title)
    if path.name == "README.md":
        rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H154 narrow coupling](research/narrow_coupling_results.md):\n{gate}. 120 audited fits; 74% fewer parameters does not establish quality.\n\n"
    else:
        rest = old[len(title) :].lstrip().replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: learned narrow coupling\n\n[H154](narrow_coupling_results.md): **{gate}**.\n120 fits / 72,000 updates; 240 independent CPU FP64 scores verified.\nTask qualification: {qualifications}. No reconstructed-backward or real-data\nclaim. Broad VRAM/quality goal stays open.\n\n"
    path.write_text(title + "\n\n" + intro + rest, encoding="utf-8")
files = [
    f
    for f in ROOT.iterdir()
    if f.is_file() and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    report_path,
    Path("research/narrow_coupling_plan.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
receipt = dict(
    study="H154",
    status="EVIDENCE_VERIFIED",
    gate=gate,
    training_updates=72000,
    scored_states=120,
    scored_splits=240,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
print({k: v for k, v in receipt.items() if k != "files"})
