"""Document H084's bounded mathematical and checkpoint findings."""

import json
import os
import statistics
from pathlib import Path

ROOT = Path("results/latent_affine_mechanism_v1")


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value, exclusive=False):
    data = value.encode("utf-8")
    with path.open("xb" if exclusive else "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    assert path.read_bytes() == data


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join("---" for _ in headers)+"|",
                      *("| "+" | ".join(str(v) for v in row)+" |" for row in rows)])


r = read(ROOT / "result.json")
a = read("results/verification/latent_affine_mechanism_analysis_v1.json")
assert a["status"] == "PASS" and a["new_reset_rescores_exact"] == 24
assert not r["earns_training"] and r["optimizer_updates"] == 0
tasks = ("smooth", "oscillatory", "multiplicative", "piecewise")
mode_names = {"original": "Original affine", "reset_gain": "Remove gain", "reset_offset": "Remove offset",
              "reset_both": "Remove both", "fold_gain": "Fold gain into first factor",
              "fold_all": "Fold gain and cache output bias"}
ratios = table(["Intervention", "Aggregate MSE / original", *[t.title() for t in tasks]],
               [(label, f"{r['reset_to_original_ratios'][mode]:.6f}",
                 *[f"{r['task_reset_to_original_ratios'][t][mode]:.6f}" for t in tasks])
                for mode, label in mode_names.items()])
cells = table(["Task", "Seed", "Original MSE", "Remove gain", "Remove offset", "Remove both", "Factorial interaction"],
              [(row["task"], row["seed"], *[f"{row['modes'][mode]['mse']:.6f}"
                                          for mode in ("original", "reset_gain", "reset_offset", "reset_both")],
                f"{row['factorial_mse_interaction']:+.6f}") for row in r["rows"]])
mean_rows = []
for task in tasks:
    for mode in ("original", "reset_gain", "reset_offset", "reset_both"):
        values = [row["modes"][mode] for row in r["rows"] if row["task"] == task]
        mean_rows.append((task, mode_names[mode],
                          f"{statistics.mean(v['double_residual_mse'] for v in values):.6f}",
                          f"{statistics.mean(v['mean_error_energy'] for v in values):.6f}",
                          f"{statistics.mean(v['centered_error_energy'] for v in values):.6f}"))
mean_table = table(["Task", "Mode", "Residual MSE (FP64)", "Mean-error energy", "Centered error"], mean_rows)
origin_table = table(["Task", "Seed", "Original output norm at zero", "Original eight-JVP norm", "After removing offset"],
                     [(row["task"], row["seed"], f"{row['origin']['original']['output_norm']:.6f}",
                       f"{row['origin']['original']['jvp_norm']:.6f}", "Both exactly zero") for row in r["rows"]])
fp64_output = max(row["fp64_max_absolute_errors"][mode]["output"] for row in r["rows"] for mode in ("fold_gain", "fold_all"))
fp64_gradient = max(row["fp64_max_absolute_errors"][mode]["input_gradient"] for row in r["rows"] for mode in ("fold_gain", "fold_all"))
headline = """The internal affine control depends on both learned gain and offset, but its
gain is algebraically redundant with the first weight factor. Removing gain
without compensation multiplies aggregate held-out MSE by 2.972; removing offset
multiplies it by 1.843. Folding the affine operations preserves predictions
within the declared FP64 and FP32 tolerances. This is a checkpoint-mechanism
result, not a new trained model, causal training explanation or speed result."""
report = f"""# H084 - What the internal affine control learned

{headline}

All 12 selected H083 affine checkpoints pass. Independent verification exactly
reproduces the 24 new gain-only/offset-only resets and their prediction hashes.
There are zero training updates, no new rate selection, and no earned language
or full-model resource trial. H083's curves remain rejected; its affine control
still trails narrow GELU by 6.512% aggregate held-out MSE.

## Algebra and the limited expressivity claim

For a column-vector projection, write

    f(x) = P_out^-1 B2 P_mid (G B1 x + b).

G repeats each learned group gain g=1+0.5*tanh(theta_a) on that group's
first-factor outputs. The vector b repeats 0.5*tanh(theta_b) within each group.
Left-multiplying B1 by G preserves B1's block-diagonal structure and parameter
shape. Therefore gain alone adds no represented functions when B1 is trainable.
It can still affect optimization through a different parameterization.

There are two ways to preserve an already trained function in real arithmetic:

1. Replace B1 by G B1, set gain controls to zero, and retain the offset.
2. Replace B1 by G B1, remove the affine module and add the cached output vector
   c=P_out^-1 B2 P_mid b after the ordinary two-factor projection.

The first removes 24 effective gain degrees of freedom from one FFN; the
diagnostic copy retains their zeroed parameter slots. The second removes all
48 curve parameters and stores 4,480 bias-buffer values across up/gate/down:
17,920 bytes in FP32. It has 350,208 factor parameters plus those buffers.
This preserves the evaluated function, not Adam states, gradients with respect
to a changed parameterization, or its original optimization trajectory. It does
not prove lower runtime, BF16 fidelity or zero deployment storage overhead.

An isolated bias-free SwiGLU with zero-preserving differentiable projections has
F(0)=0 and DF(0)=0, by the product rule and SiLU(0)=0. H083's tanh/sine curves
also preserve zero and retain this restriction. Internal offsets can remove it:
the verified width-one witness U(x)=1/4, V(x)=x and D(x)=x has
F(x)=SiLU(1/4)*x, with strictly positive slope at zero. The actual grouped-factor
implementation reproduces this witness.

This is a local statement at the origin, already related to the earlier H037
and H072 analyses. It is not a whole-Transformer gradient guarantee, a
distributional approximation bound, a proof that every learned Jacobian has
full rank, or a claim that bias is a new mechanism. The
[frozen derivation](latent_affine_mechanism_plan.md) gives its assumptions.

Featurewise affine modulation appears in [FiLM](https://arxiv.org/abs/1709.07871);
our parameters are not outputs of a conditioning network. Bias-only adaptation
appears in [BitFit](https://arxiv.org/abs/2106.10199), which studies fine-tuning
pretrained models rather than training these structured layers from scratch.
The gated baseline follows [GLU variants](https://arxiv.org/abs/2002.05202).
Those papers do not establish the present checkpoint result or our gold target.

## Resetting is different from folding

Ratios are geometric means of paired errors over all 12 task/seed checkpoints,
or over the three seeds within one task. Values above one mean worse error.
Projection weights stay fixed for resets; gain folding changes the first factor
to compensate exactly in real arithmetic.

{ratios}

The two reset effects are not additive. Removing both can be less harmful than
removing gain alone, as in the multiplicative and piecewise tasks. These are
coadapted checkpoints, so a reset is not equivalent to training a gain-only or
offset-only model. The results cannot identify which training mechanism caused
H083's 15.03% advantage over plain. Such a claim requires separately controlled
training and must still beat narrow GELU.

## Numerical folding checks

All 24 original/folded CPU FP64 comparisons pass for output and input gradient
on the fixed 9x384 probe. The maximum absolute output difference is
{fp64_output:.6e}; the maximum input-gradient difference is {fp64_gradient:.6e}.
The tolerance was rtol=atol=1e-10, fixed before execution.

Across both folding modes and all checkpoints, the largest relative FP32
held-out MSE change is {r['maximum_fold_mse_drift']:.6e}, below the frozen 1e-5
limit. Original and reset-both scores reproduce H083 exactly. FP32 folding
changes arithmetic order, so bitwise equality is not claimed. No BF16,
compiler, latency, training-gradient or full-model resource qualification was run.

## Response at the origin

For each checkpoint, eight fixed FP64 input directions probe the derivative at
zero. Removing offset produces exactly zero output and JVPs in every case,
whether gain is retained or reset. The table describes finite probes; it does
not certify Jacobian rank or gradient behavior away from the origin.

{origin_table}

## Mean mismatch and remaining error

For residual e=prediction-target, squared error decomposes into

    E[||e||^2]/d = ||E[e]||^2/d + E[||e-E[e]||^2]/d.

The following entries are arithmetic averages over the three seeds. Reductions
use FP64 residuals from FP32 predictions. No target centering or refitting was
performed; this is a descriptive decomposition of the same reporting errors.

{mean_table}

The decompositions reproduce from saved per-coordinate residual moments. Their
sums match the FP64 error to the declared 1e-12 tolerance and the original
FP32-accumulated MSE to 1e-6. Mean correction and remaining functional error
must both be considered; improvements cannot be labeled complex-pattern
learning solely from aggregate MSE.

## Every checkpoint intervention

The interaction is E(no gain,no offset)-E(no gain,offset)-E(gain,no offset)
+E(gain,offset). It is descriptive and uses the fixed trained weights.

{cells}

## Allocation, verification and next decision

The fixed H083 reporting split has 4,096 vectors per task, width 384, scored in
batches of 256. Six modes on each of 12 selected checkpoints give 72 passes and
294,912 example presentations. The study worker took
{read(ROOT / 'process.json')['elapsed_seconds']:.2f} seconds, including CPU FP64 probes and GPU FP32 scores.
Independent verification repeated the two new reset modes: 24 passes and
98,304 examples in {read(ROOT / 'analysis_process.json')['elapsed_seconds']:.2f} seconds, with exact MSE and prediction hashes.
All original checkpoints, sources, data, selections and completed reports remain
unchanged. Sixty input checkpoint/metadata files are pinned by hash, size and
mtime; the H083 archive anchors remain intact. All 131 scientific sources and
73 frozen plans are retained. There are no numerical retries or training updates.

This establishes a representational distinction between gain and offset and an
evaluation simplification. It does not promote a candidate. A distinct future
training experiment would need gain-only, offset-only, full-affine and
fixed-function controls, both narrow baselines, equal budgets and a material
gain. The old outer-affine failures and H083 curve failures remain closed.
The active tree stays at three model folders, six variants and nine recipes;
the full research goal remains unmet.

[Frozen plan](latent_affine_mechanism_plan.md),
[all observations](../results/latent_affine_mechanism_v1/result.json),
[independent audit](../results/verification/latent_affine_mechanism_analysis_v1.json),
[preceding fitting result](latent_activation_recovery_results.md).
"""
report_path = Path("research/latent_affine_mechanism_results.md")
assert not report_path.exists()
docs = {n: Path(n).read_text(encoding="utf-8") for n in ("README.md", "research/CURRENT_STATE.md",
        "research/PROGRESS_OVERVIEW.md", "research/idea_bank.md", "research/learnable_activation_domain.md")}
assert all("latent_affine_mechanism_results.md" not in text for text in docs.values())
root_note = """The [affine mechanism study](research/latent_affine_mechanism_results.md)
separates gain and offset on all 12 selected checkpoints, without retraining.
Both affect predictions, but gain can be absorbed into existing factors. Folding
the affine operations preserves FP32 error within a relative 1.7e-8; runtime
benefit and a causal training advantage remain unestablished.

"""
marker = "The [internal-activation fitting screen]"
assert docs["README.md"].count(marker) == 1
docs["README.md"] = docs["README.md"].replace(marker, root_note+marker)
note = """## Latest mechanism result

H084 separates learned internal gain and offset without new training. Removing
gain multiplies aggregate checkpoint MSE by 2.972; removing offset multiplies it
by 1.843. Gain alone does not add represented functions, because it can be
absorbed into the first factor. An offset can give an isolated SwiGLU a nonzero
linear response at zero. These are local algebra and checkpoint results.

All 24 independent new reset scores match exactly. Folding the affine operations
preserves FP32 held-out error within a relative 1.7e-8, with explicit bias-buffer
storage. No BF16, speed, training-causality or full-model benefit is established.
No candidate is promoted. [Completed analysis](latent_affine_mechanism_results.md).

"""
marker = "## Latest completed nonlinear fitting study"
assert docs["research/CURRENT_STATE.md"].count(marker) == 1
docs["research/CURRENT_STATE.md"] = docs["research/CURRENT_STATE.md"].replace(marker, note+marker)
marker = "## Latest learned-activation result"
assert docs["research/PROGRESS_OVERVIEW.md"].count(marker) == 1
overview = """## Latest follow-up: what the affine control does

All 12 selected affine checkpoints were checked without retraining. Removing
gain alone makes aggregate error 2.97 times worse; removing offset makes it 1.84
times worse. The effects interact. Gain can instead be folded into existing
weights while preserving the function; offsets can be cached as output biases.
The measured FP32 error drift is below a relative 1.7e-8. This is a mathematical
and numerical simplification, not a measured speed or training improvement.
[Mechanism report](latent_affine_mechanism_results.md).

"""
docs["research/PROGRESS_OVERVIEW.md"] = docs["research/PROGRESS_OVERVIEW.md"].replace(marker, overview+marker)
docs["research/idea_bank.md"] += "\n## H084 - Internal affine mechanism (CHARACTERIZED; NO PROMOTION)\n\n"+headline+"""

All 12 checkpoints, 24 FP64 folding pairs, 48 origin cases and the nonzero-slope
witness pass. Independent checks exactly reproduce the 24 new reset scores and
prediction hashes. Gain folding changes coordinates; uncompensated reset changes
the function. Gains and offsets cannot be assigned causal training credit from
these interventions. No new training, rate search, BF16 or runtime claim follows.
[Complete analysis](latent_affine_mechanism_results.md),
[frozen plan](latent_affine_mechanism_plan.md).
"""
docs["research/learnable_activation_domain.md"] += """
## Separating an affine function from learned curvature

The [H084 checkpoint analysis](latent_affine_mechanism_results.md) shows internal
affine gain can be absorbed into a block-diagonal factor without increasing its
shape or function family. An internal offset can change the isolated SwiGLU's
zero-input output and derivative. This distinction explains what each parameter
can represent, but does not explain its optimization effect causally.

Both components matter to the selected trained checkpoints: deleting gain or
offset without compensation multiplies aggregate error by 2.972 or 1.843.
Compensated folding preserves FP32 error within a relative 1.7e-8. It adds cached
bias storage and has no measured runtime or BF16 qualification. A useful learned
parameter need not enlarge the function family, and folding it after training
does not reproduce training without it. Controlled fitting remains necessary
before claiming a better mechanism; narrow GELU still wins the H083 comparison.
"""
write(report_path, report, exclusive=True)
for name, value in docs.items():
    write(Path(name), value)
print(json.dumps({"status": "PASS", "report": report_path.as_posix(), "updated_docs": list(docs),
                  "optimizer_updates": 0, "earns_training": False}), flush=True)
