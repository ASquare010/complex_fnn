"""Write the completed H083 result and replace running descriptions in navigation."""

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path("results/latent_activation_recovery_v1")


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, text, exclusive=False):
    payload = text.encode("utf-8")
    with path.open("xb" if exclusive else "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    assert path.read_bytes() == payload


def replace_section(text, start, end, replacement):
    assert text.count(start) == text.count(end) == 1
    a, b = text.index(start), text.index(end)
    assert a < b
    return text[:a]+replacement.rstrip()+"\n\n"+text[b:]


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join("---" for _ in headers)+"|",
                      *("| "+" | ".join(str(x) for x in row)+" |" for row in rows)])


p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
a = read("results/verification/latent_activation_recovery_analysis_v1.json")
assert a["status"] == "PASS" and r["scientific_verdict"] == "REJECTED_AT_THIS_FITTING_BUDGET"
assert not any(g["earns_full_model_resource_qualification"] for g in a["gates"].values())
assert a["selection_rescores_exact"] == 192 and a["selected_heldout_rescores_exact"] == 96
assert a["reset_ablations_exact"] == 36 and a["analysis_optimizer_updates"] == 0
names = dict(zip(p["forms"], ("BlockShuffle plain", "Internal affine", "Internal tanh curve",
                              "Internal sine curve", "Narrow SwiGLU", "Narrow GELU", "Full SwiGLU", "Full GELU")))
tasks = p["tasks"]
coordinator = read(ROOT / "coordinator_status.json")
minutes = (datetime.fromisoformat(coordinator["finished_utc"])-datetime.fromisoformat(coordinator["started_utc"])).total_seconds()/60
changes = {f: 100*(a["ratios"][f]["plain"]-1) for f in p["forms"]}
headline = f"""Both internal curves improve synthetic fitting over plain BlockShuffle, but
neither qualifies: tanh changes aggregate held-out MSE by {changes['latent_tanh']:+.3f}%
and sine by {changes['latent_sine']:+.3f}%, while the equal-count affine control changes
it by {changes['latent_affine']:+.3f}%. Both curves lose to affine and narrow GELU.
The positive-control assay passes. These fixed recipes are closed; no language
or full-model resource run is earned."""

aggregate = table(["Form", "Weights per FFN", "MSE change vs plain", "Ratio vs affine", "Ratio vs narrow GELU"],
                  [(names[f], f"{p['counts'][f]:,}", f"{changes[f]:+.3f}%", f"{a['ratios'][f]['latent_affine']:.6f}",
                    f"{a['ratios'][f]['narrow_gelu']:.6f}") for f in p["forms"]])
task_table = table(["Form", *[t.title() for t in tasks]],
                   [(names[f], *[f"{r['task_mean_mse'][t][f]:.6f}" for t in tasks]) for f in p["forms"]])
gate_table = table(["Frozen gate", "Tanh", "Sine"],
                   [(name.replace("_", " "), *["PASS" if a["gates"][f]["tests"][name] else "FAIL"
                                                for f in ("latent_tanh", "latent_sine")])
                    for name in a["gates"]["latent_tanh"]["tests"]])
resource_table = table(["Form", "Mean median update ms", "Max peak MiB", "Mean clipped %", "Upper LR / 12"],
                       [(names[f], f"{a['resources'][f]['mean_cell_median_update_ms']:.3f}",
                         f"{a['resources'][f]['max_peak_allocated_mib']:.2f}",
                         f"{100*a['resources'][f]['mean_clip_fraction']:.3f}",
                         a["resources"][f]["upper_rate_selected"]) for f in p["forms"]])
reset_table = table(["Learned function", "Reset / original MSE", *[t.title() for t in tasks]],
                    [(names[f], f"{a['learned_shapes'][f]['reset_to_original_geometric_ratio']:.6f}",
                      *[f"{a['learned_shapes'][f]['per_task_reset_to_original_geometric_ratio'][t]:.6f}" for t in tasks])
                     for f in ("latent_affine", "latent_tanh", "latent_sine")])
rate_table = table(["Task", "Form", "Seed 17", "Seed 29", "Seed 43"],
                   [(t, names[f], *[str(next(row["rate"] for row in r["selected_rows"]
                                             if (row["task"], row["form"], row["seed"]) == (t, f, seed)))
                                     for seed in p["seeds"]]) for t in tasks for f in p["forms"]])
report = f"""# H083 - Internal learned activations: completed fitting result

{headline}

All 192 allocated runs finish, with three optimization seeds, two rates and
600 updates each. Independent verification reproduces all 192 selection scores,
96 selected held-out scores and 36 reset ablations exactly, without updates.
This study provides synthetic function-fitting evidence, not language NLL,
convergence, broad generalization or a breakthrough. The research goal is unmet.

## What changed and why

The curve sits between the two BlockShuffle factors, before their middle
permutation. The existing outer SwiGLU stays in place. Eight groups share two
controls each, independently for up/gate/down: 48 extra parameters per FFN.
Each curve starts at the identity, sharing every ordinary factor and initial
output with plain BlockShuffle. The equal-count affine control separates a
curved function from the option to learn gain and offset.

For a = 0.5 tanh(theta_a) and s = exp(log(4) tanh(theta_b)):

    phi(z) = z + a*s*q(z/s)^2, where q is tanh or sin.

The affine control is phi(z) = (1 + 0.5 tanh(theta_a))*z + 0.5 tanh(theta_b).
Tanh's real scalar input derivative is bounded by about [0.6151, 1.3849]; sine's
by [0.5, 1.5]. These are local scalar bounds. They do not remove rank limits of
the factors or guarantee whole-network gradients, convergence or GPU speed.
The sine variant is related to [Snake](https://arxiv.org/abs/2006.08195).
The [theory note](latent_activation_theory.md) gives exact assumptions and the
constrained quadratic Bezier identity; no new-family priority is established.

## Held-out quality

Aggregate ratios are geometric means of 12 paired task/seed MSE ratios. Lower
is better. Each rate is selected using a separate 4,096-example split, then the
final checkpoint is scored on 4,096 held-out examples. Rows below the table are
arithmetic means over three optimization seeds for each individual task.

{aggregate}

The curves improve over plain in every seed's four-task aggregate. However,
tanh is {100*(a['ratios']['latent_tanh']['latent_affine']-1):.3f}% worse than equal-count affine and
sine is {100*(a['ratios']['latent_sine']['latent_affine']-1):.3f}% worse. Both beat narrow SwiGLU but fail to beat
narrow GELU. Affine also remains {100*(a['ratios']['latent_affine']['narrow_gelu']-1):.3f}% worse than narrow GELU
in the aggregate; it is a diagnostic control and is not promoted by this plan.

{task_table}

Full GELU satisfies the required positive-control improvement on at least two
tasks. Absolute error and the zero-predictor comparisons are retained in the
[result JSON](../results/latent_activation_recovery_v1/result.json). Normalization
divides by training population standard deviation without centering; these are
variance-scaled MSE values, not centered NMSE or language loss.

{gate_table}

Both curves satisfy the count, finite, per-task regression and plain-improvement
gates. Each fails two quality gates. No threshold or training budget is changed
after looking at the result.

## Did the learned functions matter?

Yes, the final checkpoints depend on the corrections. Resetting curved amplitude
to zero, or both affine controls to zero, worsens aggregate held-out error. All
projection weights stay fixed; curved scales stay fixed. Every reset score is
independently reproduced. Values above 1 mean the reset made error worse.

{reset_table}

This tests dependence after coadaptation; it does not prove the correction caused
the training gain. For the oscillatory task, curve resets have almost no effect,
despite the sine function's periodic form. A useful periodic scalar does not
automatically learn a useful periodic high-dimensional mapping.

The saved shape diagnostics show no controls crossing the recorded saturation
threshold. Affine's sampled slopes are all below one, approximately 0.543 to
0.726 across selected models. This suggests gain/offset and optimization deserve
separate analysis, but does not identify which caused the benefit. Any follow-up
must isolate those mechanisms and include narrow GELU; this is not an automatic
retry, longer allocation or promotion of affine.

## Compute and memory

{resource_table}

The time column averages each selected run's median update time after its first
50 updates. CUDA is synchronized around every update. Peak allocation is the
maximum across selected runs and includes the standalone GPU dataset cache.
All entries are native FP32 on the RTX 4070 Laptop GPU, TF32 off. These are
sequential laptop timings, not full-Transformer memory, compiled performance or
serving throughput. The extra curve computation roughly doubles plain update
time despite adding only 48 weights. Parameter efficiency alone is insufficient.

All 12 selections for each learned-function form choose the upper tested rate,
0.003. That is a limitation of this fixed two-rate screen. No additional rates
or steps were allocated after observing it.

## Data and budget

The four tasks use 73,728 fixed uniform input vectors of dimension 384:
65,536 training, 4,096 rate-selection and 4,096 reporting examples. A fixed input
permutation breaks the original contiguous group alignment; approximately
86.98% of neighboring target coordinates cross an input group. A fixed output
rotation mixes outputs. The formulas are smooth, oscillatory, multiplicative
and piecewise; none uses an architecture teacher or an easy linear task.

Three optimization seeds share one dataset. Both rates receive exactly 600
updates with batch 256, AdamW betas (0.9, 0.95), zero decay, clipping at one,
and the inherited factor/narrow learning-rate calibration. The whole allocation
is 115,200 updates and 29,491,200 training-example presentations. There are zero
language targets and zero full-model resource workers. No training checkpoint
recomputation or compiler is used in these cells.

The coordinator ran from {coordinator['started_utc']} to
{coordinator['finished_utc']}: {minutes:.2f} minutes including preflight,
evaluation, startup and durable artifact writing. Independent verification took
{read(ROOT / 'analysis_process.json')['elapsed_seconds']:.2f} seconds and presented 786,432 selection plus
540,672 held-out/reset examples, with zero optimizer updates.

## Failure preservation and verification

H082's first preflight stopped at an adapter argument collision, before any
fitting update. H083 changes only the binding mechanism and artifact destination.
All eight unchanged checks pass in the explicit recovery; there are two total
preflight attempts, one explicit preflight repetition and no fitting repetition.
The original failure remains in its [report](latent_activation_results.md).

The independent audit regenerates all inputs, target rotations/permutations,
training-only scales and sampling streams exactly. It verifies every checkpoint,
600-step optimizer history, moment shape/value finiteness, parameter membership,
learning-rate group, initialization hash, final hash and RNG state. It checks
selection order against source and persisted file timestamps. All 128 scientific
sources and the original 27 H082 files remain unchanged. Seven original
observations and six tensor payloads match recovery results; all 120 saved
identity/checkpoint gradient pairs match exactly.

Active scope remains three model folders, six registered variants and nine
recipes. The unchanged active test suite is not replaced by these eight isolated
checks. Negative fitting evidence closes these two fixed recipes without
discarding their measurements or overstating the local derivative proofs.

## Every selected rate

Both rate checkpoints and every training record remain under the study's cells
directory. The table records choices made before held-out scoring.

{rate_table}

[Frozen original plan](latent_activation_plan.md),
[explicit recovery plan](latent_activation_recovery_plan.md),
[independent audit](../results/verification/latent_activation_recovery_analysis_v1.json),
[all measurements](../results/latent_activation_recovery_v1/result.json).
"""
report_path = Path("research/latent_activation_recovery_results.md")
assert not report_path.exists()
docs = {n: Path(n).read_text(encoding="utf-8") for n in
        ("README.md", "research/CURRENT_STATE.md", "research/PROGRESS_OVERVIEW.md",
         "research/idea_bank.md", "research/learnable_activation_domain.md")}
docs["README.md"] = replace_section(docs["README.md"], "The [internal-activation fitting screen]", "The [BLAST learning screen]",
    """The [internal-activation fitting screen](research/latent_activation_recovery_results.md)
is complete: 192 runs, three seeds, all selection and selected held-out scores
independently reproduced. Tanh and sine curves lower error by 5.76% and 7.07%
versus plain, but the same-size affine control improves 15.03%. Both curves lose
to affine and narrow GELU, so neither earns a language trial. Local gradient
proofs remain valid within their scope; the full research target is unmet.""")
docs["research/CURRENT_STATE.md"] = replace_section(docs["research/CURRENT_STATE.md"],
    "## Active nonlinear fitting study", "## Latest completed short screen", f"""## Latest completed nonlinear fitting study

{headline}

H083 completes all 192 cells: four permuted nonlinear tasks, two rates, three
seeds, 600 updates and batch 256. The curves add 48 weights per FFN. All eight
recovery checks pass; independent analysis exactly reproduces 192 selection
scores, 96 selected held-out scores and 36 resets. Resetting learned curves
worsens error, but does not explain the training advantage over plain.
No active model is added. [Completed result](latent_activation_recovery_results.md),
[original adapter failure](latent_activation_results.md),
[scalar proofs and limits](latent_activation_theory.md).
""")
docs["research/PROGRESS_OVERVIEW.md"] = replace_section(docs["research/PROGRESS_OVERVIEW.md"],
    "## Current work", "### Latest completed language screen", """## Latest learned-activation result

The internal-activation study finished all 192 synthetic runs in 13.06 minutes.
Tanh and sine curves improve aggregate held-out error by 5.76% and 7.07% over
plain, but equally sized affine improves 15.03%. Both curves lose to affine and
narrow GELU, so these fixed recipes are closed. No language trial is earned.

Each learned function adds 48 weights per FFN. All eight checks pass; independent
verification exactly reproduces 192 selection scores, 96 selected held-out scores
and 36 activation resets. The current curves are useful to their trained models,
but do not provide a combined quality/runtime win. The active tree remains three
model folders. [Completed study](latent_activation_recovery_results.md).
""")
old = "tracker recorded about 28.3 hours of accumulated agent work: reading, coding,"
assert old in docs["research/PROGRESS_OVERVIEW.md"]
docs["research/PROGRESS_OVERVIEW.md"] = docs["research/PROGRESS_OVERVIEW.md"].replace(old,
    "tracker recorded about 30.1 hours of accumulated agent work: reading, coding,")
ledger = docs["research/idea_bank.md"]
marker = "## H083 - Binding-only recovery and nonlinear fitting (EXPERIMENTING)"
assert ledger.count(marker) == 1
docs["research/idea_bank.md"] = ledger[:ledger.index(marker)]+f"""## H083 - Binding-only recovery and nonlinear fitting (COMPLETE: CURVES REJECTED)

{headline}

All eight checks pass after one explicit adapter-only preflight recovery. All
192 fitting cells finish on their first attempt. Independent analysis reproduces
192 selection scores, 96 selected held-out scores and 36 reset ablations exactly.
The curves add 48 weights per FFN but take about 9.8 ms/update versus plain's
5.3 ms and narrow GELU's 2.4 ms in standalone FP32 fitting. Learned corrections
affect final predictions; that dependence is not a causal training explanation.
Affine's gain/offset contributions remain unresolved, and it also loses to
narrow GELU. No automatic extension, new active model or language trial follows.
[Complete result](latent_activation_recovery_results.md),
[recovery plan](latent_activation_recovery_plan.md), [local proofs](latent_activation_theory.md).
"""
domain = docs["research/learnable_activation_domain.md"]
domain = domain.replace("## Internal placement under test", "## Internal placement: completed synthetic comparison")
old = """The original adapter failed before fitting. One documented binding-only recovery
passes all eight unchanged checks; its nonlinear fitting comparison is running.
Actual held-out improvements, useful learned shape and GPU costs remain to be
measured before promotion. [Recovery plan](latent_activation_recovery_plan.md)."""
assert old in domain
docs["research/learnable_activation_domain.md"] = domain.replace(old, """The original adapter failed before fitting. One documented binding-only recovery
passes all eight unchanged checks and completes 192 runs. Tanh and sine reduce
aggregate held-out MSE by 5.76% and 7.07% versus plain; equal-count affine reduces
it by 15.03%. Both curves lose to affine and narrow GELU and take about twice
plain's standalone update time. Neither earns full-model resources or language
training. All selected held-out and reset scores reproduce exactly.

Resetting the learned curves worsens final error, so the models use them. That
does not establish that curvature is better than gain/offset or explain training
causally. The affine control itself remains behind narrow GELU. The outcome is
evidence for studying the simpler mechanism before adding more curve complexity;
it does not authorize another tuning sweep. [Completed comparison](latent_activation_recovery_results.md).""")
write(report_path, report, exclusive=True)
for name, text in docs.items():
    write(Path(name), text)
print(json.dumps({"status": "PASS", "report": report_path.as_posix(), "updated_docs": list(docs),
                  "scientific_verdict": r["scientific_verdict"], "optimizer_updates": 0}), flush=True)
