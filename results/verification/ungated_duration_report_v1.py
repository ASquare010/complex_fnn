"""Report the audited single-seed H078 duration test without a convergence claim."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/ungated_duration_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


r = read(ROOT / "result.json")
a = read("results/verification/ungated_duration_analysis_v1.json")
assert a["status"] == "PASS" and a["all_6_rescores_exact"]
assert a["independently_rederived_gates"] == r["gates"]
forms = ("full_swiglu", "full_gelu", "narrow_swiglu", "narrow_gelu", "plain", "gelu_matched")
names = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "narrow_swiglu": "Narrow SwiGLU",
    "narrow_gelu": "Narrow GELU",
    "plain": "BlockShuffle SwiGLU",
    "gelu_matched": "BlockShuffle GELU h3264",
}
colors = {
    "full_swiglu": "#333333",
    "full_gelu": "#777777",
    "narrow_swiglu": "#b85c00",
    "narrow_gelu": "#cc6677",
    "plain": "#276b9b",
    "gelu_matched": "#7853a6",
}
rows = {f: r["rows"][f"{f}_seed17"] for f in forms}
primary = r["primary_pass"]
verdict = "PASSES the frozen duration gates" if primary else "FAILS the frozen duration gates"
next_step = (
    "This fixed recipe earns only a separately frozen replication with additional long-budget seeds."
    if primary
    else "This fixed duration recipe earns no automatic extra steps, rates, widths, activations or optimizer repair."
)
summary = [
    "| Form | Fixed peak LR | Final NLL | FFN weights | Total weights | Peak MiB | Mean timed update ms | Clipped |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
resources = [
    "| Form | Training targets/s | Full-sequence inference targets/s | Inference latency ms | Final preclip norm |",
    "|---|---:|---:|---:|---:|",
]
trends = [
    "| Form | Step 2400 NLL | Step 3200 NLL | Relative final-800 change | Final above best logged | Plateau diagnostic |",
    "|---|---:|---:|---:|---:|---|",
]
histories = {}
for f, m in rows.items():
    h = list(
        map(json.loads, (ROOT / "runs" / f"{f}_seed17" / "history.jsonl").read_text().splitlines())
    )
    histories[f] = h
    summary.append(
        f"| {names[f]} | {m['training']['learning_rate']:.4f} | {m['validation_loss']:.9f} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.3f} | {m['timed_training_seconds'] * 1000 / 3190:.3f} | {100 * m['clipped_step_fraction']:.2f}% |"
    )
    resources.append(
        f"| {names[f]} | {m['training_tokens_per_second']:.0f} | {m['inference_tokens_per_second']:.0f} | {m['inference_latency_ms']:.3f} | {m['final_gradient_norm']:.6f} |"
    )
    trends.append(
        f"| {names[f]} | {h[-2]['validation_loss']:.9f} | {h[-1]['validation_loss']:.9f} | {m['late_nll_change_percent']:+.4f}% | {m['final_above_best_percent']:+.4f}% | {'PASS' if r['operational_late_plateau'][f'{f}_seed17'] else 'FAIL'} |"
    )
comparisons = [
    "| Matched GELU versus | Absolute NLL difference | Relative NLL difference |",
    "|---|---:|---:|",
]
for f, v in r["comparisons"].items():
    comparisons.append(
        f"| {names[f]} | {v['nll_difference']:+.9f} | {v['relative_nll_percent']:+.4f}% |"
    )
gates = ["| Frozen primary gate | Result |", "|---|---|"]
for name, value in r["gates"].items():
    gates.append(f"| {name.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |")
summary, resources, trends, comparisons, gates = map(
    "\n".join, (summary, resources, trends, comparisons, gates)
)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
for f, h in histories.items():
    axes[0].plot(
        [800, 1600, 2400, 3200],
        [v["validation_loss"] for v in h[1:]],
        color=colors[f],
        marker="o",
        label=names[f],
    )
axes[0].set_xlabel("Optimizer update (fresh 3,200-step schedule)")
axes[0].set_ylabel("Validation NLL; seed 17")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.15)
refs = [f for f in forms if f != "gelu_matched"]
for i, f in enumerate(refs):
    gap = r["comparisons"][f]["relative_nll_percent"]
    axes[1].scatter(gap, i, color=colors[f], s=60)
    axes[1].annotate(
        f"{gap:+.2f}%", (gap, i), xytext=(5, 6), textcoords="offset points", fontsize=9
    )
    if f.startswith("full_"):
        axes[1].scatter(1, i, marker="|", s=180, color="#aa3333")
axes[1].axvline(0, color="#888888", linewidth=1)
axes[1].set_yticks(range(5), [names[f] for f in refs])
axes[1].invert_yaxis()
axes[1].margins(x=0.2)
axes[1].set_xlabel(
    "Matched GELU relative NLL difference (%)\nNegative favors candidate; red marks: full-control 1% caps"
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "H078: WikiText-2, seed 17, fixed rates, 3,200 updates\nReused development validation; single-seed result, no convergence claim",
    fontsize=12,
)
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/ungated_duration.{ext}", dpi=160)
plt.close(fig)
m = rows["gelu_matched"]
plain_gap = r["comparisons"]["plain"]["relative_nll_percent"]
plain_time = 100 * (m["timed_training_seconds"] / rows["plain"]["timed_training_seconds"] - 1)
worker_seconds = sum(
    read(ROOT / "processes" / f"{f}_seed17.json")["elapsed_seconds"] for f in forms
)
plateau = sum(r["operational_late_plateau"].values())
text = f"""# H078 - Matched GELU at 3,200 steps

**Matched-budget BlockShuffle GELU {verdict}.** All six fresh seed-17 trials
complete. Matched final NLL is **{m["validation_loss"]:.9f}**, with **70.3125%
fewer FFN weights**. {next_step} The complete research goal remains unmet.

Compared with plain at its fixed selected rate, candidate NLL changes by
**{plain_gap:+.4f}%** and mean timed update changes by **{plain_time:+.2f}%**;
negative means lower loss or shorter time. This single seed does not establish
consistent plain-quality superiority or replicate long-duration behavior.

## Every allocated endpoint

{summary}

Every model keeps its H077 recipe. Rates originally came from H076's equal
three-rate short screen; they are not newly optimized for this duration.
There is no selected intermediate checkpoint or omitted trial.

{comparisons}

{gates}

Both full-control NLL allowances are 1%, while BOTH calibrated narrow NLLs must
be beaten strictly. The separate >=0.2% narrow margins are descriptive:
narrow SwiGLU **{"YES" if r["descriptive_point_two_percent_narrow_margin"]["narrow_swiglu"] else "NO"}**,
narrow GELU **{"YES" if r["descriptive_point_two_percent_narrow_margin"]["narrow_gelu"] else "NO"}**.
These diagnostics do not alter any primary threshold after results.

## Duration and resources

{trends}

The operational late-plateau diagnostic passes **{plateau}/6** models. It requires
absolute final-800 NLL change <=0.2% and final NLL <=0.2% above the best logged
endpoint. Even a passing plateau diagnostic would not prove optimal convergence.
No confidence interval, significance or replicated long-budget claim follows
from one seed. The earlier three-seed result concerns 800-step schedules only.

![Duration trajectories and endpoint comparisons](figures/ungated_duration.png)

The curve panel starts at step 800; initial and step-1 scores remain in each run.
The right panel uses each named reference's NLL as denominator. Red marks apply
only to full controls; narrow references require a strict negative difference.

{resources}

Allocated peak includes GPU corpus cache, training and intervening/final
validation. Time averages 3,190 updates after ten timing-warmup updates, includes
sampling/optimizer/observer overhead, and excludes validation/checkpoint writes.
Optimizer warmup is 320 updates. The sequential laptop-GPU cohort does not
establish paired hardware-speed gains across sessions. Fewer parameters alone
do not establish faster training. Inference is full sequence B16/T128 without KV
cache, ten warmups and three repeats of 30 forwards; it is not generation.

## Unchanged recipe and explicit scope

The [frozen plan](ungated_duration_plan.md) changes only steps to 3,200 and
log_every to 800 from H077's seed-17 recipes. Shared 10% warmup grows to 320,
then cosine decay ends at 0.1 peak. These are fresh schedules, not checkpoint
continuations; their first 800 updates differ from the shorter schedules.

Keep d384/L8/heads6/context128/vocab4096, batch 16, structured groups 8, native
BF16 with FP32 parameters, TF32 off, four CPU threads, name-local initialization
and down residual scale 1/4. AdamW (0.9,0.95), epsilon 1e-8, base decay 0.1 and clip 1
are unchanged. Structured factors retain fan-in LR and parameter decay. Both
narrows retain exactly-once down initialization/LR calibration and product decay.
Whole-block checkpointing is fixed except narrow GELU's block-plus-inner mode.

The unchanged H076 constructor/observers/operation counter call the unchanged
src/core trainer, attention, normalization, data, loss and evaluation. Full models
have 9,437,184 FFN / 15,735,168 total weights; compressed forms have
2,801,664 / 9,099,648, a 70.3125% FFN and 42.1700% total reduction. Approximate
FFN matrix forward FLOPs are twice FFN weights across eight layers per token;
nonlinearities, shuffles, norms, loss, optimizer and other costs are excluded.
All norms, clipping, gradient/activation diagnostics and finite moments are
retained. These checks supply no whole-network gradient lower bound.

The frozen cache is Salesforce/wikitext revision
f776294184f13b8ff2337b3841cf9269a6216d1e, train-only 4096-token BPE, 3,083,650 train
and 322,802 validation tokens. CUDA sampler seed 10017 and exact policy/order are
unchanged. Full validation is 322,688 targets in 158 batches, last nine windows;
113 suffix tokens lie outside complete windows. Official test is not fetched or
scored. This heavily reused development split is not an independent holdout.

Each run presents 6,553,600 training targets, about 2.125 cache token exposures
with replacement, not ordered epochs. Total scientific work is **19,200 updates
and 39,321,600 training targets**. Seven full-validation passes per run produce
**13,552,896 validation exposures**. Worker processes total **{worker_seconds:.2f} s**.

## Independent verification and remaining work

Five isolated checks pass with zero optimizer updates. They verify all six
initial states/counts/calibration, the complete schedule, primary boundaries,
separate diagnostics and a read-only 3,200-batch CUDA sampler reconstruction.
The latter constructs 6,553,600 unscored target elements with zero model forwards;
its state after 800 matches every H077 seed-17 trial.

Independent audit checks all six final checkpoints and AdamW moments at step 3,200,
all 3,200 loss/norm/rate records per run, exact initial weights/NLL/first training
loss against H077, data/source hashes, parameter/operation counts, optimizer
calibration and regenerated final sampler state. Every final BF16 validation NLL
rescores exactly: **1,936,128 additional validation targets and zero updates**.
All primary and descriptive decisions are independently rederived.

See [run metrics](../results/ungated_duration_v1/result.json),
[independent audit](../results/verification/ungated_duration_analysis_v1.json) and
[final preservation audit](../results/verification/ungated_duration_final_v1.json).
All trials finish first attempt, with no scientific retry or active model addition.
All 171 original language/profile runs, 21 H076 and 18 H077 trials, 798 fitting
cells, prior resource pairs/failures and frozen plans remain preserved.

[H051's longer plain failure](long_duration_replication_results.md),
[H077's finite-budget replication](ungated_lm_replication_results.md), the
[local proof limits](ungated_blockshuffle_results.md) and
[remaining published comparators](ungated_comparator_note.md) remain part of
the evidence. Additional long-budget seeds, duration-specific rate sensitivity,
convergence, scale, broader data and comparator/novelty evidence remain open.
A fixed-budget pass does not establish state-of-the-art or arbitrary-data superiority.
"""
Path("research/ungated_duration_results.md").write_text(text, encoding="utf-8")
print(
    json.dumps(
        {"status": "PASS", "verdict": verdict, "primary_pass": primary, "scientific_reruns": 0}
    )
)
