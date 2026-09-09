"""Report the fixed H077 replication after independent endpoint verification."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/ungated_lm_replication_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


r = read(ROOT / "result.json")
a = read("results/verification/ungated_lm_replication_analysis_v1.json")
assert a["status"] == "PASS" and a["all_18_rescores_exact"]
assert a["independently_rederived_gates"] == r["per_seed_gates"]
forms = ("full_swiglu", "full_gelu", "narrow_swiglu", "narrow_gelu", "plain", "gelu_matched")
seeds = (17, 29, 43)
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
rows = r["rows"]
primary = r["replicated_primary_pass"]
verdict = (
    "PASSES every seed's primary gates" if primary else "FAILS the frozen every-seed replication"
)
next_step = (
    "This fixed recipe earns a separately frozen duration/convergence study. Scale, broader data and strong published controls remain open."
    if primary
    else "The fixed replicated recipe earns no automatic longer allocation, rate, width, activation or optimizer repair. Preserve the short-screen result as a single-seed finding."
)
aggregate = [
    "| Form | Frozen peak LR | Mean NLL +/- sample SD | FFN weights | Total weights | Mean peak MiB | Mean timed update ms |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
allrows = [
    "| Form | Seed | NLL | Peak MiB | Mean update ms | Training targets/s | Inference targets/s | Clipped | Final-200 NLL change |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
histories = {}
for f in forms:
    ms = [rows[f"{f}_seed{s}"] for s in seeds]
    n = r["nll_summary"][f]
    aggregate.append(
        f"| {names[f]} | {ms[0]['training']['learning_rate']:.4f} | {n['mean']:.9f} +/- {n['sample_sd']:.9f} | {ms[0]['ffn_parameters']:,} | {ms[0]['total_parameters']:,} | {statistics.mean(m['peak_allocated_vram_bytes'] / 2**20 for m in ms):.3f} | {statistics.mean(m['timed_training_seconds'] * 1000 / 790 for m in ms):.3f} |"
    )
    for seed, m in zip(seeds, ms):
        allrows.append(
            f"| {names[f]} | {seed} | {m['validation_loss']:.9f} | {m['peak_allocated_vram_bytes'] / 2**20:.3f} | {m['timed_training_seconds'] * 1000 / 790:.3f} | {m['training_tokens_per_second']:.0f} | {m['inference_tokens_per_second']:.0f} | {100 * m['clipped_step_fraction']:.2f}% | {m['late_nll_change_percent']:+.4f}% |"
        )
        histories[f"{f}_seed{seed}"] = list(
            map(
                json.loads,
                (ROOT / "runs" / f"{f}_seed{seed}" / "history.jsonl").read_text().splitlines(),
            )
        )
comparisons = [
    "| Reference | Mean paired NLL difference | Paired sample SD | Exploratory 95% interval | Relative mean NLL difference | Seed 17 | Seed 29 | Seed 43 |",
    "|---|---:|---:|---|---:|---:|---:|---:|",
]
for f, v in r["paired_candidate_minus_control"].items():
    ci = v["exploratory_ci95"]
    per = v["per_seed_relative_nll_percent"]
    comparisons.append(
        f"| {names[f]} | {v['mean']:+.9f} | {v['sample_sd']:.9f} | [{ci[0]:+.9f}, {ci[1]:+.9f}] | {v['relative_mean_nll_percent']:+.4f}% | {per['17']:+.4f}% | {per['29']:+.4f}% | {per['43']:+.4f}% |"
    )
gates = ["| Primary gate | Seed 17 | Seed 29 | Seed 43 |", "|---|---|---|---|"]
for g in r["per_seed_gates"]["17"]:
    gates.append(
        "| "
        + g.replace("_", " ")
        + " | "
        + " | ".join("PASS" if r["per_seed_gates"][str(s)][g] else "FAIL" for s in seeds)
        + " |"
    )
margins = [
    "| Descriptive >=0.2% gain over narrow | Seed 17 | Seed 29 | Seed 43 |",
    "|---|---|---|---|",
]
for f in ("narrow_swiglu", "narrow_gelu"):
    margins.append(
        "| "
        + names[f]
        + " | "
        + " | ".join(
            "YES" if r["descriptive_point_two_percent_narrow_margin"][str(s)][f] else "NO"
            for s in seeds
        )
        + " |"
    )
aggregate, allrows, comparisons, gates, margins = map(
    "\n".join, (aggregate, allrows, comparisons, gates, margins)
)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
for f in forms:
    values = [[v["validation_loss"] for v in histories[f"{f}_seed{s}"][1:]] for s in seeds]
    means = [statistics.mean(v) for v in zip(*values)]
    sds = [statistics.stdev(v) for v in zip(*values)]
    axes[0].plot([200, 400, 600, 800], means, color=colors[f], marker="o", label=names[f])
    axes[0].fill_between(
        [200, 400, 600, 800],
        [m - s for m, s in zip(means, sds)],
        [m + s for m, s in zip(means, sds)],
        color=colors[f],
        alpha=0.13,
    )
axes[0].set_xlabel("Optimizer update (fresh 800-step schedules)")
axes[0].set_ylabel("Validation NLL; mean +/- sample SD")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.15)
refs = [f for f in forms if f != "gelu_matched"]
for i, f in enumerate(refs):
    v = r["paired_candidate_minus_control"][f]
    ci = v["exploratory_ci95"]
    axes[1].errorbar(
        v["mean"],
        i,
        xerr=[[v["mean"] - ci[0]], [ci[1] - v["mean"]]],
        fmt="o",
        color=colors[f],
        capsize=4,
    )
    for seed in seeds:
        difference = (
            rows[f"gelu_matched_seed{seed}"]["validation_loss"]
            - rows[f"{f}_seed{seed}"]["validation_loss"]
        )
        axes[1].scatter(difference, i + 0.15, s=15, color=colors[f], alpha=0.65)
    if f.startswith("full_"):
        axes[1].scatter(0.01 * r["nll_summary"][f]["mean"], i, marker="|", s=180, color="#aa3333")
axes[1].axvline(0, color="#888888", linewidth=1)
axes[1].set_yticks(range(len(refs)), [names[f] for f in refs])
axes[1].invert_yaxis()
axes[1].set_xlabel(
    "Matched GELU - reference NLL (negative is better)\nBars: exploratory 95% paired intervals; small dots: seeds"
)
axes[1].set_title(
    "Red marks: 1% of mean full-control NLL\nPrimary gates are evaluated separately for each seed",
    fontsize=10,
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "H077: WikiText-2, fixed selected rates, seeds 17 / 29 / 43\nReused development validation; no official test or convergence claim",
    fontsize=12,
)
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/ungated_lm_replication.{ext}", dpi=160)
plt.close(fig)
worker_seconds = sum(read(ROOT / "processes" / (c + ".json"))["elapsed_seconds"] for c in rows)
plateau_passes = sum(r["operational_late_plateau"].values())
matched = r["nll_summary"]["gelu_matched"]
plain_wins = sum(
    rows[f"gelu_matched_seed{s}"]["validation_loss"] < rows[f"plain_seed{s}"]["validation_loss"]
    for s in seeds
)
plain_gap = r["paired_candidate_minus_control"]["plain"]["relative_mean_nll_percent"]
plain_ci = r["paired_candidate_minus_control"]["plain"]["exploratory_ci95"]
text = f"""# H077 - Longer three-seed BlockShuffle GELU replication

**Matched-budget BlockShuffle GELU {verdict}.** All 18 fresh trials complete at
800 updates each. Matched mean NLL is **{matched["mean"]:.9f} +/- {matched["sample_sd"]:.9f}**
(sample SD), with **70.3125% fewer FFN weights**. {next_step}
The full research goal remains unachieved. A replicated fixed-budget comparison
alone establishes neither convergence nor a new architecture.

Against plain BlockShuffle, matched GELU wins **{plain_wins}/3** seeds, with a
**{plain_gap:+.4f}%** relative mean NLL difference. Its exploratory paired interval
is [{plain_ci[0]:+.6f}, {plain_ci[1]:+.6f}]. These plain comparisons must accompany
the full/narrow gate result; a small mean gain does not establish dependable
quality improvement over plain. Resource and timing tradeoffs remain separate.

## Complete cohort and primary decisions

Each form keeps its minimum-final-NLL rate from H076's equal three-rate budget.
No new rate or best-intermediate checkpoint is selected from these results.
Resource means below average the same three seed endpoints; every seed is listed
separately below. A passing mean cannot rescue a failed seed.

{aggregate}

{gates}

Relative differences below are matched GELU minus the named reference; negative
values favor matched GELU. The full-control allowance is 1% for each seed and
EACH full reference. BOTH calibrated narrow references must be beaten strictly.
The paired intervals use three seeds, sample SD and the df=2 critical value
sqrt(2 * 0.95^2 / (1 - 0.95^2)), approximately 4.30265. They depend on independent,
approximately normal paired differences and are exploratory small-sample summaries,
not a claim of significance or universal superiority.

{comparisons}

{margins}

The 0.2% narrow margin remains descriptive and changes no primary gate. The
operational late-plateau diagnostic passes **{plateau_passes}/18** cells: absolute
final-200 NLL change <=0.2% and final <=0.2% above the best recorded endpoint.
Even a passing plateau diagnostic would not prove optimal convergence.

![Mean trajectories and paired differences](figures/ungated_lm_replication.png)

The curve panel displays logged steps 200 through 800, with mean +/- sample SD.
Initial and step-1 evaluations are retained in every run. The difference panel's
red marks show 1% of the MEAN full-control NLL for context; the primary decisions
use each seed's own full-control values, as in the gate table.

## Every allocated endpoint

{allrows}

These timings include sampling, optimizer and scalar-observer overhead across
790 post-timing-warmup updates. Validation and checkpoint writes are excluded.
The timing warmup is ten updates; optimizer warmup is 80 updates. All runs share
one laptop GPU and are sequential, with order reversed/rotated across seeds;
thermal and session variation limit timing interpretation. Fewer weights do not
imply faster training. Inference is full sequence without KV cache, not generation.

## Frozen methods and explicit limitations

The [plan](ungated_lm_replication_plan.md) fixes six forms, seeds 17/29/43, batch 16,
context 128, d384/L8/heads6/vocab4096, structured groups 8, native BF16 with FP32
parameters, TF32 off and four CPU threads. Every trial starts fresh. Only steps,
log interval and seed differ from the selected H076 recipe. The shared 10%
warmup grows from 20 to 80 and cosine decay stretches to 800, ending at 0.1 peak.
The first 200 updates therefore differ from the old short schedule.

Name-local initialization, residual down scale 1/4, AdamW (0.9,0.95), eps 1e-8,
base decay 0.1 and clip 1 are unchanged. Structured factors retain fan-in LR
correction and parameter decay. Both narrows keep exactly-once down initialization/
LR calibration and product decay. Frozen execution is whole-block checkpointing
except narrow GELU's block-plus-inner mode. The unchanged H076 observers and
operation counter call the shared src/core trainer without changing loss or
updates. Active registered code/configurations remain unchanged.

Each full model has 9,437,184 FFN / 15,735,168 total weights; all four compressed
forms have 2,801,664 / 9,099,648, giving 70.3125% FFN and 42.1700% total reduction.
FFN matrix forward FLOPs are twice FFN weights across eight layers per token.
This excludes nonlinearities, shuffles, norms, loss, optimizer and non-matrix work.
Allocated peak includes GPU corpus cache, training and intervening/final validation.
Full-sequence inference uses B16/T128, ten warmups and three repeats of 30 forwards.
Finite parameter/moment checks, preclip norms, clipping and per-layer activation/
gradient diagnostics are retained; they do not prove a whole-network gradient bound.

The immutable cache is Salesforce/wikitext revision
f776294184f13b8ff2337b3841cf9269a6216d1e with train-only 4096-token BPE:
3,083,650 training and 322,802 validation tokens. CUDA sampler seeds are 10017,
10029 and 10043. Full validation covers 322,688 targets in 158 contiguous batches,
last nine windows; 113 suffix tokens lie outside complete windows. All batch
orders and final sampler states are audited. No official test is fetched or scored.
This repeatedly reused development split is not an unbiased holdout evaluation.
Seed 17 influenced H076's rate choice; the two additional training seeds do not
supply new validation data or duration-specific rate optimization.

The [H076 short screen](ungated_lm_screen_results.md), its rejected same-width
recipe, [H051 longer plain failure](long_duration_replication_results.md) and
[local GELU proof and limitations](ungated_blockshuffle_results.md) remain intact.
Conventional GELU and the existing structured factors establish no novelty claim.
A separate [primary-source comparator note](ungated_comparator_note.md) records
BLAST and BTT-MoE scope differences and a count-compatible, unallocated BLAST
control. It changes none of this replication's frozen decisions.

## Execution and independent audit

Five isolated checks pass with zero optimizer updates. They verify all 18 initial
model states, counts, calibration, fixed rates/schedule, every-seed decisions and
statistics. Three read-only 800-batch sampler reconstructions construct 4,915,200
unscored target elements, with zero model forwards; seed 17's 200-step state matches
H076. The prior 108-test active suite and H076 fidelity checks are unchanged.

All 18 scientific trials complete first attempt: **14,400 updates and 29,491,200
training targets**. Seven full-validation evaluations per trial total **40,658,688
validation exposures**. Worker processes total **{worker_seconds:.2f} seconds**.
An earlier implementation-write tool observation was interrupted before a file
or live writer was found; that record precedes all scientific launches and is
preserved. It caused no scientific retry or source change after freezing.

Independent analysis reloads every final checkpoint, verifies finite AdamW moments
at step 800, all 800 loss/norm/rate records, actual counts, initial weights, optimizer
calibration, data/source hashes and sampler states. All seed-17 initial NLLs and
first pre-update losses match H076 exactly. All 18 final BF16 validation NLLs rescore
exactly: **5,808,384 additional validation targets, zero updates**. All per-seed
gates, paired statistics and late diagnostics are independently rederived. This
verifies saved scores without supplying independent generalization data.

See [run metrics](../results/ungated_lm_replication_v1/result.json),
[independent audit](../results/verification/ungated_lm_replication_analysis_v1.json)
and [final preservation audit](../results/verification/ungated_lm_replication_final_v1.json).
All 171 original language/profile runs, 21 H076 language trials, 798 fitting
checkpoints, prior resource pairs/failures and frozen plans remain preserved.
"""
Path("research/ungated_lm_replication_results.md").write_text(text, encoding="utf-8")
print(
    json.dumps(
        {
            "status": "PASS",
            "verdict": verdict,
            "replicated_primary_pass": primary,
            "scientific_reruns": 0,
        }
    )
)
