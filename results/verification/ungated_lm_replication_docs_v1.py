"""Update readable current guidance only after the complete H077 result is audited."""

import json
import statistics
from pathlib import Path

ROOT = Path("results/ungated_lm_replication_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


r = read(ROOT / "result.json")
a = read("results/verification/ungated_lm_replication_analysis_v1.json")
assert r["status"] == "complete" and a["status"] == "PASS" and a["all_18_rescores_exact"]
assert read(ROOT / "report_process.json")["status"] == "PASS"
primary = r["replicated_primary_pass"]
summary = r["nll_summary"]
paired = r["paired_candidate_minus_control"]
forms = ("full_swiglu", "full_gelu", "narrow_swiglu", "narrow_gelu", "plain", "gelu_matched")
names = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "narrow_swiglu": "Narrow SwiGLU",
    "narrow_gelu": "Narrow GELU",
    "plain": "Plain BlockShuffle",
    "gelu_matched": "BlockShuffle GELU h3264",
}
means = {}
for f in forms:
    ms = [r["rows"][f"{f}_seed{s}"] for s in (17, 29, 43)]
    means[f] = {
        "mib": statistics.mean(m["peak_allocated_vram_bytes"] / 2**20 for m in ms),
        "update_ms": statistics.mean(m["timed_training_seconds"] * 1000 / 790 for m in ms),
    }
plain_wins = sum(
    r["rows"][f"gelu_matched_seed{s}"]["validation_loss"]
    < r["rows"][f"plain_seed{s}"]["validation_loss"]
    for s in (17, 29, 43)
)
plain_time_change = 100 * (means["gelu_matched"]["update_ms"] / means["plain"]["update_ms"] - 1)
verdict = (
    "passes every seed's primary gates" if primary else "fails the frozen every-seed primary gates"
)
status = (
    "PROMISING; fixed-budget replication passes"
    if primary
    else "REJECTED at the fixed replicated recipe"
)
next_step = (
    "Only a separately frozen duration/convergence comparison is earned."
    if primary
    else "No automatic longer run, extra rate, width, activation or optimizer repair is earned."
)
lead = f"""**The research goal remains unmet. Matched-budget BlockShuffle GELU {verdict}.**
The [H077 replication](research/ungated_lm_replication_results.md) completes 18
fresh 800-step trials across seeds 17/29/43. Matched mean validation NLL is
**{summary["gelu_matched"]["mean"]:.6f} +/- {summary["gelu_matched"]["sample_sd"]:.6f}**
(sample SD), with **70.3125% fewer FFN weights**. {next_step}

Relative mean NLL is **{paired["full_gelu"]["relative_mean_nll_percent"]:+.3f}%** versus full GELU,
**{paired["full_swiglu"]["relative_mean_nll_percent"]:+.3f}%** versus full SwiGLU and
**{paired["plain"]["relative_mean_nll_percent"]:+.3f}%** versus plain BlockShuffle; negative is better.
Mean allocated peak is **{means["gelu_matched"]["mib"]:.1f} MiB** and mean update time
**{means["gelu_matched"]["update_ms"]:.1f} ms**, versus full GELU's
**{means["full_gelu"]["mib"]:.1f} MiB / {means["full_gelu"]["update_ms"]:.1f} ms**.
Matched wins against plain in **{plain_wins}/3** seeds; the small mean NLL gain
is not a dependable quality improvement. Mean update time changes by
**{plain_time_change:+.2f}%** relative to plain (negative is faster).
These are fixed selected rates on reused development data, not convergence,
rate-optimal or state-of-the-art evidence. The failed same-width GELU recipe stays closed.

"""
p = Path("README.md")
s = p.read_text(encoding="utf-8")
start = s.index("## Retained code")
s = "# Parameter-efficient FFN research\n\n" + lead + s[start:]
s = s.replace(
    "GELU has only short-screen language evidence",
    "GELU has the fixed-budget replication verdict above",
)
s = s.replace(
    "[Language screen](research/ungated_lm_screen_results.md): every rate, selected\n  control, resource measurement and gate. All 21 checkpoint NLLs rescore exactly.",
    "[Longer replication](research/ungated_lm_replication_results.md): every seed,\n  paired comparison and gate. All 18 final checkpoint NLLs rescore exactly.\n- [Earlier language screen](research/ungated_lm_screen_results.md): all 21 rates\n  that fixed the replication recipes; all saved NLLs independently verified.",
)
s = s.replace(
    "with **21** new language trials kept separately.",
    "with **39** new language trials kept separately: 21 H076 and 18 H077.",
)
s = s.replace("The new short GELU result", "The new fixed-budget GELU replication")
s = s.replace(
    "[screen artifacts](results/ungated_lm_screen_v1/)",
    "[replication artifacts](results/ungated_lm_replication_v1/)",
)
s = s.replace(
    "[frozen GELU screen plan](research/ungated_lm_screen_plan.md)",
    "[frozen GELU replication plan](research/ungated_lm_replication_plan.md)",
)
p.write_text(s, encoding="utf-8")
table = [
    "| Form | Fixed peak LR | Mean NLL +/- sample SD | Mean peak MiB | Mean update ms |",
    "|---|---:|---:|---:|---:|",
]
for f in forms:
    rate = r["rows"][f"{f}_seed17"]["training"]["learning_rate"]
    table.append(
        f"| {names[f]} | {rate:.4f} | {summary[f]['mean']:.6f} +/- {summary[f]['sample_sd']:.6f} | {means[f]['mib']:.3f} | {means[f]['update_ms']:.3f} |"
    )
table = "\n".join(table)
current = f"""# Current research state

**The gold target remains unmet. H077 {verdict}.** The current model experiment
is matched-budget BlockShuffle GELU h3264, with 2,801,664 FFN / 9,099,648 total
weights: 70.3125% FFN and 42.1700% total reduction. {next_step}

## Latest complete comparison

The [H077 report](ungated_lm_replication_results.md) contains all 18 fresh
800-step trials at seeds 17/29/43. Rates were selected earlier with equal H076
search budgets and remain fixed here. Every seed must pass both full-control
1% allowances, strictly beat BOTH narrows and pass actual memory. Means cannot
rescue a failed seed.

{table}

Matched's relative mean NLL differences are {paired["full_gelu"]["relative_mean_nll_percent"]:+.4f}%
versus full GELU, {paired["full_swiglu"]["relative_mean_nll_percent"]:+.4f}% versus full SwiGLU,
{paired["narrow_gelu"]["relative_mean_nll_percent"]:+.4f}% versus narrow GELU,
{paired["narrow_swiglu"]["relative_mean_nll_percent"]:+.4f}% versus narrow SwiGLU and
{paired["plain"]["relative_mean_nll_percent"]:+.4f}% versus plain. Negative means lower NLL.
Matched beats plain in {plain_wins}/3 seeds, with {plain_time_change:+.2f}% mean update time
relative to plain. Its small mean NLL gain is not a consistent quality advantage.
The report retains per-seed decisions, paired sample SD and exploratory intervals;
these three-seed summaries do not establish universal superiority.

The operational late-plateau diagnostic passes {sum(r["operational_late_plateau"].values())}/18 cells.
No convergence claim follows. The schedule stretches warmup to 80 updates,
so these are fresh longer schedules rather than short-checkpoint continuations.
Actual memory includes the GPU corpus cache and validation; time is the mean
of 790 post-timing-warmup updates. Full-sequence inference is not autoregressive
serving. The reused WikiText-2 development split is not an independent holdout.

## Retained repository scope

The active tree has three model folders, six registered variants and nine recipes.
Experimental GELU uses the existing ungated operator with an isolated adapter;
it is not a duplicated model folder or new registered variant.

| Component | Role and limitation |
|---|---|
| [dense_ffn](../src/dense_ffn/README.md) | Four full/narrow GELU/SwiGLU controls for fair quality and resource comparisons |
| [blockshuffle_ffn](../src/blockshuffle_ffn/README.md) | Structured operator and plain SwiGLU reference; supports the isolated GELU experiment |
| [rational_blockshuffle_ffn](../src/rational_blockshuffle_ffn/README.md) | Small learned-activation reference; one-seed quality potential, excessive memory, tested repairs closed |

All 171 original language/profile runs remain, with 21 H076 and 18 H077 trials
isolated beside their protocols. Discarded architectures and obsolete drivers
are in the [archive](archive/README.md). Historical current-state pages are saved
in the per-round before-documents archives; detailed decisions stay in the
[complete ledger](idea_bank.md) and linked reports.

## Findings that still constrain the next step

| Evidence | Decision that remains in force |
|---|---|
| [H051 longer plain replication](long_duration_replication_results.md) | 3,200-step mean NLL 4.147443 is 1.256% above full SwiGLU; longer-training goal fails |
| [Affine matched-rate correction](affine_rate_replication_results.md) | Only 0.101% mean benefit, wins in 2/3 seeds; earlier larger claim was rate-confounded |
| [Native rational activation](learnable_activation_results.md) | 200-step NLL 5.898003, but 883.17 MiB peak; learned-shape reset barely changes quality |
| [H063 native](native_recompute_results.md) / [H066 staged](staged_resource_results.md) recomputation | Numerical fidelity passes, rational memory gates fail; no automatic partition or language repeat |
| [H061 additive low-rank screen](additive_block_lowrank_screen_results.md) | Quality gates fail; implementation retired in [H062](additive_retirement_results.md) |
| [H068 conditioned activation fitting](token_activation_fit_results.md) | Static/dynamic gains miss the frozen 2% gate; richer routing receives no automatic allocation |
| [H069 factor balance](factor_balance_results.md) | Exact local preservation but insufficient measured imbalance; no optimizer experiment earned |
| [H071 rotated-shuffle fitting](rotated_shuffle_fit_results.md) | Both tested rotations fail learning gates despite a valid local rank witness |
| [H072 local GELU analysis](ungated_blockshuffle_results.md) | Linear odd-component restriction and scoped population error floor; no whole-network or learning guarantee |
| [H073 fitting](ungated_fit_results.md), [H075 resources](ungated_resource_recovery_results.md) | Qualified the language screen; synthetic gains and timings are not corpus or convergence proof |
| [H076 language screen](ungated_lm_screen_results.md) | Same-width GELU fails; matched passes with only 0.1167% gain over selected narrow GELU at 200 steps |

Bezier/quadratic, single-factor grouped, coupled/shared, paired-feature,
parallel/headwise/overcomplete, shifted/affine and failed rational compiler branches
remain archived. Local proofs survive within their stated assumptions; failures
do not justify automatic tuning or relabeling known components as novel.

## What remains before the research goal can be accepted

1. Follow the H077 decision above with a separately frozen duration/convergence
   hypothesis if earned. Preserve full and both calibrated narrow controls,
   equal tuning effort, data policy and actual memory/runtime accounting.
2. Establish scale and broader-data results with multiple seeds. Reused development
   validation and fixed short-selected rates cannot prove those requirements.
3. Reproduce strong published structured comparators under the same task scope.
   The [primary-source comparator note](ungated_comparator_note.md) identifies
   BLAST and BTT-MoE distinctions and an unallocated count-compatible BLAST budget.
4. Any further learnable activation needs a distinct justified mechanism, correct
   derivatives/initialization, practical memory and a measured gain over the
   strongest relevant control. See the [activation guide](learnable_activation_domain.md).
5. Preserve the distinction between local representation proofs, measured learning,
   training speed and full-sequence inference. No arbitrary-data or global-gradient
   guarantee, novelty priority or automatic publication is established.

## Verification

Five H077 integration checks pass with zero optimizer updates. All 18 initial
states reconstruct exactly, sampler states and full validation order agree,
and every final checkpoint's BF16 NLL rescores exactly. Independent checks cover
all weights/moments, all 800 step records, counts, calibration, per-seed gates and
paired statistics. The [final audit](../results/verification/ungated_lm_replication_final_v1.json)
preserves 109 scientific sources, 66 frozen plans, 798 earlier fitting cells,
21 H074/H075 resource checkpoint pairs and all historical language evidence.

The previously passing 108-test active suite and its source are unchanged;
these isolated checks do not replace or inflate that count. A pre-implementation
tool observation interruption is preserved separately from successful scientific
execution. No allocated training trial was retried.
"""
Path("research/CURRENT_STATE.md").write_text(current, encoding="utf-8")
p = Path("research/idea_bank.md")
s = p.read_text(encoding="utf-8")
assert "## H077 -" in s
s = (
    s[: s.index("## H077 -")]
    + f"""## H077 - Longer three-seed GELU replication ({status})

The [complete comparison](ungated_lm_replication_results.md) finishes all 18
fresh 800-step trials across seeds 17/29/43, using six fixed H076-selected
recipes. Matched-budget GELU {verdict}. {next_step}
Its mean NLL is {summary["gelu_matched"]["mean"]:.9f} +/- {summary["gelu_matched"]["sample_sd"]:.9f}
(sample SD), with 2,801,664 FFN / 9,099,648 total weights, 70.3125% FFN reduction.

Relative mean NLL is {paired["full_gelu"]["relative_mean_nll_percent"]:+.4f}% versus full GELU,
{paired["full_swiglu"]["relative_mean_nll_percent"]:+.4f}% versus full SwiGLU,
{paired["narrow_gelu"]["relative_mean_nll_percent"]:+.4f}% versus narrow GELU,
{paired["narrow_swiglu"]["relative_mean_nll_percent"]:+.4f}% versus narrow SwiGLU and
{paired["plain"]["relative_mean_nll_percent"]:+.4f}% versus plain. Negative favors matched.
Matched beats plain in only {plain_wins}/3 seeds; the mean NLL benefit does not
establish a dependable plain-quality improvement. Mean update time changes by
{plain_time_change:+.2f}% versus plain.
All per-seed gates, paired differences/sample SD and exploratory df=2 intervals
are in the report. A passing mean cannot rescue a failed seed. The 0.2% narrow
margin is descriptive. Late-plateau diagnostics pass {sum(r["operational_late_plateau"].values())}/18;
no convergence or statistical-significance claim follows.

Matched mean allocated peak is {means["gelu_matched"]["mib"]:.3f} MiB, mean timed update
{means["gelu_matched"]["update_ms"]:.3f} ms. Full GELU uses {means["full_gelu"]["mib"]:.3f} MiB /
{means["full_gelu"]["update_ms"]:.3f} ms. Actual corpus memory, full-sequence inference,
all clipping and diagnostics are retained; fewer weights do not imply faster
training or a global gradient guarantee.

Only steps/logging/seed change from H076's selected recipes. Fresh 800-step
schedules have 80-update warmup; initial weights, native execution, structured LR,
narrow product decay and all shared trainer/data/evaluation sources are unchanged.
Five integration checks pass with zero optimizer updates. All 18 final BF16 NLLs
rescore exactly; initialization, moments, logs, sampler and source/data hashes
are independently verified. Work is 14,400 updates / 29,491,200 training targets,
40,658,688 original validation exposures and 5,808,384 audit validation targets.
Read-only sampler checks construct 4,915,200 unscored targets with no model forwards.

All scientific trials finish first attempt. The earlier implementation-write
tool observation interruption left no file or live writer; it is preserved and
is not a scientific retry. The current-state page is shortened for readability;
its full prior version survives in before_docs.zip. The active three-folder/
six-variant/nine-recipe tree and all earlier evidence remain unchanged.

A separate [comparator note](ungated_comparator_note.md) reads BLAST, StructuredFFN
and BTT-MoE primary sources and derives an unallocated equal-parameter BLAST
control. It changes no H077 hypothesis or decision. The reused development split,
short-selected rates, missing convergence/scaling/broader-data/comparator evidence
and known-component novelty limits prevent completion of the research goal.
"""
)
p.write_text(s, encoding="utf-8")
print(json.dumps({"status": "PASS", "updated_docs": 3, "replicated_primary_pass": primary}))
