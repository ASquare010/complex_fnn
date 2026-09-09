"""Update the three navigation documents from independently audited H078 results."""

import json
from pathlib import Path

ROOT = Path("results/ungated_duration_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


r = read(ROOT / "result.json")
a = read("results/verification/ungated_duration_analysis_v1.json")
assert r["status"] == "complete" and a["status"] == "PASS"
assert a["all_6_rescores_exact"] and a["independently_rederived_gates"] == r["gates"]
assert read(ROOT / "report_process.json")["status"] == "PASS"
assert not (ROOT / "support.zip").exists()
forms = ("full_swiglu", "full_gelu", "narrow_swiglu", "narrow_gelu", "plain", "gelu_matched")
names = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "narrow_swiglu": "Narrow SwiGLU",
    "narrow_gelu": "Narrow GELU",
    "plain": "BlockShuffle SwiGLU",
    "gelu_matched": "BlockShuffle GELU h3264",
}
rows = {f: r["rows"][f"{f}_seed17"] for f in forms}
m = rows["gelu_matched"]
primary = r["primary_pass"]
verdict = "passes" if primary else "fails"
state = "PROMISING; single-seed duration gates pass" if primary else "REJECTED; fixed duration recipe fails"
next_step = (
    "Only a separately frozen replication at additional long-budget seeds is earned."
    if primary else
    "This fixed duration recipe is closed; no automatic rate, width, activation or optimizer repair is earned."
)
failed = [key.replace("_", " ") for key, value in r["gates"].items() if not value]
gate_sentence = "All frozen quality, compression and memory gates pass." if primary else "Failed gates: " + "; ".join(failed) + "."
peak = m["peak_allocated_vram_bytes"] / 2**20
ms = m["timed_training_seconds"] * 1000 / 3190
time_gap = 100 * (m["timed_training_seconds"] / rows["plain"]["timed_training_seconds"] - 1)
gaps = {f: v["relative_nll_percent"] for f, v in r["comparisons"].items()}
plateau = sum(r["operational_late_plateau"].values())
table = [
    "| Form | Fixed peak LR | Final NLL | Peak MiB | Mean update ms |",
    "|---|---:|---:|---:|---:|",
]
for f, row in rows.items():
    table.append(
        f"| {names[f]} | {row['training']['learning_rate']:.4f} | {row['validation_loss']:.6f} | {row['peak_allocated_vram_bytes'] / 2**20:.3f} | {row['timed_training_seconds'] * 1000 / 3190:.3f} |"
    )
table = "\n".join(table)

readme = Path("README.md")
old = readme.read_text(encoding="utf-8")
assert "## Retained code" in old and "is running six" in old
prefix = f"""# Parameter-efficient FFN research

**The research goal remains unmet. Matched-budget BlockShuffle GELU {verdict}
the fixed 3,200-step test.** The [H078 comparison](research/ungated_duration_results.md)
completes all six fresh seed-17 trials. The candidate reaches validation NLL
**{m['validation_loss']:.6f}**, with **70.3125% fewer FFN weights** and
**42.1700% fewer total model weights**. {next_step}

Candidate NLL differs by **{gaps['full_gelu']:+.3f}%** versus full GELU,
**{gaps['full_swiglu']:+.3f}%** versus full SwiGLU,
**{gaps['narrow_gelu']:+.3f}%** versus narrow GELU and
**{gaps['narrow_swiglu']:+.3f}%** versus narrow SwiGLU. Negative is better.
{gate_sentence}

Its allocated peak is **{peak:.1f} MiB** and mean update time **{ms:.1f} ms**.
Versus plain BlockShuffle, candidate NLL changes by **{gaps['plain']:+.3f}%**
and mean update time by **{time_gap:+.2f}%**. See the report for every control's
measured time and memory. Fewer weights do not establish faster dense-relative training.

This is one seed with rates fixed from the earlier short search. Late-plateau
diagnostics pass for **{plateau}/6** models; no convergence or replicated long-budget claim
follows. [H077](research/ungated_lm_replication_results.md) passes all three seeds
at 800 steps, but its small mean gain over plain wins only 1/3 seeds. Reused
development data and known building blocks do not establish state-of-the-art or
novelty. The failed same-width GELU recipe stays closed.

"""
new = prefix + "## Retained code" + old.split("## Retained code", 1)[1]
# Preserve the retained-code/run sections while pointing navigation at the latest experiment.
new = new.replace(
    "GELU has the fixed-budget replication verdict above",
    "GELU has the fixed-duration verdict above",
)
new = new.replace(
    "configuration is isolated in the [replication artifacts](results/ungated_lm_replication_v1/)",
    "configuration is isolated in the [duration artifacts](results/ungated_duration_v1/)",
)
new = new.replace(
    "- [Longer replication](research/ungated_lm_replication_results.md): every seed,",
    "- [Duration comparison](research/ungated_duration_results.md): six fresh 3,200-step\n  trials, every endpoint, memory/time and late-loss changes. All six NLLs rescore exactly.\n- [800-step replication](research/ungated_lm_replication_results.md): every seed,",
)
new = new.replace(
    "prior language/profile runs remain, with **39** new language trials kept separately: 21 H076 and 18 H077.",
    "prior language/profile runs remain, with **45** new language trials kept separately: 21 H076, 18 H077 and six H078.",
)
readme.write_text(new, encoding="utf-8")

current = Path("research/CURRENT_STATE.md")
old = current.read_text(encoding="utf-8")
assert "## Duration study in progress" in old
middle = old.split("## Retained repository scope", 1)[1].split("## What remains before the research goal can be accepted", 1)[0]
middle = middle.replace(
    "All 171 original language/profile runs remain, with 21 H076 and 18 H077 trials",
    "All 171 original language/profile runs remain, with 21 H076, 18 H077 and six H078 trials",
)
header = f"""# Current research state

**The gold target remains unmet. H078 {verdict} its single-seed duration gates.**
The current model experiment is matched-budget BlockShuffle GELU h3264, with
2,801,664 FFN / 9,099,648 total weights: 70.3125% FFN and 42.1700% total reduction.
{next_step}

## Latest complete comparison

The [H078 report](ungated_duration_results.md) contains all six fresh seed-17
trials at 3,200 steps. Models retain H077's fixed recipes, with rates originally
selected under H076's equal three-rate short search. Only steps and logging change;
warmup grows to 320. These are fresh schedules, not checkpoint continuations.

{table}

Matched's relative NLL differences are {gaps['full_gelu']:+.4f}% versus full GELU,
{gaps['full_swiglu']:+.4f}% versus full SwiGLU,
{gaps['narrow_gelu']:+.4f}% versus narrow GELU and
{gaps['narrow_swiglu']:+.4f}% versus narrow SwiGLU. Negative favors matched.
{gate_sentence} Each full-control allowance is 1%; BOTH narrows must be beaten strictly.
The separate 0.2% narrow margins remain descriptive.

Versus plain, candidate NLL changes by {gaps['plain']:+.4f}% and mean update time
by {time_gap:+.2f}%. These are one-seed, sequential laptop-GPU measurements.
No confidence interval or replicated long-duration conclusion follows.
Late-plateau diagnostics pass {plateau}/6 models; the report gives each final-800
loss change. Plateau screening does not establish optimal convergence.

Peak allocation includes the CUDA corpus cache and validation. Time averages
3,190 updates after ten timing-warmup updates; optimizer warmup is separately
320. Full-sequence inference is not autoregressive serving. Heavily reused
WikiText-2 development validation is not an independent holdout.

The [H077 three-seed comparison](ungated_lm_replication_results.md) remains
shorter-budget evidence: all 18 fresh 800-step trials pass each seed's primary
gates for matched GELU, mean NLL 4.834511 +/- 0.013382 (sample SD), but it beats
plain only in 1/3 seeds with a 0.0959% mean gain. None of its 18 plateau screens pass.

## Retained repository scope"""
remaining = f"""## What remains before the research goal can be accepted

1. {next_step} Keep both full and both calibrated narrow controls, equal tuning
   effort, data policy and actual memory/runtime accounting. Duration-specific
   rate sensitivity and convergence remain unestablished.
2. Establish scale and broader-data results with multiple seeds. Reused development
   validation and fixed short-selected rates cannot prove those requirements.
3. Reproduce strong published structured comparators under the same task scope.
   The [primary-source comparator note](ungated_comparator_note.md) identifies
   BLAST and BTT-MoE distinctions and an unallocated count-compatible BLAST budget.
4. Further learnable activation work needs a distinct justified mechanism, correct
   derivatives/initialization, practical memory and measured gain over the strongest
   relevant control. See the [activation guide](learnable_activation_domain.md).
5. Local representation proofs, measured learning, training speed and full-sequence
   inference remain separate claims. No arbitrary-data or global-gradient guarantee,
   novelty priority or automatic publication is established.

## Verification

Five H078 integration checks pass with zero optimizer updates/model forwards.
All six initial states reconstruct exactly, initial NLL/first training loss match
H077, sampler states and full validation order agree, and every final checkpoint's
BF16 NLL rescores exactly. Independent checks cover finite weights/moments, all
3,200 step records per run, counts, calibration and primary/descriptive decisions.

The [final audit](../results/verification/ungated_duration_final_v1.json) preserves
112 scientific sources, 67 frozen plans, 798 earlier fitting cells, 21 H074/H075
resource checkpoint pairs and all historical language evidence. H078 uses
19,200 updates / 39,321,600 training targets; independent rescoring adds
1,936,128 validation targets and zero updates. All scientific trials finish on
the first attempt. The previously passing 108-test active suite and its source
are unchanged; isolated integration checks do not inflate that count.
"""
current.write_text(header + middle + remaining, encoding="utf-8")

ledger = Path("research/idea_bank.md")
old = ledger.read_text(encoding="utf-8")
marker = "## H078 - Fixed-recipe 3,200-step GELU duration test (EXPERIMENTING)"
assert old.count(marker) == 1
entry = f"""## H078 - Fixed-recipe 3,200-step GELU duration test ({state})

The [complete report](ungated_duration_results.md) finishes all six fresh seed-17
trials at 3,200 updates. Matched GELU {verdict} the frozen primary gates, with NLL
{m['validation_loss']:.9f}, 2,801,664 FFN / 9,099,648 total weights, 70.3125% FFN
and 42.1700% total reduction. {next_step}

{gate_sentence} Candidate relative NLL differences are
{gaps['full_gelu']:+.4f}% versus full GELU, {gaps['full_swiglu']:+.4f}% versus full SwiGLU,
{gaps['narrow_gelu']:+.4f}% versus narrow GELU, {gaps['narrow_swiglu']:+.4f}% versus narrow
SwiGLU and {gaps['plain']:+.4f}% versus plain. Candidate peak is {peak:.3f} MiB,
mean timed update {ms:.3f} ms, {time_gap:+.2f}% versus plain. All controls' resource
and inference measurements remain in the report. No replicated plain advantage,
whole-network gradient property or universal speed benefit follows.

Late-plateau diagnostics pass {plateau}/6 models. Each final-800 loss change is
reported, and the separate 0.2% narrow margins do not change primary gates.
This one-seed duration comparison supplies no confidence interval, significance
or convergence proof. H077's three-seed result applies only to 800-step schedules.

Rates, models, initialization, optimizer calibration, execution, data and all
shared training/evaluation code match H077. Only steps/logging change; warmup
stretches to 320. These are fresh schedules with exactly matching initial weights,
NLL and first pre-update training loss; later short/long trajectories differ.

Five preflight checks pass with zero updates/model forwards, including all six
initial states and a 3,200-batch CUDA sampler reconstruction. All six final BF16
NLLs rescore exactly. Independent checks verify every checkpoint/moment, 3,200
step records per run, source/data hashes, sampler and full validation order,
parameter/operation counts and all decisions.

Work is 19,200 updates / 39,321,600 training targets and 13,552,896 original
validation exposures. Independent rescoring adds 1,936,128 validation targets,
zero updates. Read-only sampler qualification constructs 6,553,600 unscored target
elements. All trials finish first attempt. Three active model folders, six variants,
nine recipes, 171 original runs, 21 H076 and 18 H077 trials, 798 fitting cells and
all prior resource evidence/failures remain preserved.

Heavily reused development validation, short-selected rates and missing long-budget
replication/convergence/scaling/broader-data/published-comparator evidence leave
the gold research goal unmet. H051's plain failure and the local proof's limits
remain evidence; this result does not establish novel components or state-of-the-art.
"""
ledger.write_text(old.split(marker)[0] + entry, encoding="utf-8")
print(json.dumps({"status": "PASS", "updated_documents": [str(readme), str(current), str(ledger)], "primary_pass": primary}))
