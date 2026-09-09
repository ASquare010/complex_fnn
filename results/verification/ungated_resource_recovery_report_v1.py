"""Report the complete H074/H075 resource study with explicit recovery provenance."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/ungated_resource_recovery_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


result = read(ROOT / "result.json")
audit = read("results/verification/ungated_resource_recovery_analysis_v1.json")
assert audit["status"] == "PASS" and result["gates"] == audit["independently_rederived_gates"]
forms = (
    "full_swiglu",
    "full_gelu",
    "narrow_swiglu",
    "narrow_gelu",
    "plain",
    "gelu_same",
    "gelu_matched",
)
names = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "narrow_swiglu": "Narrow SwiGLU",
    "narrow_gelu": "Narrow GELU",
    "plain": "Plain BlockShuffle",
    "gelu_same": "GELU same width",
    "gelu_matched": "GELU matched",
}
selected = {f: result["rows"][f + "_" + result["selected_modes"][f]] for f in forms}
summary = [
    "| Form | Selected mode | FFN weights | Total weights | Allocated MiB | Update ms |",
    "|---|---|---:|---:|---:|---:|",
]
allrows = [
    "| Form | Mode | Origin | Allocated MiB | Reserved MiB | Update ms | Targets/s | Clipped |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for f in forms:
    r = selected[f]
    summary.append(
        f"| {names[f]} | {r['mode']} | {r['ffn_parameters']:,} | {r['total_parameters']:,} | {r['peak_allocated_bytes'] / 2**20:.3f} | {r['median_step_ms']:.3f} |"
    )
    for mode in ("none", "block", "block_inner"):
        cell = f"{f}_{mode}"
        row = result["rows"][cell]
        origin = "H075" if result["origins"][cell]["root"] == ROOT.as_posix() else "H074"
        allrows.append(
            f"| {names[f]} | {mode} | {origin} | {row['peak_allocated_bytes'] / 2**20:.3f} | {row['peak_reserved_bytes'] / 2**20:.3f} | {row['median_step_ms']:.3f} | {row['training_tokens_per_second']:.0f} | {100 * row['clipped_step_fraction']:.1f}% |"
        )
summary = "\n".join(summary)
allrows = "\n".join(allrows)
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), layout="constrained")
colors = ["#718096"] * 4 + ["#276b9b", "#15806f", "#0b5448"]
for ax, key, scale, title in (
    (axes[0], "peak_allocated_bytes", 2**20, "Peak allocated memory (MiB)"),
    (axes[1], "median_step_ms", 1, "Median update time (ms)"),
):
    values = [selected[f][key] / scale for f in forms]
    ax.barh(range(7), values, color=colors)
    ax.set_yticks(range(7), [names[f] for f in forms])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.17)
    ax.set_xlabel(title)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.015, i, f"{v:.1f}", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "H074 + H075: complete resource qualification; minimum allocated-memory modes\n17 original runs retained + 4 recovered runs; identical updates across all checkpoint options",
    fontsize=12,
)
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/ungated_resource_recovery.{ext}", dpi=160)
plt.close(fig)
same, matched, plain, sg, gelu = [
    selected[f] for f in ("gelu_same", "gelu_matched", "plain", "full_swiglu", "full_gelu")
]
new_worker_seconds = sum(
    read(ROOT / "processes" / (cell + ".json"))["elapsed_seconds"]
    for cell in result["rows"]
    if result["origins"][cell]["root"] == ROOT.as_posix()
)
gate_lines = ["| Gate | Same-width GELU | Matched GELU |", "|---|---|---|"]
for gate in result["gates"]["gelu_same"]:
    gate_lines.append(
        "| "
        + gate.replace("_", " ")
        + " | "
        + ("PASS" if result["gates"]["gelu_same"][gate] else "FAIL")
        + " | "
        + ("PASS" if result["gates"]["gelu_matched"][gate] else "FAIL")
        + " |"
    )
gate_table = "\n".join(gate_lines)
text = f"""# H075 - Complete ungated Transformer resource qualification

**Both GELU forms qualify for a separately frozen short language-model screen.**
The bounded recovery completes all four missing H074 cells while retaining its17
completed runs. Independent CPU analysis verifies all21 initial/final checkpoint
pairs, all14 execution comparisons exactly, and every original resource gate.
This is a resource and numerical-fidelity result, not language-quality,
convergence, gradient-health, architecture-novelty or breakthrough evidence.

Matched GELU retains **70.3125% fewer FFN weights**, uses
**{matched["peak_allocated_bytes"] / 2**20:.3f} MiB** and takes **{matched["median_step_ms"]:.3f} ms/update**
in its selected whole-block mode. Same-width GELU retains **80.2083% fewer FFN
weights**, uses **{same["peak_allocated_bytes"] / 2**20:.3f} MiB** and takes **{same["median_step_ms"]:.3f} ms/update**.
All forms receive the same three execution choices. Selection minimizes allocated
memory, ties measured median update time then original mode order.

{summary}

![Selected full-model resource measurements](figures/ungated_resource_recovery.png)

Matched GELU's selected peak is {100 * (1 - matched["peak_allocated_bytes"] / sg["peak_allocated_bytes"]):.2f}% below full
SwiGLU and {100 * (1 - matched["peak_allocated_bytes"] / gelu["peak_allocated_bytes"]):.2f}% below full GELU. Its update takes
{100 * (1 - matched["median_step_ms"] / plain["median_step_ms"]):.2f}% less time than plain BlockShuffle, but
{100 * (matched["median_step_ms"] / sg["median_step_ms"] - 1):.2f}% more than full SwiGLU and
{100 * (matched["median_step_ms"] / gelu["median_step_ms"] - 1):.2f}% more than full GELU. These are short-window measurements
from one GPU, with a time gap between the original and recovery cohorts. Small
time differences between the two GELU candidates are not evidence of a consistent
speed ordering. Fewer weights do not imply faster dense-kernel execution.

## Original gates, independently rederived

The [H074 plan](ungated_resource_plan.md) is unchanged. Candidates require all21
completed workers and exact within-form comparisons, finite diagnostics/states,
at least70% FFN reduction, peak allocation no greater than110% of EACH full
control's best option, and update time no greater than125% of plain in the same
mode. Both candidates select whole-block checkpointing. The selected full
controls also use block; narrow GELU selects block_inner. The runtime criterion
limits cost relative to the compressed reference, not relative to full dense.

{gate_table}

Both earn only a separately specified short language comparison with equal
training/tuning budgets, full and calibrated narrow controls, actual corpus
memory and validation NLL. H073's synthetic gains are not language gains.
Subsequent promotion still requires quality against both full controls, narrow
comparison, three seeds, longer/converged training, scaling, broader data and
strong published alternatives. The overall research goal remains unachieved.

## Every measurement and its origin

{allrows}

The unchanged worker uses batch16/context128/d384/L8/heads6/vocab4096, BF16
autocast with FP32 parameters, TF32 disabled, four CPU threads and native eager
Torch. All forms start at seed17 from the standard name-local Transformer
initializer, with down residual scale1/4. All non-FFN tensors match; plain and
same-width GELU also share common up/down factors. The isolated GELU config
correctly counts two structured projections without adding an active variant.

AdamW uses fixed base rate0.0012, betas(0.9,0.95), eps1e-8, decay0.1 and global
clip1. Existing structured-factor LR correction and exactly-once dense narrow
down initialization/LR calibration are preserved. This rate was fixed for
resource qualification, not selected for GELU language quality. CPU token stream
seed60017, shape20x16x129, is shared exactly. No corpus data or cache is used.

Each worker performs one initial probe and20 synthetic updates. First10 updates
warm up; synchronized updates11-20 supply timings, and peak allocation resets
after update10. Model, gradients and AdamW storage are included. Initial probes,
layer/slope diagnostics and serialization are outside the measurement window.
All reported throughput is full Transformer training on synthetic token batches;
it is neither autoregressive serving nor corpus-training throughput.

Projection-only MAC counts per token/FFN are1,179,648 for both full controls,
350,208 for narrow/plain/matched, and233,472 for same-width GELU. These follow
from actual linear weights and exclude attention, activation, shuffle and
backward work. Parameter reduction therefore describes neither full-model FLOPs
nor measured GPU speed. Layer magnitudes, near-zero fractions, sampled activation
slopes, gradient norms and clipping are retained for every worker; finite values
at this budget are not a vanishing/exploding-gradient theorem.

## Diagnostic and operational recovery

H074 stopped at `SystemError: Unmatched paren in format` during AdamW's lazy Torch
configuration import, before its same-width block_inner probe or updates. The
original traceback, terminal failure, empty worker directory and17 completed
runs remain untouched. Its earlier read-only AST/tokenization check passed.

The [H075 recovery plan](ungated_resource_recovery_plan.md) specifies three
construction-only probes in fresh processes: one two-weight CPU optimizer and
two original full-size same-width CUDA model/optimizer constructions. All three
pass. Both full-size probes reproduce the original initial weights, empty
optimizer state and optimizer groups exactly. They perform no forward, backward,
update or scored target. No dependency, cache, machine setting or scientific
source is changed. Successful fresh constructions do not identify or repair the
original failure's cause, which remains unknown.

The plan then permits one explicit retry of the pre-update failed cell and the
first launches of three matched-width cells. All four complete first attempt
under the new coordinator, in {new_worker_seconds:.2f} s total worker process time.
The implementation calls the original frozen worker directly with only its
artifact root rebound. No completed cell is rerun, no source/threshold changes,
and no hidden retry loop is used. Results explicitly retain their old/new origins.

New work adds **80 updates, 163,840 synthetic training targets and8,192 initial
probe targets**. Combined with H074, this is exactly its planned **420 updates,
860,160 training targets and43,008 probe targets**. There are22 scientific worker
launches:21 completed and the original zero-update failure. The three diagnostic
constructions are separate. The original successful cells receive no extra budget.

Independent CPU audit directly compares final weights and all AdamW moments,
initial signatures, all20 losses/preclip norms, token/RNG records, actual counts,
state finiteness and preserved shared initialization. All14 within-form execution
comparisons are exact. The same-width inner comparison spans cohorts; matched
comparisons use the newly completed native baseline. The original H074 five
integration tests and prior108 active tests are not rerun just for orchestration.

A report-writer quoting error occurred before its generator file was created;
the command was corrected with distinct string delimiters. Its record is kept,
and no scientific worker or audit was repeated for this documentation correction.
See [independent audit](../results/verification/ungated_resource_recovery_analysis_v1.json),
[combined result and origins](../results/ungated_resource_recovery_v1/result.json),
and [preservation audit](../results/verification/ungated_resource_recovery_final_v1.json).
The active tree remains three model folders, six variants and nine recipes.
The103 frozen source files, prior plans, checkpoints and archived failures are
preserved. This conventional control remains isolated until language evidence
justifies an active model decision.
"""
Path("research/ungated_resource_recovery_results.md").write_text(text, encoding="utf-8")
print(
    json.dumps(
        {
            "status": "PASS",
            "earns_language_screen": result["earns_language_screen"],
            "rows": 21,
            "scientific_reruns": 0,
        }
    )
)
