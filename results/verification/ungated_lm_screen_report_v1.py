"""Report all H076 rates and decisions from independently verified language endpoints."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/ungated_lm_screen_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


r = read(ROOT / "result.json")
a = read("results/verification/ungated_lm_screen_analysis_v1.json")
assert (
    a["status"] == "PASS"
    and a["all_21_rescores_exact"]
    and a["independently_rederived_gates"] == r["gates"]
)
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
    "plain": "BlockShuffle SwiGLU",
    "gelu_same": "BlockShuffle GELU h2048",
    "gelu_matched": "BlockShuffle GELU h3264",
}
short = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "narrow_swiglu": "Narrow SwiGLU",
    "narrow_gelu": "Narrow GELU",
    "plain": "Plain BlockShuffle",
    "gelu_same": "GELU same width",
    "gelu_matched": "GELU matched",
}
colors = {
    "full_swiglu": "#333333",
    "full_gelu": "#777777",
    "narrow_swiglu": "#b85c00",
    "narrow_gelu": "#cc6677",
    "plain": "#276b9b",
    "gelu_same": "#15806f",
    "gelu_matched": "#7853a6",
}
selected = {f: r["rows"][r["selected_cells"][f]] for f in forms}
passing = [f for f, v in r["earns_longer_comparison"].items() if v]
verdict = (
    "Both GELU forms pass the frozen short-screen gates."
    if len(passing) == 2
    else names[passing[0]] + " passes the frozen short-screen gates; the other form fails."
    if passing
    else "Both GELU forms fail the frozen short-screen gates."
)
summary = [
    "| Form | Selected peak LR | Final NLL | FFN weights | Total weights | Peak MiB | Mean timed update ms | Clipped |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
allrates = [
    "| Form | Peak LR | Final NLL | Peak MiB | Training targets/s | Full-sequence targets/s | Clipped |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
comparisons = [
    "| Candidate | vs full SwiGLU | vs full GELU | vs narrow SwiGLU | vs narrow GELU | vs plain |",
    "|---|---:|---:|---:|---:|---:|",
]
for f in forms:
    m = selected[f]
    summary.append(
        f"| {names[f]} | {m['training']['learning_rate']:.4f} | {m['validation_loss']:.9f} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.3f} | {1000 * m['timed_training_seconds'] / 190:.3f} | {100 * m['clipped_step_fraction']:.1f}% |"
    )
    for rate in (0.0003, 0.0006, 0.0012):
        m = r["rows"][f"{f}_lr{round(rate * 1e6)}"]
        allrates.append(
            f"| {names[f]} | {rate:.4f} | {m['validation_loss']:.9f} | {m['peak_allocated_vram_bytes'] / 2**20:.3f} | {m['training_tokens_per_second']:.0f} | {m['inference_tokens_per_second']:.0f} | {100 * m['clipped_step_fraction']:.1f}% |"
        )
for f in ("gelu_same", "gelu_matched"):
    values = [
        100 * (selected[f]["validation_loss"] / selected[ref]["validation_loss"] - 1)
        for ref in ("full_swiglu", "full_gelu", "narrow_swiglu", "narrow_gelu", "plain")
    ]
    comparisons.append("| " + names[f] + " | " + " | ".join(f"{v:+.4f}%" for v in values) + " |")
gatelines = ["| Frozen gate | GELU h2048 | GELU h3264 |", "|---|---|---|"]
for g in r["gates"]["gelu_same"]:
    gatelines.append(
        "| "
        + g.replace("_", " ")
        + " | "
        + ("PASS" if r["gates"]["gelu_same"][g] else "FAIL")
        + " | "
        + ("PASS" if r["gates"]["gelu_matched"][g] else "FAIL")
        + " |"
    )
summary, allrates, comparisons, gates = map("\n".join, (summary, allrates, comparisons, gatelines))
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
for f in forms:
    values = [
        r["rows"][f"{f}_lr{round(rate * 1e6)}"]["validation_loss"]
        for rate in (0.0003, 0.0006, 0.0012)
    ]
    axes[0].plot(
        [0.0003, 0.0006, 0.0012],
        values,
        marker="o",
        color=colors[f],
        linestyle="--" if f.startswith(("full_", "narrow_")) else "-",
        label=short[f],
    )
axes[0].set_xscale("log", base=2)
axes[0].set_xticks([0.0003, 0.0006, 0.0012], ["0.0003", "0.0006", "0.0012"])
axes[0].set_xlabel("Peak learning rate")
axes[0].set_ylabel("Final validation NLL (lower is better)")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(alpha=0.15)
best_full = min(selected[f]["validation_loss"] for f in ("full_swiglu", "full_gelu"))
gaps = [100 * (selected[f]["validation_loss"] / best_full - 1) for f in forms]
axes[1].axvline(0, color="#999999", linewidth=1)
axes[1].axvline(1, color="#aa3333", linestyle="--", linewidth=1, label="1% full-control allowance")
for i, (f, gap) in enumerate(zip(forms, gaps)):
    axes[1].scatter(gap, i, color=colors[f], s=60)
    axes[1].annotate(
        f"{gap:+.2f}%", (gap, i), xytext=(5, 6), textcoords="offset points", fontsize=9
    )
axes[1].set_yticks(range(7), [short[f] for f in forms])
axes[1].invert_yaxis()
axes[1].set_xlim(min(-0.2, min(gaps) - 0.4), max(1.8, max(gaps) + 0.7))
axes[1].set_xlabel("Selected relative NLL vs the better full control (%)")
axes[1].legend(fontsize=8, loc="lower right")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "H076: WikiText-2, seed 17, 200 updates; three equal peak-rate budgets per form\nDevelopment validation selection; no official test or convergence claim",
    fontsize=12,
)
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/ungated_lm_screen.{ext}", dpi=160)
plt.close(fig)
worker_seconds = sum(
    read(ROOT / "processes" / (cell + ".json"))["elapsed_seconds"] for cell in r["rows"]
)
end_trends = []
for f in forms:
    m = selected[f]
    h = list(
        map(
            json.loads,
            (ROOT / "runs" / r["selected_cells"][f] / "history.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    )
    end_trends.append(
        f"| {names[f]} | {100 * (h[-1]['validation_loss'] / h[-2]['validation_loss'] - 1):+.4f}% |"
    )
trends = "\n".join(
    ["| Selected form | Validation NLL change, step 150 to 200 |", "|---|---:|", *end_trends]
)
next_step = (
    "Passing forms earn only a separately frozen longer comparison. Failed fixed recipes receive no automatic tuning or repair."
    if passing
    else "Neither tested GELU recipe earns longer training at this budget. The failures do not disprove every ungated structure, but they close these fixed recipes without automatic tuning or repair."
)
text = f"""# H076 - Ungated GELU WikiText language screen

**{verdict}** All 21 fresh trials complete: seven forms, three equal rates,
seed 17 and 200 updates each. Independent analysis audits every checkpoint and
rescores all 21 final validation NLLs exactly. {next_step}
The matched candidate's selected NLL is **5.895276**, versus narrow GELU's
**5.902164**: a **0.1167%** improvement, below the descriptive 0.2% margin.
It is **0.2721%** above full GELU, with **28.79%** less allocated peak memory
and **21.81%** more mean update time. The selected narrow GELU rate is 0.0006;
its 0.0012 result is not the strongest narrow reference. The same-width recipe
fails full-GELU and both narrow quality gates and earns no longer allocation.

The full research goal remains unachieved: a short single-seed development screen
does not establish convergence, general superiority or a novel architecture.

## Selected endpoints

Selection uses minimum final validation NLL within each form, ties lower rate.
Every count, memory value and timing below comes from that same selected run.

{summary}

Relative NLL differences below use selected endpoints; negative values mean lower
candidate NLL. The full-control allowance is 1% against EACH full reference. Both
calibrated narrow references must be beaten strictly; the separate 0.2% narrow
margin remains descriptive and does not alter these gates.

{comparisons}

{gates}

![All rates and selected full-control gaps](figures/ungated_lm_screen.png)

The same-width candidate uses 1,867,776 FFN weights (80.2083% fewer than full),
while matched uses 2,801,664 (70.3125% fewer). Their total counts are 8,165,760 and
9,099,648, respectively. Dense narrow controls match the larger candidate's FFN
budget; they have more weights than the same-width candidate. The width comparison
therefore exposes a parameter/quality/resource tradeoff rather than a pure
activation-only ablation. Results are selected development scores, not test scores.

## Every allocated rate

{allrates}

All rates and trajectories are retained. A winner at 0.0012 is on the search
boundary and does not establish an optimal rate. No later rate is added based on
this result. Same-rate comparisons are available directly in this complete table;
older language runs do not supply the current gate values.

{trends}

Endpoint changes describe this short schedule. There is no convergence or
statistical-significance claim from one seed. Historical H051 longer-training
failure for plain BlockShuffle remains part of the evidence; H073's synthetic
GELU advantages and H075's resource pass do not imply language superiority.

## Shared methods and explicit recipe differences

The [frozen plan](ungated_lm_screen_plan.md) sets batch 16/context 128/d384/L8/heads 6/
vocab 4096, structured groups 8, BF16 with FP32 parameters, TF32 disabled, four CPU
threads and one sequential GPU worker. Every trial starts from the original
name-local Transformer initialization at seed 17, with down residual scale 1/4.
All non-FFN tensors match; plain and same-width GELU share common up/down factors.
Initial weights independently reconstruct exactly and match H074/H075.

The shared trainer, sampler, attention, norm, loss, evaluation and schedule source
remain unchanged. Twenty warmup updates lead into cosine decay ending at 0.1 times
peak. AdamW betas (0.9, 0.95), eps 1e-8, base decay 0.1 and global clip 1 are common.
Structured projections use fan-in factor LR correction with parameter decay.
Narrow down initialization and LR receive reference_hidden/actual_hidden exactly
once. Both narrow forms use product-decay correction, weight_decay 0.1/LR_scale
for their larger down LR, following the established narrow language recipe.
H074/H075 used parameter decay for synthetic narrow resource probes; this explicit
change prevents an extra effective down-decay penalty in the language control.

Execution is fixed from H075 before language results: whole-block checkpointing
for all forms except narrow GELU, which uses block plus inner native activation
checkpointing. The isolated constructor correctly labels blockshuffle_gelu and
counts two projections, using the existing operator. The active registry is
unchanged. The operation counter reports 2 times actual FFN weights per token:
full 18,874,368; narrow/plain/matched 5,603,328; same-width 3,735,552 matrix FLOPs
across eight layers. Attention is a dense upper estimate, and nonlinearities,
shuffles, norms, softmax, loss and optimizer costs are excluded.

Scalar observer hooks call the original loss/initialization/clipping functions,
record every training loss/preclip norm/rate, and do not alter parameters or
gradients. Their logging overhead is included for every model. Training timing
excludes first 10 updates, evaluation and checkpoint writes, and includes sampling,
optimizer and occasional gradient diagnostics. Reported update time is the mean
over the 190 timed updates, not H075's short-window median.

Allocated peak includes GPU train/validation cache, training and intervening/final
validation, as the shared trainer measures it. It is compared with fresh controls
under this same harness, not H075's synthetic peaks. Full-sequence inference uses
B16/T128 without a KV cache: 10 warmups and three repeats of 30 forwards. It is not
autoregressive serving. End-of-run layer activations, near-zero fractions,
gradients, clipping, finite weights and optimizer moments are preserved. These
measurements do not prove a whole-network gradient guarantee.

## Data, execution and independent verification

The frozen WikiText-2 cache is Salesforce/wikitext revision
f776294184f13b8ff2337b3841cf9269a6216d1e with train-only 4096-token BPE. It contains
3,083,650 training and 322,802 validation tokens. CUDA sampler seed 10017 and the
200-step final sampler state agree across all 21 trials. Full validation covers
2,521 contiguous 128-target windows: 322,688 targets in 158 batches, last 9 windows;
113 suffix tokens lie outside complete windows. The exact target order is audited.
Official test data is neither fetched nor scored. Repeated adaptive development
on this already-used validation split is not an independent holdout evaluation.

All 21 trials provide **4,200 updates and 8,601,600 training targets**, with seven
shared full-validation evaluations per run: **47,435,136 validation exposures**.
The bounded trial processes total {worker_seconds:.2f} s. Five isolated integration
checks pass first. Their native/selected narrow observer-fidelity comparisons
perform 8 separate CPU updates on a fixed 2x16 training-prefix batch (256 target
presentations). Those temporary states do not initialize or select language runs.
The prior 108 active tests are not rerun solely for isolated adapters.

Independent analysis reloads all 21 final checkpoints, verifies all weights and
AdamW moments at step 200, checks every sidecar/log/schedule/clip record, actual
parameter and operation counts, initial signatures, source/data hashes and
optimizer calibration. It reconstructs the CUDA sampler state and validation
order, then rescores every checkpoint once through the BF16 forward evaluator:
**6,776,448 additional validation targets, zero updates**, all NLLs exact. This
checks the saved development scores and does not create new generalization data.
All selections and original gates are independently rederived.

See [independent audit](../results/verification/ungated_lm_screen_analysis_v1.json),
[all run metrics](../results/ungated_lm_screen_v1/result.json), and
[final preservation audit](../results/verification/ungated_lm_screen_final_v1.json).
Every allocated trial, prior failure, frozen plan and earlier result is preserved.
There is no automatic scientific retry, active-model addition or publication.
"""
Path("research/ungated_lm_screen_results.md").write_text(text, encoding="utf-8")
print(
    json.dumps(
        {
            "status": "PASS",
            "verdict": verdict,
            "earns_longer_comparison": r["earns_longer_comparison"],
            "scientific_reruns": 0,
        }
    )
)
