"""Render the audited incomplete H074 resource study; no promotion from partial data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/ungated_resource_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


audit = read("results/verification/ungated_resource_partial_analysis_v1.json")
failure = read(root / "failure.json")
assert audit["status"] == "PASS" and audit["completed_comparisons_exact"]
rows = {
    cell: read(root / "workers" / cell / "result.json") for cell in failure["completed_workers"]
}
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
modes = ("none", "block", "block_inner")
lines = [
    "| Form | Mode | Allocated MiB | Reserved MiB | Update ms | Targets/s | Clipped |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for form in forms:
    for mode in modes:
        cell = f"{form}_{mode}"
        if cell not in rows:
            reason = "failed before updates" if cell == failure["failed_worker"] else "unlaunched"
            lines.append(f"| {names[form]} | {mode} | {reason} | - | - | - | - |")
            continue
        r = rows[cell]
        lines.append(
            f"| {names[form]} | {mode} | {r['peak_allocated_bytes'] / 2**20:.3f} | {r['peak_reserved_bytes'] / 2**20:.3f} | {r['median_step_ms']:.3f} | {r['training_tokens_per_second']:.0f} | {100 * r['clipped_step_fraction']:.1f}% |"
        )
table = "\n".join(lines)
selected = [f for f in forms if f"{f}_block" in rows]
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), layout="constrained")
colors = ["#718096"] * 4 + ["#276b9b", "#15806f"]
for ax, key, scale, title in (
    (axes[0], "peak_allocated_bytes", 2**20, "Peak allocated memory (MiB)"),
    (axes[1], "median_step_ms", 1, "Median update time (ms)"),
):
    values = [rows[f"{f}_block"][key] / scale for f in selected]
    ax.barh(range(len(selected)), values, color=colors)
    ax.set_yticks(range(len(selected)), [names[f] for f in selected])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.17)
    ax.set_xlabel(title)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.015, i, f"{v:.1f}", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "H074 incomplete: completed whole-block checkpoint runs only\nMatched GELU unlaunched; same-width GELU inner-checkpoint run failed before updates",
    fontsize=12,
)
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/ungated_resource.{ext}", dpi=160)
plt.close(fig)
same, plain, full = rows["gelu_same_block"], rows["plain_block"], rows["full_gelu_block"]
text = f"""# H074 - Ungated full-model resources: incomplete runtime study

**INCOMPLETE_RUNTIME_FAILURE.** Seventeen of21 planned workers completed;
the eighteenth failed during AdamW construction, before its initial probe or any
optimizer update. The final three were not launched. Neither GELU form earns a
language screen. This is an incomplete qualification, not an architectural
rejection. H073's positive fitting evidence remains unchanged.

The [frozen plan](ungated_resource_plan.md) and [failure record](../results/ungated_resource_v1/failure.json)
preserve the allocation, stopping rule and exact trace. No scientific worker is
retried, no source or threshold is changed after dispatch, and no corpus is used.

## Completed evidence

Whole-block same-width GELU uses **{same["peak_allocated_bytes"] / 2**20:.3f} MiB** at
**{same["median_step_ms"]:.3f} ms/update**, compared with plain BlockShuffle's
{plain["peak_allocated_bytes"] / 2**20:.3f} MiB / {plain["median_step_ms"]:.3f} ms and full GELU's
{full["peak_allocated_bytes"] / 2**20:.3f} MiB / {full["median_step_ms"]:.3f} ms. Same-width GELU is
{100 * (1 - same["median_step_ms"] / plain["median_step_ms"]):.2f}% faster than plain in this measurement,
but {100 * (same["median_step_ms"] / full["median_step_ms"] - 1):.2f}% slower than full GELU.
Its inner-checkpoint execution remains unmeasured. No matched-width GELU
full-model result exists. Do not select a winning mode or apply the all21-worker
promotion gate to these partial data.

![Completed whole-block results only](figures/ungated_resource.png)

{table}

All times are fresh-initialization synthetic **training updates**, including
forward, backward, global clipping and AdamW. Synchronization brackets each
update; first10 warm up and the following10 define the timing window. Memory
peaks reset after update10, include model/gradient/optimizer storage and exclude
initial probes, diagnostics and serialization. Reserved memory is reported
separately. No language-quality or autoregressive-serving inference follows.
Each model is a separate sequential process; the short single-device timing
window and fixed run order limit comparisons. All forms were offered the same
three execution modes; missing modes remain missing.

## Model and optimization controls

All use batch16/context128/d384/L8/heads6/vocab4096, BF16 activations with FP32
parameters, TF32 disabled, four CPU threads and native eager Torch. The isolated
ungated configuration explicitly labels two projections; the active factory and
its six variants remain unchanged. Actual parameter counts are checked:

| Form | Hidden | FFN weights, 8 layers | Total weights | FFN reduction | Projection MAC/token/FFN |
|---|---:|---:|---:|---:|---:|
| Full SwiGLU | 1024 | 9,437,184 | 15,735,168 | 0% | 1,179,648 |
| Full GELU | 1536 | 9,437,184 | 15,735,168 | 0% | 1,179,648 |
| Narrow SwiGLU | 304 | 2,801,664 | 9,099,648 | 70.3125% | 350,208 |
| Narrow GELU | 456 | 2,801,664 | 9,099,648 | 70.3125% | 350,208 |
| Plain BlockShuffle | 2048 | 2,801,664 | 9,099,648 | 70.3125% | 350,208 |
| GELU same width | 2048 | 1,867,776 | 8,165,760 | 80.2083% | 233,472 |
| GELU matched (constructor only) | 3264 | 2,801,664 | 9,099,648 | 70.3125% | 350,208 |

MAC counts cover linear projections only; they exclude activation, shuffle,
attention and backward work. Shared seed17 name-local initialization preserves
all non-FFN tensors. Plain and same-width GELU share their common up/down factors.
Transformer down residual scale is1/4. This differs from H073's standalone
unit-row-variance initialization; cross-study resource values are not paired.

All workers use AdamW base rate0.0012, betas(0.9,0.95), eps1e-8, decay0.1 and
clip1. Structured factors use the existing fan-in LR multipliers. Narrow down
weights and learning rates receive their calibrated fan-in corrections exactly
once. The rate is fixed for resource/fidelity qualification, not tuned for GELU
quality. Each form has a different initial function; exactness comparisons are
within forms only. Initial/final layer magnitudes, sampled activation slopes,
near-zero fractions and gradient magnitudes are saved outside timed loops.
Finite measurements are not a whole-network gradient guarantee.

## Independent audit and failure scope

Five isolated harness checks pass before dispatch. Independent CPU inspection
reloads **17 initial/final checkpoint pairs** and verifies finite weights and
all AdamW moments at step20. All **11 available within-form comparisons** are
exact for initial logits/loss/every gradient, all20 losses/preclip norms and final
weights/moments. CPU regeneration reproduces the stream exactly. Shared initial
parameters and adapter identity/optimizer-reference preservation pass.

The completed work comprises **340 updates, 696,320 synthetic training targets
and 34,816 no-update probe targets**. The failed worker performs zero updates or
probe targets; its artifact directory is empty. There are zero corpus targets.
The runtime stops on `SystemError: Unmatched paren in format`, through
`torch._functorch.config`, `torch.utils._config_module` and CPython tokenization.
The source snapshot remains unchanged and no scientific retry occurs.

A separate read-only AST/tokenization probe passes on the implicated configuration
files and tokenizer source. It imports no Torch and does not reproduce the
original process/import state; it establishes neither a cause nor a repair.
The failure cause remains unknown. The prepared full-study audit/coordinator
are retained unexecuted; the partial audit explicitly grants no promotion.

See [independent partial audit](../results/verification/ungated_resource_partial_analysis_v1.json),
[tokenizer probe](../results/verification/ungated_resource_tokenizer_probe_v1.json),
and [final preservation audit](../results/verification/ungated_resource_final_v1.json).
Before continuing the earned resource comparison, investigate the import failure
and explicitly specify any operational recovery while preserving all completed
measurements. No automatic tuning, new architecture, larger budget or language
allocation is justified by an incomplete study. The full research goal remains
unachieved.
"""
Path("research/ungated_resource_results.md").write_text(text, encoding="utf-8")
print(
    json.dumps(
        {
            "status": "PASS",
            "scientific_status": "INCOMPLETE_RUNTIME_FAILURE",
            "completed_rows": len(rows),
            "scientific_reruns": 0,
        }
    )
)
