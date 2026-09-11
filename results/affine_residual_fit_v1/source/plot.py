"""Standalone scientific figure from audited summary; no Torch import."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("results/affine_residual_fit_v1")
summary = json.loads((root / "summary.json").read_text())
forms = ("affine_gelu", "affine_swiglu", "gelu", "relu", "leaky_relu", "prelu", "silu", "swiglu")
labels = (
    "Affine\nGELU",
    "Affine\nSwiGLU",
    "GELU",
    "ReLU",
    "Leaky\nReLU",
    "PReLU",
    "SiLU",
    "SwiGLU",
)
colors = {"gelu": "#137e8a", "swiglu": "#be632f"}
records = {(r["teacher"], r["form"]): r for r in summary["statistics"]}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.12, top=0.86, wspace=0.28, hspace=0.52)
fig.suptitle(
    "Affine capacity survives; does learned compression?",
    x=0.075,
    ha="left",
    y=0.97,
    fontsize=18,
    weight="bold",
)
fig.text(
    0.075,
    0.918,
    "d384 FFNs · two fixed teachers · three depths · three fresh sampling seeds · 600 updates per rate",
    fontsize=10,
    color="#52616b",
)
ax = axes[0, 0]
for teacher, color in colors.items():
    data = [r for r in summary["capacity"] if r["teacher"] == teacher]
    ax.plot(
        [r["rank"] for r in data],
        [r["floor"]["mean"] for r in data],
        "o-",
        color=color,
        label=f"{teacher.upper()} oracle floor",
    )
    ax.plot(
        [r["rank"] for r in data],
        [r["training_basis_oracle_upper"]["mean"] for r in data],
        "--",
        color=color,
        alpha=0.8,
    )
ax.axhline(0.05, ls=":", color="#555555", lw=1)
ax.set(
    xlabel="Correction rank h",
    ylabel="Normalized reconstruction error",
    title="A. Capacity diagnostic (H106)",
    yscale="log",
)
ax.legend(fontsize=8, loc="lower left")
ax.text(
    0.98,
    0.94,
    "Dashed: train-basis oracle upper\nDotted: local allocation gate",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8,
)
ax = axes[0, 1]
x = np.arange(len(forms))
for index, (teacher, color) in enumerate(colors.items()):
    values = [records[teacher, f]["reporting_mse"]["mean"] for f in forms]
    seeds = [list(records[teacher, f]["seed_mse"].values()) for f in forms]
    errors = np.array(
        [[v - min(s) for v, s in zip(values, seeds)], [max(s) - v for v, s in zip(values, seeds)]]
    )
    ax.bar(
        x + (index - 0.5) * 0.36,
        values,
        0.34,
        color=color,
        yerr=errors,
        capsize=2,
        label=f"{teacher.upper()} teacher",
    )
ax.axhline(0.05, ls=":", color="#555555", lw=1)
ax.set(
    xticks=x,
    xticklabels=labels,
    ylabel="Selected reporting MSE",
    title="B. Actual learning at matched parameters",
)
ax.legend(fontsize=8, loc="upper right")
ax.text(
    0.02,
    0.95,
    "Bars: means; whiskers: range of 3 seed means",
    transform=ax.transAxes,
    va="top",
    fontsize=8,
)
ax = axes[1, 0]
for index, (teacher, color) in enumerate(colors.items()):
    values = [records[teacher, f]["peak_allocated_bytes"]["mean"] / 2**20 for f in forms]
    values.append(summary["full_profiles"][teacher]["peak_allocated_bytes"]["mean"] / 2**20)
    ax.bar(np.arange(9) + (index - 0.5) * 0.36, values, 0.34, color=color)
ax.set(
    xticks=np.arange(9),
    xticklabels=(*labels, "Full\nprofile"),
    ylabel="Peak allocated MiB",
    title="C. Local training tensor allocation",
)
ax.text(
    0.02,
    0.94,
    f"Entire compression pipeline peak: {summary['pipeline_peak_cuda_bytes'] / 2**20:.1f} MiB\nFull teacher capture is included in that separate total.",
    transform=ax.transAxes,
    va="top",
    fontsize=8,
)
ax.set_ylim(
    0,
    max(summary["full_profiles"][t]["peak_allocated_bytes"]["mean"] / 2**20 for t in colors) * 1.36,
)
ax = axes[1, 1]
names = (
    "Affine GELU\nupdate",
    "Affine GELU\ninference",
    "Affine SwiGLU\nupdate",
    "Affine SwiGLU\ninference",
)
for index, (teacher, color) in enumerate(colors.items()):
    values = []
    for form in forms[:2]:
        record = summary["decisions"][form]["teachers"][teacher]
        values.extend((record["update_ratio_to_gelu"], record["raw_inference_ratio_to_gelu"]))
    ax.bar(np.arange(4) + (index - 0.5) * 0.36, values, 0.34, color=color)
ax.axhline(1, color="#555555", lw=0.7)
ax.axhline(1.25, color="#555555", lw=1, ls=":")
ax.set(
    xticks=np.arange(4),
    xticklabels=names,
    ylabel="Time / ordinary narrow GELU",
    title="D. Measured cost at batch 256",
)
ax.text(
    0.02, 0.95, "Dotted: fixed 1.25x cost ceiling", transform=ax.transAxes, va="top", fontsize=8
)
fig.text(
    0.075,
    0.028,
    "Local post-training reconstruction, not language NLL. CPU data; FP32 students; RTX 4070 Laptop GPU. No breakthrough or novelty claim.",
    fontsize=9,
    color="#52616b",
)
destination = Path("research/figures/affine_residual.png")
fig.savefig(destination, dpi=160)
print(destination)
