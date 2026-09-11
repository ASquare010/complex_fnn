"""All-fixture H114 resource, precision and short-quality visualization."""

# ruff: noqa: I001 -- keep scientific package preloading before plotting imports

import sympy  # noqa: F401
import torch  # noqa: F401
import torch._dynamo  # noqa: F401

import csv
import gzip
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("results/fp32_classifier_profile_v1")
rows = list(
    csv.DictReader(io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode()))
)
summary = json.loads((ROOT / "summary.json").read_text())
scopes = ("narrow_wikitext2", "narrow_tinystories", "full_gelu", "full_swiglu")
labels = (
    "Narrow GELU\nWikiText, T512",
    "Narrow GELU\nTinyStories, T512",
    "Full GELU\nWikiText, T128",
    "Full SwiGLU\nWikiText, T128",
)
policies = ("block_bf16", "chunks_bf16", "block_fp32", "chunks_fp32")
names = ("Native BF16", "BF16 chunks", "Native FP32", "FP32 chunks")
colors = ("#67758b", "#bd7831", "#8867ae", "#007e80")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
specs = (
    ("memory_ratio", "Whole-job allocated memory / native BF16", 0.85),
    ("wall_ratio", "Median update wall time / native BF16", 1.25),
    ("global_gradient_relative_error", "Initial gradient relative L2 / native FP32", 0.002),
    ("final_nll_ratio", "Native validation NLL ratio after 50 updates", 1.01),
)
for axis, (key, title, gate) in zip(axes.flat, specs, strict=True):
    for i, policy in enumerate(policies):
        for j, scope in enumerate(scopes):
            peers = [r for r in rows if r["scope"] == scope and r["policy"] == policy]
            values = [float(p[key]) for p in peers]
            offsets = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else [0]
            x = j + (i - 1.5) * 0.17
            display = (
                [max(v, 1e-9) for v in values]
                if key == "global_gradient_relative_error"
                else values
            )
            axis.scatter(
                x + np.asarray(offsets),
                display,
                s=29,
                color=colors[i],
                alpha=0.85,
                label=names[i] if j == 0 else None,
                zorder=3,
            )
            axis.plot([x - 0.06, x + 0.06], [np.median(display)] * 2, color=colors[i], lw=2)
    axis.axhline(gate, color="#a03b42", ls="--", lw=1, label="Fixed candidate gate")
    if key != "global_gradient_relative_error":
        axis.axhline(1, color="#78838b", lw=0.7, alpha=0.5)
    else:
        axis.set_yscale("log")
    axis.set_title(title, loc="left", fontsize=11, weight="bold", pad=14)
    axis.set_xticks(range(4), labels)
    axis.grid(axis="y", alpha=0.15)
    axis.margins(x=0.06, y=0.16)
axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.3), ncol=5, frameon=False)
fig.suptitle(
    "FP32 classifier chunks: complete-model continuation screen",
    x=0.065,
    ha="left",
    fontsize=17,
    weight="bold",
    y=0.98,
)
fig.text(
    0.065,
    0.035,
    "14 saved models × 4 executions × 50 updates. Narrow: 3 seeds, 2 correlated starting policies per seed. Full: 1 seed each.\n"
    "Each dot is one fixture; short bars are medians. Exact-zero FP32-reference errors shown at 1e−9. No long-duration quality claim.",
    fontsize=10,
    color="#4b5563",
)
fig.subplots_adjust(left=0.065, right=0.98, top=0.84, bottom=0.14, hspace=0.42, wspace=0.25)
path = Path("research/figures/fp32_classifier_profile.png")
fig.savefig(path, dpi=160, facecolor="white")
print(path.as_posix())
