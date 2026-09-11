"""Plot complete-job allocation and update latency from verified results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results.checkpoint_input_offload_v1.source.prepare import read

s = read("results/blas_workspace_mid_v1/summary.json")
for v in s["metrics"]:
    v["arm"] = v["workspace"] + "_" + v["arm"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.73, wspace=0.25)
x = np.arange(2)
for j, (arm, color) in enumerate(
    (
        ("high_native", "#667788"),
        ("high_reuse", "#006D77"),
        ("low_native", "#7B61A8"),
        ("low_reuse", "#C26A2F"),
    )
):
    rows = [
        next(v for v in s["metrics"] if v["dataset"] == d and v["arm"] == arm)
        for d in ("wikitext2", "tinystories")
    ]
    for ax, key in zip(axes, ("peak_mib", "event_ms"), strict=True):
        bars = ax.bar(
            x + (j - 1.5) * 0.20, [v[key] for v in rows], width=0.19, label=arm, color=color
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=7)
for ax in axes:
    ax.set_xticks(x, ["WikiText", "TinyStories"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Diagnostic peak allocated MiB")
axes[1].set_ylabel("Median forward + backward event ms")
for ax, key in zip(axes, ("peak_mib", "event_ms"), strict=True):
    ax.set_ylim(0, max(v[key] for v in s["metrics"]) * 1.2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=4, frameon=False)
fig.suptitle(
    "H132 · External cuBLAS workspace: 8 versus 32 MiB",
    x=0.08,
    ha="left",
    fontsize=14,
    fontweight="bold",
)
fig.text(
    0.08,
    0.81,
    "Eight diagnostic cases · one BF16 validation batch and ten FP32 backward probes each",
    fontsize=10,
)
fig.text(
    0.08,
    0.04,
    "High = 32 MiB per workspace; low = 8 MiB. Both offload checkpoint inputs. No optimizer updates.",
    fontsize=9,
)
fig.savefig("research/figures/blas_workspace_mid.png", dpi=160, facecolor="white")
plt.close(fig)
