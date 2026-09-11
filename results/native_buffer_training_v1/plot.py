"""Plot complete-job allocation and update latency from verified results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results.checkpoint_input_offload_v1.source.prepare import read

s = read("results/native_buffer_training_v1/summary.json")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.73, wspace=0.25)
x = np.arange(2)
for j, (arm, color) in enumerate(
    (("ordinary", "#667788"), ("native", "#006D77"), ("offload", "#7B61A8"), ("reuse", "#C26A2F"))
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
axes[0].set_ylabel("Complete-job peak allocated MiB")
axes[1].set_ylabel("Median CUDA-event update ms")
for ax, key in zip(axes, ("peak_mib", "event_ms"), strict=True):
    ax.set_ylim(0, max(v[key] for v in s["metrics"]) * 1.2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=4, frameon=False)
fig.suptitle(
    "H130 · Native-buffer reuse through complete optimizer updates",
    x=0.08,
    ha="left",
    fontsize=14,
    fontweight="bold",
)
fig.text(
    0.08,
    0.81,
    "Eight matched 30-update continuations · initial/final evaluation and all training phases included",
    fontsize=10,
)
fig.text(
    0.08,
    0.04,
    "Ordinary = nondeterministic native control; other arms use deterministic-default execution. Short-run evidence only.",
    fontsize=9,
)
fig.savefig("research/figures/native_buffer_training.png", dpi=160, facecolor="white")
plt.close(fig)
