"""Plot the paired diagnostic resource measurements, not training quality."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results.checkpoint_input_offload_v1.source.prepare import read

s = read("results/native_buffer_layout_v1/summary.json")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.75, wspace=0.3)
x = np.arange(2)
for j, (arm, color) in enumerate((("native", "#006D77"), ("reuse", "#C26A2F"))):
    rows = [
        next(v for v in s["metrics"] if v["dataset"] == d and v["arm"] == arm)
        for d in ("wikitext2", "tinystories")
    ]
    for ax, key in zip(axes, ("peak_mib", "event_ms"), strict=True):
        bars = ax.bar(
            x + (j - 0.5) * 0.32, [v[key] for v in rows], width=0.3, label=arm, color=color
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
for ax in axes:
    ax.set_xticks(x, ["WikiText", "TinyStories"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("Full diagnostic-case peak allocated MiB")
axes[1].set_ylabel("Median forward + backward event ms")
axes[0].set_ylim(0, max(v["peak_mib"] for v in s["metrics"]) * 1.18)
axes[1].set_ylim(0, max(v["event_ms"] for v in s["metrics"]) * 1.18)
axes[0].legend(loc="lower left", fontsize=9)
fig.suptitle(
    "H128 · Native-buffer reuse with matching gradient layout",
    x=0.09,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.text(
    0.09,
    0.83,
    "Both arms offload checkpoint inputs; ordinary RMSNorm and resident gradients.",
    fontsize=10,
)
fig.text(
    0.09,
    0.07,
    "10 probes per arm, 3 warmup · no optimizer updates · four independent native gradient replays",
    fontsize=9,
)
fig.savefig("research/figures/native_buffer_layout.png", dpi=160, facecolor="white")
plt.close(fig)
print("Figure written")
