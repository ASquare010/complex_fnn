"""Plot trained-checkpoint validation, complete-job memory and update time."""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from results.checkpoint_input_offload_v1.source.prepare import read

ROOT = Path("results/compact_long_training_v1")
r, s = read(ROOT / "result.json"), read(ROOT / "summary.json")
colors = dict(native="#006D77", chunks="#7B61A8", combined="#C26A2F")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
fig.subplots_adjust(top=0.86, bottom=0.12, left=0.09, right=0.98, hspace=0.44, wspace=0.25)
for ax, dataset in zip(axes[0], ("wikitext2", "tinystories"), strict=True):
    for arm, color in colors.items():
        c = next(
            row["measurement"]
            for row in r["cases"]
            if row["arm"] == arm and row["measurement"]["fixture"]["dataset"] == dataset
        )
        values = [v["validation"]["nll"] for v in c["intermediate_checkpoints"]] + [
            c["final_validation"]["nll"]
        ]
        ax.plot([200, 400, 800], values, "o-", color=color, label=arm)
    ax.set_title(dataset + ": validation at saved trained states", loc="left")
    ax.set_xlabel("Training update")
    ax.set_ylabel("Validation NLL (lower is better)")
    ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=9)
x = np.arange(2)
for j, (arm, color) in enumerate(colors.items()):
    rows = [
        next(v for v in s["metrics"] if v["dataset"] == d and v["arm"] == arm)
        for d in ("wikitext2", "tinystories")
    ]
    for ax, key in zip(axes[1], ("peak_mib", "event_ms"), strict=True):
        bars = ax.bar(
            x + (j - 1) * 0.25, [v[key] for v in rows], width=0.23, color=color, label=arm
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
for ax in axes[1]:
    ax.set_xticks(x, ["WikiText", "TinyStories"])
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[1, 0].set_ylabel("Complete-job peak allocated MiB")
axes[1, 1].set_ylabel("Median CUDA-event update ms")
axes[1, 0].set_ylim(0, max(v["peak_mib"] for v in s["metrics"]) * 1.2)
axes[1, 1].set_ylim(0, max(v["event_ms"] for v in s["metrics"]) * 1.2)
fig.suptitle(
    "H126 · Long training with conventional and chunked controls",
    x=0.09,
    ha="left",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.09,
    0.905,
    "Six 800-step runs · same initial weights and batches per corpus · one seed per corpus",
    fontsize=10,
)
fig.text(
    0.09,
    0.045,
    "Memory includes all diagnostics/evaluation; runtime uses steps 21–800. This does not establish multiseed or architectural success.",
    fontsize=9,
)
fig.savefig("research/figures/compact_long_training.png", dpi=160, facecolor="white")
plt.close(fig)
print("research/figures/compact_long_training.png")
