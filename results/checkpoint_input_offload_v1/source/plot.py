"""Audited allocated peaks and untraced phase-event costs; no training claim."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from results.checkpoint_input_offload_v1.source.prepare import ROOT, read  # noqa: E402

p, r, s = [read(ROOT / n) for n in ("protocol.json", "result.json", "summary.json")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
fig.subplots_adjust(left=0.075, right=0.98, top=0.78, bottom=0.26, wspace=0.3)
x = np.arange(4)
for j, (mode, color) in enumerate((("native", "#006D77"), ("offload", "#C26A2F"))):
    cases = [
        next(c for c in r["cases"] if c["fixture"] == f and c["mode"] == mode)
        for f in p["fixtures"]
    ]
    values = [c["peak_allocated_bytes"] / 2**20 for c in cases]
    bars = axes[0].bar(x + (j - 0.5) * 0.36, values, width=0.34, label=mode, color=color)
    axes[0].bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
axes[0].set_ylabel("Complete diagnostic-case peak allocated MiB")
axes[0].set_ylim(0, 480)
axes[0].set_title("A   VRAM across all measured phases", loc="left", fontweight="bold")
axes[0].legend(ncol=2, loc="upper right", fontsize=9)
for j, (key, color, label) in enumerate(
    (("event_sum_ms", "#7B61A8", "CUDA events"), ("wall_ms", "#507F33", "Synchronized wall"))
):
    values = [v["time_ratios"][key] for v in s["comparisons"]]
    bars = axes[1].bar(x + (j - 0.5) * 0.36, values, width=0.34, label=label, color=color)
    axes[1].bar_label(bars, fmt="%.2fx", fontsize=8, padding=3)
axes[1].axhline(1.15, ls="--", lw=1, color="#AA3333", label="1.15x limit")
axes[1].set_ylim(
    0, max(1.65, 1.2 * max(v["time_ratios"][k] for v in s["comparisons"] for k in v["time_ratios"]))
)
axes[1].set_ylabel("Offload / native median time")
axes[1].set_title("B   Untraced forward + backward time", loc="left", fontweight="bold")
axes[1].legend(ncol=2, loc="upper left", fontsize=8)
for ax in axes:
    ax.set_xticks(
        x,
        [
            "WikiText\nnative loss",
            "WikiText\nchunked loss",
            "TinyStories\nnative loss",
            "TinyStories\nchunked loss",
        ],
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
fig.suptitle(
    "H121 · Checkpoint inputs in pinned host memory",
    x=0.075,
    ha="left",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.075,
    0.84,
    "Two trained states · 8 matched cases · 262 backwards including audits · zero optimizer updates",
    fontsize=10,
)
fig.text(
    0.075,
    0.105,
    "Memory includes warmup, trace and serialization; timings use the last 20 untraced repetitions. FP32, RTX 4070 Laptop.",
    fontsize=9,
)
fig.text(
    0.075,
    0.06,
    "Pinned host memory is additional RAM. These are fixed-state probes, not complete training jobs or endpoint quality tests.",
    fontsize=9,
)
fig.savefig("research/figures/checkpoint_input_offload.png", dpi=170, facecolor="white")
plt.close(fig)
print("research/figures/checkpoint_input_offload.png")
