"""Compare optimizer-phase and complete-job peaks from audited profiles."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from results.optimizer_memory_v1.source.prepare import ROOT, read  # noqa: E402

p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.25, wspace=0.28)
x = np.arange(4)
for j, (mode, color) in enumerate(
    (("default", "#006D77"), ("single", "#C26A2F"), ("fused", "#7B61A8"))
):
    cases = [
        next(c for c in r["cases"] if c["fixture"] == f and c["mode"] == mode)
        for f in p["fixtures"]
    ]
    optimizer = [
        max(m["peak_allocated_bytes"] for m in c["phases"] if m["phase"].startswith("optimizer_"))
        / 2**20
        for c in cases
    ]
    whole = [c["peak_allocated_bytes"] / 2**20 for c in cases]
    for ax, values in zip(axes, (optimizer, whole), strict=True):
        ax.bar(x + (j - 1) * 0.24, values, width=0.22, label=mode, color=color)
axes[0].set_title("A   Optimizer peak falls", loc="left", fontweight="bold")
axes[1].set_title("B   Complete-job peak does not fall", loc="left", fontweight="bold")
for ax in axes:
    ax.set_ylabel("Peak allocated MiB")
    ax.set_xticks(
        x, ["WikiText\nnative", "WikiText\nchunks", "TinyStories\nnative", "TinyStories\nchunks"]
    )
    ax.set_ylim(0, 460 if ax == axes[1] else 260)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].legend(ncol=3, fontsize=9, loc="upper center")
axes[1].text(
    0.5,
    0.9,
    "Backward sets all 12 complete-job peaks",
    ha="center",
    transform=axes[1].transAxes,
    fontsize=9,
)
fig.suptitle(
    "H120 · Smaller optimizer temporaries do not solve this model's VRAM limit",
    x=0.08,
    ha="left",
    fontsize=14,
    fontweight="bold",
)
fig.text(
    0.08,
    0.84,
    "Two trained checkpoints · 12 matched continuations · 360 updates · all independent audits pass",
    fontsize=10,
)
fig.text(
    0.08,
    0.09,
    "All warmup, evaluation, diagnostics and serialization phases count. Allocated memory excludes driver/context memory.",
    fontsize=9,
)
fig.text(
    0.08,
    0.045,
    "Single/fused both fail the fixed 10% whole-job memory gate. Native fused is 25 KiB higher due to additional CUDA storage.",
    fontsize=9,
)
fig.savefig("research/figures/optimizer_memory.png", dpi=170, facecolor="white")
plt.close(fig)
print("research/figures/optimizer_memory.png")
