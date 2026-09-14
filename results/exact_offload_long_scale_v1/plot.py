"""Plot all H163 seeds and descriptive training variation; no selection."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("results/exact_offload_long_scale_v1")
s = json.loads((root / "summary.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
colors = {"ordinary": "#335eaa", "helper": "#df7a24"}
for arm in ("ordinary", "helper"):
    runs = [r for r in s["runs"] if r["arm"] == arm]
    losses = np.array([[v["loss"] for v in r["records"]] for r in runs])
    smooth = np.array([np.convolve(v, np.ones(25) / 25, "valid") for v in losses])
    x = np.arange(25, 801)
    mean, std = smooth.mean(0), smooth.std(0, ddof=1)
    axes[0, 0].plot(x, mean, color=colors[arm], label=arm)
    axes[0, 0].fill_between(x, mean - std, mean + std, color=colors[arm], alpha=0.13)
    for r in runs:
        vals = [r["initial_score"]["nll"]] + [e["score"]["nll"] for e in r["endpoints"]]
        axes[0, 1].plot(
            [0, 200, 400, 800],
            vals,
            marker="o",
            ms=3,
            color=colors[arm],
            alpha=0.7,
            linestyle=["-", "--", ":"][(401, 409, 419).index(r["seed"])],
            label=f"{arm} {r['seed']}",
        )
    memory = [r["memory"]["allocated"] / 2**20 for r in runs]
    axes[1, 0].bar(arm, np.mean(memory), color=colors[arm], alpha=0.8)
    axes[1, 0].scatter([arm] * 3, memory, c="black", s=15)
    axes[1, 0].text(arm, np.mean(memory) + 10, f"{np.mean(memory):.1f}", ha="center")
for i, pair in enumerate(s["pairs"]):
    axes[1, 1].scatter(
        i - 0.06,
        pair["ratios"]["cuda"],
        marker="o",
        color="#335eaa",
        label="CUDA" if i == 0 else None,
    )
    axes[1, 1].scatter(
        i + 0.06,
        pair["ratios"]["wall"],
        marker="x",
        color="#df7a24",
        label="wall" if i == 0 else None,
    )
axes[1, 1].axhline(1, color="gray", lw=1)
axes[1, 1].axhline(1.15, color="red", ls="--", label="limit")
axes[1, 1].set_xticks(range(3), ["401", "409", "419"])
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
axes[0, 0].set(title="Training loss: 25-update moving average", xlabel="Update", ylabel="NLL")
axes[0, 0].legend(title="Mean ± sample SD, 3 seeds", fontsize=8)
axes[0, 1].set(title="Full validation: every seed", xlabel="Update", ylabel="NLL")
axes[0, 1].legend(fontsize=7, ncol=2)
axes[1, 0].set(title="Whole-job allocated GPU peak", ylabel="MiB")
axes[1, 0].set_ylim(0, axes[1, 0].get_ylim()[1] * 1.15)
axes[1, 1].set(title="Median complete-update ratio", xlabel="Seed", ylabel="Helper / ordinary")
axes[1, 1].legend(fontsize=8)
fig.suptitle("H163 · 22.2M parameters · FP32 · WikiText2 · 800 updates", fontsize=13)
fig.savefig("research/figures/exact_offload_long_scale.png", dpi=170)
plt.close(fig)
