"""Standalone long-training figure; all seeds retained, SD is not a confidence interval."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/checkpoint_fp16_long_v1")
r = json.loads((ROOT / "result.json").read_text())
s = json.loads((ROOT / "summary.json").read_text())
colors = {"ordinary": "#536878", "fp16": "#007f82"}
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for row, dataset in enumerate(("wikitext2", "tinystories")):
    for arm, color in colors.items():
        cases = [
            v["measurement"]
            for v in r["cases"]
            if v["arm"] == arm and v["fixture"]["dataset"] == dataset
        ]
        assert len(cases) == 3
        curves = np.asarray(
            [
                [
                    c["initial_validation"]["nll"],
                    *[v["validation"]["nll"] for v in c["intermediate_checkpoints"]],
                    c["final_validation"]["nll"],
                ]
                for c in cases
            ]
        )
        mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        x = [0, 200, 400, 800]
        axes[row, 0].plot(x, mean, "o-", color=color, label=arm)
        axes[row, 0].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12)
        points = [v for v in s["metrics"] if v["dataset"] == dataset and v["arm"] == arm]
        axes[row, 1].scatter(
            [v["peak_mib"] for v in points], [v["nll"] for v in points], color=color, label=arm
        )
    axes[row, 0].set(
        title=dataset + ": validation (mean +/- SD)",
        xlabel="Updates",
        ylabel="Native validation NLL",
    )
    axes[row, 1].set(
        title="All three seeds", xlabel="Whole-job CUDA peak MiB", ylabel="Final validation NLL"
    )
    for i, control in enumerate(("ordinary",)):
        points = [
            v for v in s["comparisons"] if v["dataset"] == dataset and v["control"] == control
        ]
        axes[row, 2].scatter(
            [i + (j - 1) * 0.14 for j in range(3)],
            [v["ratios"]["event_ms"] for v in points],
            color=colors[control],
        )
        for j, v in enumerate(points):
            axes[row, 2].annotate(
                str(v["seed"]),
                (i + (j - 1) * 0.14, v["ratios"]["event_ms"]),
                xytext=(0, 6),
                ha="center",
                textcoords="offset points",
                fontsize=8,
            )
    axes[row, 2].axhline(1.15, color="crimson", linestyle="--", label="Acceptance limit")
    axes[row, 2].axhline(1, color="gray", linewidth=0.6)
    axes[row, 2].set(
        xticks=[0],
        xticklabels=["ordinary"],
        ylabel="Candidate / control CUDA time",
        title="Every paired seed",
    )
    axes[row, 2].set_xlim(-0.3, 0.3)
    axes[row, 0].legend()
    axes[row, 2].margins(y=0.15)
fig.suptitle("H160: FP16 checkpoint storage; 3 seeds x 2 corpora x 2 arms")
fig.savefig("research/figures/checkpoint_fp16_long.png", dpi=150)
plt.close(fig)
