"""Standalone H109 figure; no Torch import or scientific rerun."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/sobolev_learning_v1")
summary = json.loads((root / "summary.json").read_text())
groups = {(g["teacher"], g["method"]): g for g in summary["groups"]}
methods = ("value_greedy", "value_select_sobolev_fit", "sobolev_greedy", "random")
labels = ("Value only", "Derivative readout", "Derivative both", "Random subset")
colors = ("#475569", "#0284c7", "#0f766e", "#d97706")
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
for col, teacher in enumerate(("gelu", "swiglu")):
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        g = groups[teacher, method]
        axes[0, col].plot(
            [0, 1],
            [g["initial"]["reporting_mse"]["mean"], g["reporting_mse"]["mean"]],
            "o-",
            color=color,
            label=label,
        )
        axes[1, col].plot(
            [0, 1],
            [g["initial"]["derivative_relative_mse"]["mean"], g["derivative_relative_mse"]["mean"]],
            "o-",
            color=color,
            label=label,
        )
    for row in (0, 1):
        axes[row, col].set_xticks([0, 1], ["Initialization", "Selected after fitting"])
        axes[row, col].set_xlim(-0.15, 1.15)
        axes[row, col].grid(alpha=0.2)
        axes[row, col].set_title(
            f"{teacher.upper()} — {'output error' if row == 0 else 'directional-derivative error'}"
        )
        axes[row, col].axhline(0.05 if row == 0 else 0.1, linestyle=":", color="#9f1239", alpha=0.8)
    axes[0, col].set_ylabel("MSE / calibration output variance")
    axes[1, col].set_ylabel("JVP error / teacher JVP energy")
axes[0, 0].legend(fontsize=9)
fig.suptitle(
    "H109: value fitting improves outputs but erodes derivative fidelity\n~71% fewer FFN parameters · 18 fresh input sets · 144 fits · 600 updates per fit",
    fontsize=13,
)
fig.savefig("research/figures/sobolev_learning.png", dpi=170)
plt.close(fig)
print("Saved research/figures/sobolev_learning.png")
