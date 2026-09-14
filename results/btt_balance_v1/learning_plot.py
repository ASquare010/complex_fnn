"""Supplement H164 with validation-selected learning curves; raw evidence unchanged."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results.btt_balance_v1.model import ARMS
from results.btt_balance_v1.study import ROOT, read, sha, write

r = read(ROOT / "result.json")
fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
for task, ax in zip(("teacher", "product"), axes, strict=True):
    for arm in ARMS:
        curves = []
        for row in r["selected"]:
            if row["task"] == task and row["arm"] == arm:
                curves.append(
                    np.convolve([v["loss"] for v in row["trace"]], np.ones(25) / 25, mode="valid")
                )
        values = np.array(curves)
        assert values.shape == (3, 576)
        x = np.arange(25, 601)
        mean, sd = values.mean(0), values.std(0, ddof=1)
        (line,) = ax.plot(x, mean, label=arm, lw=1.5)
        ax.fill_between(x, mean - sd, mean + sd, color=line.get_color(), alpha=0.12)
    ax.set_title(task)
    ax.set_xlabel("Update")
    ax.set_ylabel("Training normalized MSE")
    ax.grid(alpha=0.2)
axes[1].legend(fontsize=8)
fig.suptitle("H164: validation-selected LR; all seeds; 25-update average ± sample SD")
path = Path("research/figures/btt_balance_learning.png")
fig.savefig(path, dpi=150)
plt.close(fig)
write(
    ROOT / "learning_figure.json",
    dict(
        figure=str(path),
        figure_sha256=sha(path),
        source_sha256=sha(Path(__file__)),
        input_sha256=sha(ROOT / "result.json"),
        optimizer_updates=0,
        selection="Validation-selected rate; all600updates and all3seeds per arm/task",
        smoothing=25,
        band="sample standard deviation, not confidence interval",
    ),
)
