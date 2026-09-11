"""Scientific summary of the audited selection frontier; no Torch import."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("results/sobolev_selection_v1")
summary = json.loads((root / "summary.json").read_text())
ablation = json.loads((root / "ablation.json").read_text())
groups = {(r["teacher"], r["budget"], r["method"]): r for r in summary["groups"]}
budgets = ("three_quarter", "half", "primary")
colors = {"gelu": "#137e8a", "swiglu": "#be632f"}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8))
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.12, top=0.845, wspace=0.28, hspace=0.43)
fig.suptitle(
    "Derivative-aware calibration helps; the full compression gate still fails",
    x=0.075,
    ha="left",
    y=0.97,
    fontsize=15.5,
    weight="bold",
)
fig.text(
    0.075,
    0.919,
    "432 compressed FFNs · zero SGD · two fixed teachers · three depths · three sampling seeds · reused development inputs",
    fontsize=9.5,
    color="#52616b",
)
for column, metric, title, threshold in (
    (0, "value_nmse", "A. Output reconstruction", 0.05),
    (1, "derivative_relative_mse", "B. Directional-derivative reconstruction", 0.1),
):
    ax = axes[0, column]
    for teacher, color in colors.items():
        for method, style, label in (
            ("value_greedy", "--", "value-only"),
            ("sobolev_greedy", "-", "derivative-aware"),
        ):
            data = [groups[teacher, b, method] for b in budgets]
            ax.plot(
                [r["parameter_reduction"] * 100 for r in data],
                [r[metric]["mean"] for r in data],
                style,
                marker="o",
                color=color,
                label=f"{teacher.upper()} / {label}",
            )
    ax.axhline(threshold, ls=":", color="#555555", lw=1)
    ax.set(
        xlabel="FFN parameter reduction (%)",
        ylabel="Normalized error (mean of 9 cells)",
        title=title,
        yscale="log",
        xlim=(20, 75),
    )
    if column == 0:
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(
            0.02,
            0.95,
            "Dotted lines: separate absolute allocation gates",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
ax = axes[1, 0]
labels, combined, readout = [], [], []
for teacher in colors:
    for budget in budgets:
        data = next(
            r for r in ablation["rows"] if (r["teacher"], r["budget"]) == (teacher, budget)
        )["metrics"]["derivative_relative_mse"]
        percent = groups[teacher, budget, "sobolev_greedy"]["parameter_reduction"] * 100
        labels.append(f"{teacher.upper()}\n{percent:.0f}% fewer")
        combined.append(100 * (1 - data["both"] / data["baseline"]))
        readout.append(100 * (1 - data["readout_only"] / data["baseline"]))
x = np.arange(6)
ax.bar(x - 0.18, combined, 0.34, color="#3b648b", label="Selection + readout")
ax.bar(x + 0.18, readout, 0.34, color="#9db8cd", label="Readout objective only")
ax.set(
    xticks=x,
    xticklabels=labels,
    ylabel="Derivative error reduction (%)",
    title="C. Most improvement comes from the readout",
    ylim=(0, 21),
)
ax.legend(fontsize=8, loc="upper right")
ax = axes[1, 1]
for i, (teacher, color) in enumerate(colors.items()):
    values = [ablation["full_reference"][teacher]["peak_allocated_bytes"]["mean"] / 2**20]
    values += [
        groups[teacher, b, "sobolev_greedy"]["peak_allocated_bytes"]["mean"] / 2**20
        for b in budgets
    ]
    ax.bar(
        np.arange(4) + (i - 0.5) * 0.36,
        values,
        0.34,
        color=color,
        label=f"{teacher.upper()} teacher",
    )
ax.set(
    xticks=np.arange(4),
    xticklabels=("Full", "~25% fewer", "~50% fewer", "~71% fewer"),
    ylabel="Peak allocated MiB",
    title="D. Raw-input inference memory, batch 256",
    ylim=(0, 20.5),
)
ax.legend(fontsize=8, loc="upper right")
fig.text(
    0.075,
    0.034,
    "The derivative objective is used only during calibration. Deployment uses ordinary dense FFNs. Memory is tensor allocation, not whole-process VRAM.",
    fontsize=8.8,
    color="#52616b",
)
fig.text(
    0.075,
    0.014,
    f"Preprocessing peak: {summary['preprocessing_peak_cuda_bytes'] / 2**20:.1f} MiB; including prerequisite teacher capture: {summary['pipeline_peak_cuda_bytes'] / 2**20:.1f} MiB. Training memory and NLL are untested.",
    fontsize=8.8,
    color="#52616b",
)
output = Path("research/figures/sobolev_selection.png")
fig.savefig(output, dpi=160)
print(output)
