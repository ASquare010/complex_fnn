"""Plot audited JSON without loading Torch or checkpoints into the render process."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run():
    data = json.loads(Path("results/spectral_fitting_v1/summary.json").read_text())
    assert data["audit_passed"] and data["score_audits"] == 816
    forms = (
        "full_swiglu_random", "full_swiglu_bounded", "narrow_cubic_bounded",
        "tiny_cubic_bounded", "factor_cubic_random", "factor_cubic_bounded",
    )
    labels = ("Full SwiGLU\nrandom", "Full SwiGLU\nspectral", "Narrow cubic\nspectral",
              "Tiny cubic\nspectral", "Factor cubic\nrandom", "Factor cubic\nspectral")
    colors = ["#8994a8", "#416c9a", "#b48c59", "#239d86", "#b095af", "#7961a9"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=.085, right=.98, bottom=.115, top=.88, hspace=.56, wspace=.25)
    for ax, task in zip(axes.flat, ("quadratic", "cubic", "product", "piecewise"), strict=True):
        for i, form in enumerate(forms):
            row = data["per_task"][form][task]
            ax.bar(i, row["mean"], color=colors[i], width=.7)
            ax.scatter([i-.12, i, i+.12], row["values"], s=19, c="#202636", zorder=3)
        ax.set_xticks(range(len(forms)), labels, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("Reporting MSE (log scale)")
        ax.set_title(task.capitalize(), loc="left", fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Feature discovery helps; cubic still loses to a simpler control", fontsize=15, y=.965)
    fig.text(.5, .923, "Bars: mean; dots: three paired dataset seeds. Spectral = bounded training-label energy.", ha="center")
    fig.text(.5, .025, "300 updates per fit; two rates selected on a separate split. Synthetic Gaussian tasks; no language claim.", ha="center", fontsize=10)
    fig.savefig("research/figures/spectral_fitting.png", dpi=160)
    plt.close(fig)
    print("Audited fitting figure written", flush=True)


if __name__ == "__main__":
    run()
