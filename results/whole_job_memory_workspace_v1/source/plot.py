"""Plot all six native-audited comparisons against their prospective gates."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/whole_job_memory_workspace_v1")


def run():
    summary = json.loads((ROOT / "summary.json").read_text())
    pairs = summary["pairs"]
    labels = [
        f"{'WikiText-2' if p['dataset'] == 'wikitext2' else 'TinyStories'}\nseed {p['seed']}"
        for p in pairs
    ]
    colors = ["#2767a2" if p["dataset"] == "wikitext2" else "#c66b25" for p in pairs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    specs = [
        ("native_nll_ratio", "Native validation NLL change (%)", 1, False),
        ("job_memory_ratio", "Whole-job tensor allocation saved (%)", 15, True),
        ("update_time_ratio", "Median update cost change (%)", 25, False),
    ]
    for ax, (key, title, gate, saving) in zip(axes, specs, strict=True):
        values = [100 * (1 - p[key] if saving else p[key] - 1) for p in pairs]
        y = np.arange(len(pairs))
        ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(
            gate,
            color="#9a2431",
            linestyle="--",
            linewidth=1.4,
            label=f"{'minimum' if saving else 'maximum'}: {gate}%",
        )
        ax.axvline(0, color="#777777", linewidth=0.7)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, pad=12)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        for yy, value in zip(y, values, strict=True):
            ax.text(
                value,
                yy,
                f" {value:+.2f}" if not saving else f" {value:.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        left, right = ax.get_xlim()
        ax.set_xlim(left - 0.02 * (right - left), right + 0.22 * (right - left))
    fig.suptitle(
        "H112: same narrow GELU, chunked training and shared streamed evaluation\n"
        "12 fresh trials · 800 updates each · lower NLL / lower memory / lower time preferred",
        fontsize=13,
    )
    output = Path("research/figures/whole_job_memory.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    run()
