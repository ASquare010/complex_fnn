"""All-fixture classifier error, memory and runtime; preload without CUDA."""

import csv
import gzip
import io
from pathlib import Path

import sympy
import torch
import torch._dynamo


def run():
    print(f"CPU preload passed: sympy={sympy.__version__}, torch={torch.__version__}", flush=True)
    assert not torch.cuda.is_initialized()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path("results/classifier_precision_v1")
    rows = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((root / "metrics.csv.gz").read_bytes()).decode())
        )
    )
    policies = ("native_bf16", "chunks_cached", "chunks_uncached", "native_fp32", "chunks_fp32")
    labels = ("Native BF16", "Cached chunks", "Uncached chunks", "Native FP32", "FP32 chunks")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    specs = (
        ("weight_error", "Classifier-weight gradient relative error", True),
        ("memory_ratio_to_native_bf16", "Allocation / native BF16", False),
        ("time_ratio_to_native_bf16", "Forward + backward time / native BF16", False),
    )
    for ax, (key, title, log) in zip(axes, specs, strict=True):
        for dataset, offset, color, display in (
            ("wikitext2", -0.13, "#2767a2", "WikiText-2"),
            ("tinystories", 0.13, "#c66b25", "TinyStories"),
        ):
            for index, policy in enumerate(policies):
                values = np.array(
                    [
                        float(r[key])
                        for r in rows
                        if r["dataset"] == dataset and r["policy"] == policy
                    ]
                )
                yy = np.full(len(values), index + offset)
                ax.scatter(
                    values, yy, color=color, alpha=0.4, s=18, label=display if index == 0 else None
                )
                ax.scatter(
                    [np.median(values)],
                    [index + offset],
                    color=color,
                    marker="|",
                    s=190,
                    linewidths=2,
                )
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        if log:
            ax.set_xscale("log")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[1].axvline(0.85, color="#9a2431", linestyle="--", linewidth=1)
    axes[2].axvline(1.5, color="#9a2431", linestyle="--", linewidth=1)
    fig.suptitle(
        "H113: identical saved classifier inputs, zero optimizer updates\n"
        "Every fixture shown; thick markers are corpus medians. Lower is better.",
        fontsize=13,
    )
    target = Path("research/figures/classifier_precision.png")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    assert not torch.cuda.is_initialized()
    print(target)


if __name__ == "__main__":
    run()
