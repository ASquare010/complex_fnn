"""Static all-fixture gradient transport, repeat spread and diagnostic memory."""

# ruff: noqa: I001 -- retain scientific runtime preloading order
import sympy  # noqa: F401
import torch  # noqa: F401
import torch._dynamo  # noqa: F401

import csv
import gzip
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

root = Path("results/decoder_gradient_transport_v1")


def read_table(name):
    return list(
        csv.DictReader(
            io.StringIO(gzip.decompress((root / (name + ".csv.gz")).read_bytes()).decode())
        )
    )


rows, traces = read_table("metrics"), read_table("traces")
modes = (
    "bf16_default_block",
    "bf16_default_none",
    "bf16_math_block",
    "fp32_default_block",
    "fp32_math_block",
)
names = ("BF16 / default", "BF16 / no checkpoint", "BF16 / math", "FP32 / default", "FP32 / math")
short = ("BF16\ndefault", "BF16\nno ckpt", "BF16\nmath", "FP32\ndefault", "FP32\nmath")
colors = ("#bb5b35", "#d69c35", "#ab6eaa", "#087f80", "#397bb3")
boundaries = (
    "normalized_hidden",
    "block_7",
    "block_6",
    "block_5",
    "block_4",
    "block_3",
    "block_2",
    "block_1",
    "block_0",
    "embedding",
)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
for index, mode in enumerate(modes):
    values = [
        [float(r["relative_l2"]) for r in traces if r["mode"] == mode and r["boundary"] == b]
        for b in boundaries
    ]
    medians = [np.median(v) for v in values]
    axes[0, 0].plot(range(10), medians, color=colors[index], label=names[index], marker="o", ms=3)
    axes[0, 0].fill_between(
        range(10),
        [min(v) for v in values],
        [max(v) for v in values],
        color=colors[index],
        alpha=0.08,
    )
    selected = [r for r in rows if r["mode"] == mode]
    for axis, key in (
        (axes[0, 1], "paired_total_max"),
        (axes[1, 0], "maximum_replay_global"),
        (axes[1, 1], "peak_diagnostic_mib"),
    ):
        vals = [float(r[key]) for r in selected]
        if key == "maximum_replay_global":
            vals = [max(v, 1e-10) for v in vals]
        x = index + np.linspace(-0.10, 0.10, len(vals))
        axis.scatter(x, vals, color=colors[index], s=36, alpha=0.9, zorder=3)
        axis.plot([index - 0.17, index + 0.17], [np.median(vals)] * 2, color=colors[index], lw=2)
for axis in axes.flat:
    axis.grid(axis="y", alpha=0.15)
axes[0, 0].set_yscale("log")
axes[0, 0].set_xticks(range(10), ("dH", "B7", "B6", "B5", "B4", "B3", "B2", "B1", "B0", "Emb"))
axes[0, 0].set_title(
    "Gradient difference grows through the BF16 decoder", loc="left", weight="bold", pad=12
)
axes[0, 0].set_ylabel("Relative L2 difference")
axes[0, 0].set_xlabel("Backward direction: classifier → embedding")
axes[0, 1].set_title(
    "Worst paired total-gradient error per fixture", loc="left", weight="bold", pad=12
)
axes[0, 1].set_yscale("log")
axes[0, 1].axhline(0.002, color="#a03b42", ls="--", lw=1)
axes[1, 0].set_title("Worst same-input replay error per fixture", loc="left", weight="bold", pad=12)
axes[1, 0].set_yscale("log")
axes[1, 0].axhline(1e-5, color="#a03b42", ls="--", lw=1)
axes[1, 1].set_title(
    "Diagnostic tensor peak, MiB — no optimizer", loc="left", weight="bold", pad=12
)
for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
    axis.set_xticks(range(5), short)
    axis.margins(x=0.08, y=0.15)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.91), ncol=5, frameon=False)
fig.suptitle(
    "Decoder gradient transport: precision and replay are separate effects",
    x=0.065,
    ha="left",
    fontsize=16,
    weight="bold",
    y=0.98,
)
fig.text(
    0.065,
    0.033,
    "6 saved model fixtures × 5 modes × 2 fixed incoming gradients × 4 repeats. Zero optimizer updates.\n"
    "Trace bands show the full fixture/repeat range; dots show every fixture. Zero replay errors appear at 1e−10.\n"
    "Hooks can affect scheduling. These are diagnostic memory measurements, not training-job or inference claims.",
    fontsize=10,
    color="#4b5563",
)
fig.subplots_adjust(left=0.065, right=0.98, top=0.82, bottom=0.16, hspace=0.4, wspace=0.27)
path = Path("research/figures/decoder_gradient_transport.png")
fig.savefig(path, dpi=160, facecolor="white")
print(path.as_posix())
