"""Every repetition and trajectory distance for the selected failed fixture."""

import csv
import faulthandler
import gzip
import io
import json
from pathlib import Path

faulthandler.dump_traceback_later(45, repeat=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("results/training_variability_v1")
summary, audit = [json.loads((ROOT / name).read_text()) for name in ("summary.json", "audit.json")]
rows = list(
    csv.DictReader(io.StringIO(gzip.decompress((ROOT / "curves.csv.gz").read_bytes()).decode()))
)
policies = ["fp32_default_native", "fp32_default_chunks"]
colors = ["#487ca5", "#10856e"]
names = ["Native FP32", "Chunked FP32"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
for i, policy in enumerate(policies):
    values = [
        next(
            float(r["nll"])
            for r in rows
            if r["policy"] == policy and int(r["repetition"]) == repeat and int(r["step"]) == 800
        )
        for repeat in (0, 1, 2)
    ]
    axes[0, 0].plot(range(3), values, "o-", color=colors[i], label=names[i])
    for repeat in (1, 2):
        drift = []
        for step in (0, 200, 400, 800):
            peers = [r for r in rows if r["policy"] == policy and int(r["step"]) == step]
            new = next(float(r["nll"]) for r in peers if int(r["repetition"]) == repeat)
            base = next(float(r["nll"]) for r in peers if int(r["repetition"]) == 0)
            drift.append(new - base)
        axes[1, 0].plot(
            [0, 200, 400, 800],
            drift,
            marker="o",
            ls="-" if repeat == 1 else "--",
            color=colors[i],
            label=names[i] + f" r{repeat}",
        )
axes[0, 1].bar(
    range(3),
    [100 * (p["ratio"] - 1) for p in summary["paired"]],
    color=["#8a939c", "#10856e", "#487ca5"],
    width=0.5,
)
axes[0, 1].axhline(1, color="#b84950", ls="--", lw=1)
axes[0, 1].axhline(0, color="#888", lw=0.8)
label_policy = {r["label"]: r["policy"] for r in rows}
for name, color, kind in (
    ("Within native", colors[0], "native"),
    ("Within chunks", colors[1], "chunks"),
    ("Across policies", "#9b6bb7", "cross"),
):
    medians, lows, highs = [], [], []
    for step in (200, 400, 800):
        selected = [
            r["model"]
            for r in audit["checkpoint_pairs"]
            if r["step"] == step
            and (
                (not r["same_policy"])
                if kind == "cross"
                else (r["same_policy"] and label_policy[r["left"]].endswith(kind))
            )
        ]
        medians.append(float(np.median(selected)))
        lows.append(min(selected))
        highs.append(max(selected))
    axes[1, 1].plot([200, 400, 800], medians, "o-", color=color, label=name)
    axes[1, 1].fill_between([200, 400, 800], lows, highs, color=color, alpha=0.12)
axes[0, 0].set(title="Every final NLL", ylabel="Common-scorer validation NLL")
axes[0, 1].set(
    title="Chunked / native within each repetition", ylabel="Final NLL change (%); dashed = +1%"
)
axes[1, 0].set(
    title="Drift from each policy's original run",
    xlabel="Optimizer updates",
    ylabel="Validation NLL difference",
)
axes[1, 0].axhline(0, color="#888", lw=0.8)
axes[1, 1].set(
    title="Model distance: median and observed range",
    xlabel="Optimizer updates",
    ylabel="Symmetric relative L2",
)
for ax in axes[0]:
    ax.set_xticks(range(3), ["Original r0", "New r1", "New r2"])
for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
    ax.legend(frameon=False, fontsize=9)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "H118 · Can repeat variability account for the failed seed's apparent gap?",
    fontsize=16,
    x=0.07,
    ha="left",
    y=0.98,
)
fig.text(
    0.07,
    0.025,
    "One deliberately selected WikiText seed; identical initial state and batches, two new repeats per policy.\nFour new 800-update runs plus two original runs. Ranges are descriptive, not confidence intervals. H117's failed gate remains failed.",
    fontsize=9,
    color="#555",
)
fig.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.13, hspace=0.36, wspace=0.26)
fig.savefig("research/figures/training_variability.png", dpi=150)
plt.close(fig)
faulthandler.cancel_dump_traceback_later()
