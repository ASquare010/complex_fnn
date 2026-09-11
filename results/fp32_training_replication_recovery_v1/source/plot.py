"""Display every seed, fixed quality limits and the full common-scorer trajectory."""

import csv
import gzip
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("results/fp32_training_replication_recovery_v1")
rows = list(
    csv.DictReader(io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode()))
)
summary = json.loads((ROOT / "summary.json").read_text())
protocol = json.loads((ROOT / "protocol.json").read_text())
fixtures = [f["label"] for f in protocol["fixtures"]]
labels = [
    ("Wiki" if f["dataset"] == "wikitext2" else "Stories") + "\n" + str(f["seed"])
    for f in protocol["fixtures"]
]
policies = protocol["policies"]
names = ["BF16 native", "FP32 native", "FP32 chunks"]
colors = ["#778592", "#3c8bc3", "#13816b"]
x = np.arange(6)
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
for i, policy in enumerate(policies):
    selected = [
        next(r for r in rows if r["fixture"] == f and r["policy"] == policy) for f in fixtures
    ]
    axes[0, 0].plot(
        x + (i - 1) * 0.045,
        [float(r["peak_job_mib"]) for r in selected],
        "o-",
        color=colors[i],
        label=names[i],
        ms=5,
    )
    for j, corpus in enumerate(("wikitext2", "tinystories")):
        points = [
            c for c in summary["convergence"] if c["corpus"] == corpus and c["policy"] == policy
        ]
        steps = [c["step"] for c in points]
        means = np.array([c["statistics"]["mean"] for c in points])
        sd = np.sqrt([c["statistics"]["sample_variance"] for c in points])
        axes[1, j].plot(steps, means, "o-", color=colors[i], ms=4)
        axes[1, j].fill_between(steps, means - sd, means + sd, color=colors[i], alpha=0.12)
candidate = [
    next(r for r in rows if r["fixture"] == f and r["policy"] == "fp32_default_chunks")
    for f in fixtures
]
for reference, color, name in (
    ("bf16", "#13816b", "Chunks / BF16 native"),
    ("native", "#895dc2", "Chunks / FP32 native"),
):
    axes[0, 1].plot(
        x, [float(r["wall_ratio_" + reference]) for r in candidate], "o-", color=color, label=name
    )
    axes[0, 2].plot(
        x,
        [100 * (float(r["nll_ratio_" + reference]) - 1) for r in candidate],
        "o-",
        color=color,
    )
axes[0, 1].axhline(1, color="#777", lw=0.8)
axes[0, 1].axhline(1.25, color="#b84950", ls="--", lw=1)
axes[0, 2].axhline(0, color="#777", lw=0.8)
axes[0, 2].axhline(1, color="#b84950", ls="--", lw=1)
axes[1, 2].plot(
    x, [float(r["paired_global_gradient_error"]) for r in candidate], "o-", color="#13816b"
)
axes[1, 2].axhline(0.002, color="#b84950", ls="--", lw=1)
axes[0, 0].set(title="Peak full-job tensor allocation", ylabel="Allocated MiB")
axes[0, 1].set(title="Warmed update wall time", ylabel="Ratio; dashed line = 1.25 limit")
axes[0, 2].set(title="Quality after 800 updates", ylabel="NLL change (%); dashed line = +1% limit")
axes[1, 0].set(title="WikiText-2 convergence", xlabel="Optimizer updates", ylabel="Validation NLL")
axes[1, 1].set(title="TinyStories convergence", xlabel="Optimizer updates", ylabel="Validation NLL")
axes[1, 2].set(
    title="Initial FP32 chunk/native gradients",
    ylabel="Global relative L2; dashed line = 0.002 limit",
    yscale="log",
)
for ax in [*axes[0], axes[1, 2]]:
    ax.set_xticks(x, labels, fontsize=9)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 1].legend(loc="best", frameon=False, fontsize=8)
fig.suptitle(
    "H117 · Memory savings survive; does quality survive 800 updates?",
    fontsize=17,
    x=0.055,
    ha="left",
    y=0.98,
)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False
)
fig.text(
    0.055,
    0.025,
    "18 fresh runs; three seeds per corpus; identical 9,099,648 parameters. Each seed must satisfy every fixed gate.\n"
    "Common BF16 scoring. Convergence bands are ±1 sample SD, not confidence intervals. Allocation excludes driver/context memory.",
    fontsize=9,
    color="#555",
)
fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.13, hspace=0.42, wspace=0.29)
fig.savefig("research/figures/fp32_training_replication.png", dpi=150)
plt.close(fig)
