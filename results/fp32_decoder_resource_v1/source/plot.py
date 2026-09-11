"""All-fixture resource and numerical results; static publication figure."""

import csv
import gzip
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("results/fp32_decoder_resource_v1")
rows = list(
    csv.DictReader(io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode()))
)
protocol = json.loads((ROOT / "protocol.json").read_text())
fixtures = [f["label"] for f in protocol["fixtures"]]
labels = []
for f in protocol["fixtures"]:
    if f["scope"].startswith("full"):
        labels.append("Full " + f["model_config"]["variant"].upper())
    else:
        labels.append(
            ("Wiki" if f["dataset"] == "wikitext2" else "Stories") + "\n" + f["source_policy"]
        )
policies = protocol["policies"]
colors = ["#7c8796", "#4695ce", "#16856c", "#9c72b7", "#d78936"]
names = [
    "BF16 native",
    "FP32 native / default",
    "FP32 chunks / default",
    "FP32 native / math",
    "FP32 chunks / math",
]
x = np.arange(6)
fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.6))
for i, policy in enumerate(policies):
    selected = [
        next(r for r in rows if r["fixture"] == f and r["policy"] == policy) for f in fixtures
    ]
    offset = (i - 2) * 0.035
    axes[0, 0].plot(
        x + offset,
        [float(r["peak_job_mib"]) for r in selected],
        "o-",
        color=colors[i],
        lw=1.4,
        ms=5,
        label=names[i],
    )
    if policy != "bf16_default_native":
        axes[0, 1].plot(
            x + offset,
            [float(r["wall_ratio_bf16"]) for r in selected],
            "o-",
            color=colors[i],
            lw=1.4,
            ms=5,
        )
    if policy.endswith("chunks"):
        axes[1, 0].plot(
            x,
            [float(r["paired_global_gradient_error"]) for r in selected],
            "o-",
            color=colors[i],
            lw=1.8,
        )
        axes[1, 1].plot(
            x,
            [100 * (float(r["nll_ratio_bf16"]) - 1) for r in selected],
            "o-",
            color=colors[i],
            lw=1.8,
        )
axes[0, 0].set(
    title="Full-job tensor allocation",
    ylabel="Peak allocated MiB (optimizer and evaluation included)",
)
axes[0, 1].axhline(1, color="#7c8796", lw=0.8)
axes[0, 1].axhline(1.25, color="#bc4450", ls="--", lw=1.1)
axes[0, 1].set(
    title="Warmed update cost versus BF16 native",
    ylabel="Wall-time ratio; dashed line = 1.25 limit",
)
axes[1, 0].axhline(0.002, color="#bc4450", ls="--", lw=1.1)
axes[1, 0].set(
    yscale="log",
    title="Chunked versus native FP32 initial gradients",
    ylabel="Global relative L2; dashed line = 0.002 limit",
)
axes[1, 1].axhline(0, color="#7c8796", lw=0.8)
axes[1, 1].set(
    title="Common-scorer NLL after 50 updates",
    ylabel="Change versus BF16 native (%)\nEnlarged scale; acceptance limit is +1%",
)
for ax in axes.flat:
    ax.set_xticks(x, labels, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "H116 · Does FP32 gradient fidelity survive the full-job resource test?",
    fontsize=17,
    x=0.05,
    ha="left",
    y=0.98,
)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.94),
    ncol=3,
    frameon=False,
    fontsize=10,
)
fig.text(
    0.05,
    0.024,
    "Six heterogeneous saved fixtures, including correlated narrow-model pairs; not six independent seeds.\n"
    "30 continuations / 1,500 updates. Common streamed BF16 evaluation. Allocations exclude driver/context memory.",
    fontsize=9,
    color="#555",
)
fig.subplots_adjust(left=0.085, right=0.98, top=0.83, bottom=0.13, hspace=0.42, wspace=0.24)
fig.savefig("research/figures/fp32_decoder_resource.png", dpi=150)
plt.close(fig)
