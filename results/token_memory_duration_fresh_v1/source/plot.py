"""Plot every completed paired seed; separate training from whole-job allocation."""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/token_memory_duration_fresh_v1")
summary = json.loads((ROOT / "summary.json").read_text())
result = json.loads((ROOT / "result.json").read_text())
rows = result["cases"]
contexts = sorted({g["context"] for g in summary["groups"]})
colors = {"block": "#485b78", "loss_chunks": "#089b88"}
names = {"block": "Block checkpoint", "loss_chunks": "+ Loss chunks"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
for row in rows:
    axis = axes[0, 0]
    steps = [v["step"] for v in row["evaluations"]]
    nll = [v["nll"] for v in row["evaluations"]]
    axis.plot(
        steps,
        nll,
        color=colors[row["policy"]],
        linestyle="-" if row["context"] == 128 else "--",
        alpha=0.72,
        marker=".",
        label=f"{names[row['policy']]} T{row['context']}" if row["seed"] == 17 else None,
    )
axes[0, 0].set(title="Fixed development subset: all completed runs", xlabel="Update", ylabel="NLL")
axes[0, 0].legend(fontsize=8)
for pair in summary["comparisons"]:
    values = pair["trajectory_relative_nll_changes"]
    axes[0, 1].plot(
        [int(k) for k in values],
        [100 * v for v in values.values()],
        marker="o",
        label=f"T{pair['context']} seed{pair['seed']}",
    )
axes[0, 1].axhline(1, color="#c94a57", linestyle="--", linewidth=1)
axes[0, 1].axhline(-1, color="#c94a57", linestyle="--", linewidth=1)
axes[0, 1].axhline(0, color="gray", linewidth=0.6)
axes[0, 1].set(
    title="Paired NLL change; frozen band +/-1%", xlabel="Update", ylabel="Loss chunks vs block (%)"
)
axes[0, 1].legend(fontsize=8, ncol=2)
labels, train, job, time, barcolors = [], [], [], [], []
for context in contexts:
    for policy in ("block", "loss_chunks"):
        peers = [r for r in rows if r["context"] == context and r["policy"] == policy]
        labels.append(f"T{context}\n{names[policy]}")
        train.append(st.mean(r["peak_training_allocated_bytes"] for r in peers) / 2**20)
        job.append(st.mean(r["peak_job_allocated_bytes"] for r in peers) / 2**20)
        time.append(st.mean(r["timing"]["update_ms"]["median"] for r in peers))
        barcolors.append(colors[policy])
x = np.arange(len(labels))
axes[1, 0].bar(x, job, color=barcolors, alpha=0.25, label="Whole measured job")
axes[1, 0].bar(x, train, color=barcolors, label="Training")
for i, (a, b) in enumerate(zip(train, job, strict=True)):
    axes[1, 0].text(i, b + 4, f"{a:.0f} / {b:.0f}", ha="center", fontsize=9)
axes[1, 0].set_xticks(x, labels, fontsize=8)
axes[1, 0].set(
    title="Allocated tensor memory: mean of completed seeds", ylabel="MiB (training / whole job)"
)
axes[1, 0].set_ylim(0, max(job) * 1.22)
axes[1, 0].legend(fontsize=8)
axes[1, 1].bar(x, time, color=barcolors)
for i, v in enumerate(time):
    axes[1, 1].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
axes[1, 1].set_xticks(x, labels, fontsize=8)
axes[1, 1].set(title="Mean of seed median update times", ylabel="Milliseconds / update")
axes[1, 1].set_ylim(0, max(time) * 1.2)
fig.suptitle(
    "H110: loss chunking through 800 updates", x=0.07, ha="left", fontsize=18, weight="bold"
)
fig.text(
    0.07,
    0.945,
    f"{len(rows)} completed trials; {result['updates']:,} updates. Same 9.10M-parameter model. {result['status']}",
    fontsize=10,
)
fig.text(
    0.07,
    0.025,
    "RTX 4070 Laptop / BF16. WikiText-2 development data. Allocation excludes driver/process overhead.\n"
    "Original seed17/T128 control reused after recorded runtime failures; no scientific training repeated.",
    fontsize=9,
    color="#465366",
)
fig.tight_layout(rect=(0.04, 0.065, 0.99, 0.92), h_pad=2.3)
destination = Path("research/figures/token_memory_duration.png")
fig.savefig(destination, dpi=160, facecolor="white")
plt.close(fig)
print(destination)
