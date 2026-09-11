"""Plot all paired ratios and the measured timing/clock chronology."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/paired_workspace_timing_v1")
s = json.loads((ROOT / "summary.json").read_text())
fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
for i, c in enumerate(s["comparisons"]):
    axes[0].scatter(
        [i - 0.07] * 4,
        [p["ratios"]["event_ms"] for p in c["pairs"]],
        color="#087e8b",
        label="CUDA event" if i == 0 else None,
    )
    axes[0].scatter(
        [i + 0.07] * 4,
        [p["ratios"]["wall_ms"] for p in c["pairs"]],
        color="#c75c24",
        marker="x",
        label="Wall" if i == 0 else None,
    )
axes[0].axhline(1.15, color="crimson", linestyle="--", label="Median acceptance limit")
axes[0].axhline(1, color="grey", linewidth=0.6)
axes[0].set_xticks(range(4), [c["dataset"] + " / " + c["control"] for c in s["comparisons"]])
axes[0].set_ylabel("Candidate / control ratio")
axes[0].legend(ncols=3)
rows = s["bursts"]
for arm, color in (("reuse8", "#087e8b"), ("native8", "#c75c24"), ("reuse32", "#7458a3")):
    ix = [i for i, r in enumerate(rows) if r["arm"] == arm]
    axes[1].scatter(
        ix, [rows[i]["timing"]["event_ms"]["median"] for i in ix], color=color, label=arm
    )
axes[1].set_ylabel("Median forward + backward ms")
axes[1].set_xlabel("Burst order (0-15 WikiText-2; 16-31 TinyStories)")
axes[1].legend(ncols=3)
axes[2].plot(
    range(32),
    [
        r["telemetry"]["sm_mhz"]["median"] if r["telemetry"]["sm_mhz"] else float("nan")
        for r in rows
    ],
    color="#087e8b",
    marker=".",
    label="SM clock",
)
axes[2].set_ylabel("Median SM clock MHz")
axes[2].set_xlabel("Burst order")
right = axes[2].twinx()
right.plot(
    range(32),
    [
        r["telemetry"]["power_w"]["median"] if r["telemetry"]["power_w"] else float("nan")
        for r in rows
    ],
    color="#c75c24",
    marker="x",
)
right.set_ylabel("Median board power W", color="#c75c24")
fig.suptitle("H133: balanced paired timing; no optimizer updates")
fig.savefig("research/figures/paired_workspace_timing.png", dpi=150)
plt.close(fig)
