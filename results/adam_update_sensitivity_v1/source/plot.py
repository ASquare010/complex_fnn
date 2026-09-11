"""Static scientific figure from audited measurements only."""

import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from results.adam_update_sensitivity_v1.source.prepare import ROOT, read  # noqa: E402

a, summary = read(ROOT / "analysis.json"), read(ROOT / "summary.json")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw={"width_ratios": [1.4, 1, 1.1]})
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.26, top=0.78, wspace=0.36)
for same, label, color, offset in (
    (False, "Cross-policy (9 pairs)", "#006D77", -0.045),
    (True, "Within-policy (6 pairs)", "#C26A2F", 0.045),
):
    pairs = [p for p in a["pairs"] if p["epsilon"] == 1e-8 and p["same_policy"] == same]
    native = [p for p in a["native_pairs"] if p["same_policy"] == same]
    values = [
        [p["raw_gradient_distance"] for p in native],
        [p["clipped_distance"] for p in pairs],
        [p["direction_distance"] for p in pairs],
        [p["realized_update_distance"] for p in native],
        [p["parameter_distance"] for p in native],
    ]
    center = [st.median(v) for v in values]
    axes[0].errorbar(
        [i + offset for i in range(5)],
        center,
        yerr=[
            [m - min(v) for m, v in zip(center, values, strict=True)],
            [max(v) - m for m, v in zip(center, values, strict=True)],
        ],
        marker="o",
        color=color,
        label=label,
        capsize=3,
        linewidth=1.4,
    )
axes[0].set_yscale("log")
axes[0].set_xticks(
    range(5),
    ["Raw\ngradient", "Clipped\ngradient", "Ideal\ndirection", "Stored\nupdate", "Stored\nweights"],
)
axes[0].set_ylabel("Symmetric relative L2 (log scale)")
axes[0].set_title("A   First-step differences", loc="left", fontweight="bold")
axes[0].legend(fontsize=8, loc="lower left")
axes[0].grid(axis="y", alpha=0.2)
cross = [p for p in a["pairs"] if p["epsilon"] == 1e-8 and not p["same_policy"]]
shares = [100 * st.mean(p["bin_error_shares"][i] for p in cross) for i in range(4)]
axes[1].bar(range(4), shares, color=["#006D77", "#83C5BE", "#BBBBBB", "#DDDDDD"])
for i, v in enumerate(shares):
    axes[1].text(i, v + 2, f"{v:.2f}%", ha="center", fontsize=9)
axes[1].set_ylim(0, 108)
axes[1].set_xticks(range(4), ["≤1ε", "1–10ε", "10–100ε", ">100ε"])
axes[1].set_ylabel("Share of squared direction difference (%)")
axes[1].set_xlabel("max(|gradient a|, |gradient b|)")
axes[1].set_title("B   Near-zero coordinates dominate", loc="left", fontweight="bold")
points = [(0.0, 1.0, "1e−8")] + [
    (100 * c["max_direction_distortion"], c["max_error_ratio"], f"{c['epsilon']:.0e}")
    for c in a["conditioning"]
]
axes[2].plot([p[0] for p in points], [p[1] for p in points], "o-", color="#006D77")
for x, y, label in points:
    axes[2].annotate(label, (x, y), xytext=(6, 7), textcoords="offset points", fontsize=9)
axes[2].axvline(1, color="#C26A2F", linestyle="--", linewidth=1)
axes[2].axhline(0.5, color="#C26A2F", linestyle="--", linewidth=1)
axes[2].set_xlim(-1.5, 40)
axes[2].set_ylim(0, 1.13)
axes[2].set_xlabel("Worst direction distortion (%)")
axes[2].set_ylabel("Worst cross-pair error / baseline")
axes[2].set_title("C   Epsilon changes the update", loc="left", fontweight="bold")
fig.suptitle(
    "H119 · Adam amplifies tiny initial differences; larger epsilon is not a free fix",
    x=0.07,
    ha="left",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.07,
    0.83,
    "One selected WikiText seed · six saved probe gradients · no new language training",
    fontsize=10,
)
fig.text(
    0.07,
    0.09,
    "A: medians and full observed ranges. B: mean across nine dependent cross-policy pairs. C: fixed limits are dashed.",
    fontsize=9,
)
fig.text(
    0.07,
    0.045,
    f"All numerical gates passed: {summary['all_numerical_gates_passed']}. See the report for CPU clipping failures and preserved startup recovery.",
    fontsize=9,
)
path = Path("research/figures/adam_update_sensitivity.png")
fig.savefig(path, dpi=170, facecolor="white")
plt.close(fig)
print(path.as_posix())
