"""Render H064 recorded peak allocations only; no model execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/rational_memory_v1")
r = json.loads((root / "result.json").read_text())
assert r["attribution_valid"]
labels = ["Plain BlockShuffle", "Rational BlockShuffle"]
groups = [
    ("Parameters, moments, buffers and input", ("model_parameters", "optimizer_moments", "model_buffers", "synthetic_input"), "#687782", None),
    ("Other preexisting allocation", ("other_preexisting",), "#c9d0d5", None),
    ("Norm, attention and projection sites", ("norm", "attention", "projection"), "#218487", None),
    ("Rational activation sites", ("rational_pointwise",), "#dc882d", None),
    ("Runtime allocation without matched origin", ("other_runtime", "unattributed_native_backward"), "#aa90bb", "///"),
]
fig, ax = plt.subplots(figsize=(11, 4.6))
left = [0.0, 0.0]
for label, keys, color, hatch in groups:
    values = [sum(r["analyses"][n]["category_bytes"].get(k, 0) for k in keys) / 2**20 for n in ("plain", "rational")]
    ax.barh([1, 0], values, left=left, height=.5, color=color, label=label, hatch=hatch, edgecolor="white")
    for y, start, value in zip([1, 0], left, values):
        if value >= 30:
            ax.text(start + value / 2, y, f"{value:.1f}", va="center", ha="center", fontsize=9, color="white" if color == "#687782" else "#151515")
    left = [a + b for a, b in zip(left, values)]
for y, total in zip([1, 0], left):
    ax.text(total + 5, y, f"{total:.3f} MiB", va="center", fontsize=10)
ax.set_yticks([1, 0], labels)
ax.set_xlim(0, 535)
ax.set_ylim(-.5, 1.5)
ax.set_xlabel("Live allocated GPU blocks at each model's own peak (MiB)")
ax.set_title("Rational activation memory diagnostic | whole-block native recomputation", pad=28)
ax.text(.5, 1.04, "One synthetic backward pass per measured worker; no optimizer updates", transform=ax.transAxes, ha="center", fontsize=9)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", alpha=.2)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, fontsize=8, frameon=False)
fig.subplots_adjust(left=.19, right=.99, top=.80, bottom=.35)
for suffix in ("png", "svg"):
    out = Path(f"research/figures/rational_memory.{suffix}")
    assert not out.exists()
    fig.savefig(out, dpi=160)
print(json.dumps({"status": "PASS", "recorded_peak_MiB": left, "new_model_execution": False}))
