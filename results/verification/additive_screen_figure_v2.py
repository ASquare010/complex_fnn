"""Plot recorded H061 and matched historical controls; never train or score."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/additive_screen_v2")
result = json.loads((root / "result.json").read_text())
pre = json.loads((root / "preflight.json").read_text())
recipes = {
    "additive_block_lowrank": ("Additive block + low rank", "#006f91"),
    "full_swiglu": ("Full SwiGLU", "#c24e00"),
    "full_gelu": ("Full GELU", "#8c438f"),
    "calibrated_narrow": ("Calibrated narrow", "#488037"),
    "blockshuffle": ("BlockShuffle", "#656565"),
}
rows = [*pre["controls"], *result["trials"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
for recipe, (label, color) in recipes.items():
    trials = sorted((r for r in rows if r["recipe"] == recipe), key=lambda r: r["rate"])
    metrics = [json.loads((Path("results/runs") / r["run"] / "metrics.json").read_text()) for r in trials]
    losses = [m["validation_loss"] for m in metrics]
    axes[0].plot([r["rate"] * 1000 for r in trials], losses, "o-", label=label, color=color)
    selected = min(zip(trials, metrics), key=lambda pair: (pair[1]["validation_loss"], pair[0]["rate"]))
    history = [json.loads(line) for line in (Path("results/runs") / selected[0]["run"] / "history.jsonl").read_text().splitlines()]
    axes[1].plot([r["step"] for r in history], [r["validation_loss"] for r in history], "o-", color=color, label=label)
axes[0].set(xlabel="Peak learning rate (x 0.001)", ylabel="Final validation NLL (lower is better)", title="All allocated rates, 200 updates")
axes[0].set_xticks([.3, .6, 1.2])
axes[1].set(xlabel="Training updates", ylabel="Validation NLL", title="Best rate per recipe: validation trajectory")
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=8)
fig.suptitle("WikiText-2 development screen | seed 17 | 322,688 validation targets", fontsize=12)
out = Path("research/figures")
out.mkdir(exist_ok=True)
for suffix in ("svg", "png"):
    target = out / f"additive_block_lowrank_screen.{suffix}"
    assert not target.exists()
    fig.savefig(target, dpi=170)
print(json.dumps({"status": "PASS", "plotted_trials": len(rows), "scoring_performed": False}))
