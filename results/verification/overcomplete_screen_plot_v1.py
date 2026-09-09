"""H056 figures from retained artifacts; run separately from training."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("results/overcomplete_screen_v1")
    result = json.loads((root / "result.json").read_text())
    pre = json.loads((root / "preflight.json").read_text())
    assert result["status"] == "complete"
    rows = pre["controls"] + result["trials"]
    records = []
    inputs = {}
    for row in rows:
        p = Path("results/runs") / row["run"]
        for name in ("metrics.json", "history.jsonl"):
            inputs[(p / name).as_posix()] = hashlib.sha256((p / name).read_bytes()).hexdigest()
        records.append(
            (
                row,
                json.loads((p / "metrics.json").read_text()),
                [json.loads(line) for line in (p / "history.jsonl").read_text().splitlines()],
            )
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    labels = {
        "full_swiglu": "Full SwiGLU",
        "full_gelu": "Full GELU",
        "calibrated_narrow": "Calibrated narrow",
        "blockshuffle": "Plain BlockShuffle",
        "headwise": "Square headwise",
        "overcomplete_headwise": "Overcomplete headwise",
    }
    colors = {
        "full_swiglu": "#1f77b4",
        "full_gelu": "#9467bd",
        "calibrated_narrow": "#7f7f7f",
        "blockshuffle": "#2ca02c",
        "headwise": "#9c755f",
        "overcomplete_headwise": "#d1495b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for recipe, label in labels.items():
        cells = sorted([r for r in records if r[0]["recipe"] == recipe], key=lambda r: r[0]["rate"])
        axes[0].plot(
            [r[0]["rate"] for r in cells],
            [r[1]["validation_loss"] for r in cells],
            marker="o",
            color=colors[recipe],
            label=label,
            linewidth=2.5 if recipe == "overcomplete_headwise" else 1.5,
        )
    selected = {**result["references"], "overcomplete_headwise": result["selected"]}
    for recipe, run in selected.items():
        _, metrics, history = next(r for r in records if r[0]["run"] == run)
        axes[1].plot(
            [0] + [r["step"] for r in history],
            [metrics["initial_validation_loss"]] + [r["validation_loss"] for r in history],
            color=colors[recipe],
            marker=".",
            linewidth=2.5 if recipe == "overcomplete_headwise" else 1.5,
        )
    axes[0].set(
        xscale="log",
        xlabel="Peak learning rate (three fixed choices)",
        ylabel="Final validation NLL (lower is better)",
        title="Equal three-rate selection budget",
    )
    axes[0].set_xticks([0.0003, 0.0006, 0.0012], ["0.0003", "0.0006", "0.0012"])
    axes[0].minorticks_off()
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Optimizer steps",
        ylabel="Validation NLL (lower is better)",
        title="Each recipe at its selected rate",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "H056: WikiText-2 development screen, seed 17\n200 steps; 409,600 training tokens; 322,688 validation targets per evaluation",
        fontsize=12,
    )
    outputs = {}
    for suffix in ("png", "svg"):
        p = Path(f"results/plots/overcomplete_screen.{suffix}")
        assert not p.exists()
        fig.savefig(p, dpi=160)
        outputs[p.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    plt.close(fig)
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest() == h for n, h in inputs.items())
    record = {
        "status": "PASS",
        "inputs": inputs,
        "outputs": outputs,
        "trials": len(records),
        "result_sha256": hashlib.sha256((root / "result.json").read_bytes()).hexdigest(),
    }
    p = Path("results/verification/overcomplete_screen_plot_v1.json")
    assert not p.exists()
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "trials": len(records), "outputs": outputs}))


if __name__ == "__main__":
    main()
