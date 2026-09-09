"""Render the completed H057 fixed-probe diagnosis independently of model code."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("results/residual_geometry_v1")
    files = {
        str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("cpu.json", "gpu.json", "result.json")
    }
    cpu = json.loads((root / "cpu.json").read_text())["records"]
    gpu = json.loads((root / "gpu.json").read_text())["records"]
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "complete"
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for recipe, label in labels.items():
        row = cpu[recipe + "_final"]
        native = gpu[recipe + "_final"]
        color = colors[recipe]
        axes[0].plot(
            range(8),
            [row["ffns"][str(i)]["output_rms"] for i in range(8)],
            color=color,
            marker="o",
            label=label,
        )
        axes[1].plot(
            range(8),
            [
                row["norms"][f"blocks.{i}.ffn_norm"]["local_adjoint_gain"]["median"]
                for i in range(8)
            ],
            color=color,
            marker="o",
        )
        axes[1].plot(
            range(8),
            [
                native["norms"][f"blocks.{i}.ffn_norm"]["local_adjoint_gain"]["median"]
                for i in range(8)
            ],
            color=color,
            linestyle="none",
            marker="x",
            markersize=7,
        )
    axes[0].set(
        yscale="log",
        xlabel="Layer",
        ylabel="FFN output RMS on the fixed training probe",
        title="Large early-layer growth is real",
    )
    axes[1].set(
        yscale="log",
        xlabel="Layer",
        ylabel="Median local RMSNorm VJP gain",
        title="Local attenuation does not rank language quality",
    )
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xticks(range(8))
    fig.suptitle(
        "H057: selected 200-step checkpoints, 256 training-prefix targets, zero updates\nCircles: CPU FP32; crosses on right: native CUDA BF16 (nearly overlapping)",
        fontsize=12,
    )
    outputs = {}
    for ext in ("png", "svg"):
        p = Path(f"results/plots/residual_geometry.{ext}")
        assert not p.exists()
        fig.savefig(p, dpi=160)
        outputs[p.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    plt.close(fig)
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest() == h for n, h in files.items())
    record = {"status": "PASS", "inputs": files, "outputs": outputs, "optimizer_updates": 0}
    p = Path("results/verification/residual_geometry_plot_v1.json")
    assert not p.exists()
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
