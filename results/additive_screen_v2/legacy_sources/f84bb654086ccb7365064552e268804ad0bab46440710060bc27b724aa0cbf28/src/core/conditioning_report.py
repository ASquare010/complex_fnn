"""Reproduce the post-training condition-floor diagnostic and locked transfer."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.reproducibility import sha256, write_json


def report(root: Path = Path("results")) -> dict:
    names = ["conditioning_scale384_v1", *[f"conditioning_scale192_s{s}_v1" for s in (17, 29, 43)]]
    data = {name: json.loads((root / name / "result.json").read_text()) for name in names}
    result = {
        "source_sha256": {name: sha256(root / name / "result.json") for name in names},
        "numerical_bound_checks": {},
        "locked_floor_gates": {},
    }
    for name, d in data.items():
        assert all(d["integrity_checks"].values())
        base = d["cases"]["unchanged"]
        checks = []
        for key, case in d["cases"].items():
            if key.startswith("floor_"):
                alpha = case["alpha"]
                for layer, row in case["down_spectra"].items():
                    a, r, w = [
                        row[k] for k in ("square_factor", "rectangular_factor", "projection")
                    ]
                    original = base["down_spectra"][layer]["square_factor"]
                    checks += [
                        a["condition_nonzero"] <= (1 / alpha) * (1 + 1e-5),
                        abs(a["frobenius_norm"] / original["frobenius_norm"] - 1) < 1e-6,
                        w["condition_nonzero"]
                        <= a["condition_nonzero"] * r["condition_nonzero"] * (1 + 1e-5),
                    ]
                    if alpha == 1:
                        checks.append(
                            abs(w["condition_nonzero"] / r["condition_nonzero"] - 1) < 1e-5
                        )
        assert all(checks)
        result["numerical_bound_checks"][name] = {"checks": len(checks), "all_pass": all(checks)}
        result["locked_floor_gates"][name] = d["cases"]["floor_0.1"]["local_retention_gates"]
    lines = [
        "# Trained projection conditioning: intervention and transfer",
        "",
        "A moderate post-training singular-value floor substantially improves the down-projection condition numbers with small validation-loss changes. The effect transfers to all three existing smaller-model seeds. This is a measured projection-level benefit with unchanged parameter count and matrix-operation shapes; it does not establish improved training, runtime, full-network gradient stability or architectural novelty.",
        "",
        "Read the [initial frozen protocol and scoped proof](conditioning_plan.md) and the [locked transfer plan](conditioning_replication_plan.md). Original checkpoints remain unchanged. Transformed square-factor tensors, source archives, fixed data hashes, exact recipes, full spectra and directional gradient records are retained in each results/conditioning_* directory.",
        "",
        "## Width384/layer8, seed17, 800-step checkpoint",
        "",
        "| Intervention | NLL | Relative NLL change | Worst down condition | Condition improvement | Local gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, case in data[names[0]]["cases"].items():
        gates = case["local_retention_gates"]
        decision = "PASS" if all(gates.values()) else "FAIL"
        if key in ("unchanged", "svd_reconstruction"):
            decision = "Control"
        lines.append(
            f"| {key} | {case['validation_nll']:.6f} | {case['relative_nll_percent']:+.5f}% | {case['worst_down_condition']:.2f} | {case['condition_improvement_factor']:.1f}x | {decision} |"
        )
    lines += [
        "",
        "Floor strengths .01 and .1 pass the prespecified >=10x condition-improvement and <=.1% relative NLL-degradation gates. The .1 floor improves worst down conditioning by 1136x, to 26.53, at a .01118% NLL cost. Full orthogonalization (strength1) improves conditioning further but costs 3.81% NLL and fails. Random perturbations match per-block edit norm; at strength.1 they worsen conditioning despite a similarly small NLL change. This supports a directional effect in this checkpoint, not a replicated distributional claim over noise samples.",
        "",
        "All eight cases have finite loss/recorded gradients. Unchanged and SVD-reconstructed validation reproduce the archived baseline exactly. The new materialized product is computed in FP64 from FP32 factors; its original worst condition 30134.36 differs slightly from the earlier FP32-product audit's 30134.46. This precision distinction does not explain the intervention's large effect.",
        "",
        "## Locked .1 floor on width192/layer4, three seeds",
        "",
        "The moderate strength was selected using the larger-model result, then fixed before evaluating the smaller native-gate h1024 checkpoints. Each seed includes unchanged and reconstruction controls plus a fixed-direction matched random perturbation. This is transfer across size and seeds, not three-seed confirmation of the larger model's quality result.",
        "",
        "| Seed | Original NLL | Floored NLL | Relative NLL change | Original down condition | Floored down condition | Random-control down condition | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed, name in zip((17, 29, 43), names[1:], strict=True):
        cases = data[name]["cases"]
        base, floor, noise = [cases[k] for k in ("unchanged", "floor_0.1", "random_0.1")]
        lines.append(
            f"| {seed} | {base['validation_nll']:.6f} | {floor['validation_nll']:.6f} | {floor['relative_nll_percent']:+.5f}% | {base['worst_down_condition']:.2f} | {floor['worst_down_condition']:.2f} | {noise['worst_down_condition']:.2f} | {'PASS' if all(floor['local_retention_gates'].values()) else 'FAIL'} |"
        )
    original = [data[n]["cases"]["unchanged"]["validation_nll"] for n in names[1:]]
    floored = [data[n]["cases"]["floor_0.1"]["validation_nll"] for n in names[1:]]
    result["smaller_nll"] = {
        "original_mean": statistics.mean(original),
        "floor_mean": statistics.mean(floored),
        "floor_sample_sd": statistics.stdev(floored),
    }
    result["all_three_transfer_gates_pass"] = all(
        all(result["locked_floor_gates"][n].values()) for n in names[1:]
    )
    lines += [
        "",
        f"Original mean NLL is {statistics.mean(original):.6f}; floored mean is {statistics.mean(floored):.6f} (sample SD {statistics.stdev(floored):.6f}). The all-three transfer decision is {'PASS' if result['all_three_transfer_gates_pass'] else 'FAIL'}. The original small-model quality gap against full SwiGLU remains; this intervention addresses projection conditioning, not that accuracy deficit.",
        "",
        "![Condition floor across retained checkpoints](../results/plots/conditioning_transfer.png)",
        "",
        "## Proof scope and next decision",
        "",
        "The floor is applied across all singular values in each layer's square block-diagonal factor, followed by a common rescale preserving its Frobenius norm. For nonzero factors in exact arithmetic, its condition is at most 1/alpha; at alpha1 it is scaled orthogonal and preserves the rectangular factor's condition in the composed projection. All recorded finite-precision factor bounds, Frobenius preservation and product-condition inequalities pass numerical checks.",
        "",
        "The intervention changes learned weights and therefore changes the represented function. It adds no weights, buffers or inference operations after edited factors replace the originals. The offline SVD cost is separate; this diagnostic makes no serving-speed claim. The rectangular down projections necessarily have an input nullspace: the condition numbers compare only their nonzero singular values. The reported condition is not that of the gated FFN or residual-network Jacobian. Finite one-batch loss-direction gradients provide no global lower bound.",
        "",
        "A hard orthogonal constraint is not justified by the quality failure at strength1. The evidence instead supports retaining the mild floor as a post-training conditioning option and testing any training-time variant separately against matched controls. Separate [fused execution](fused_execution_results.md) and [larger three-seed quality](scale_replication_results.md) reports now supply those local checks for original weights. The [joint three-checkpoint audit](joint_conditioning_results.md) now supplies the floor-plus-fused numerical and quality checks. Convergence, broader data and combined serving speed remain unmeasured. Existing structured orthogonal work is cited in the proof plan; no new primitive is established.",
        "",
        "Reproduce with `uv run python -m src.core.conditioning_report`.",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    x = np.arange(4)
    for i, (key, label, color) in enumerate(
        (
            ("unchanged", "Original", "#505a6b"),
            ("floor_0.1", "Moderate floor", "#087e8b"),
            ("random_0.1", "Matched random", "#cc792b"),
        )
    ):
        rows = [data[n]["cases"][key] for n in names]
        axes[0].bar(
            x + (i - 1) * 0.25,
            [r["worst_down_condition"] for r in rows],
            width=0.24,
            label=label,
            color=color,
        )
        if key != "unchanged":
            positions = x + (i - 1.5) * 0.3
            values = [r["relative_nll_percent"] for r in rows]
            axes[1].bar(positions, values, width=0.29, color=color, label=label)
            for xx, yy in zip(positions, values, strict=True):
                axes[1].annotate(
                    f"{yy:.3f}%",
                    (xx, yy),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    labels = ["d384\nseed17", "d192\nseed17", "d192\nseed29", "d192\nseed43"]
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(
        yscale="log",
        ylabel="Worst down-projection condition (log scale)",
        title="Moderate floor transfers across checkpoints",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        ylabel="Relative validation NLL change (%)", title="Local allowed degradation: 0.1%"
    )
    low, high = axes[1].get_ylim()
    axes[1].set_ylim(min(0, low), high * 1.25)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(
            root / "plots" / f"conditioning_transfer.{suffix}", dpi=100, bbox_inches="tight"
        )
    plt.close(fig)
    write_json(root / "conditioning_summary.json", result)
    Path("research/conditioning_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
