"""Verified affine retraining ablation, costs, reset and real-input shape audit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.activation_report import load
from src.core.activation_screen import AFFINE_RECIPES, RATES, controls, promotion
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import select_best


def report():
    root = Path("results/affine_activation_screen_v1")
    result, rows, histories, hashes = load(root, AFFINE_RECIPES)
    assert result["protocol"]["plan_sha256"] == sha256(Path("research/affine_activation_plan.md"))
    refs = controls()
    selected = {k: select_best(v) for k, v in rows.items()}
    assert {k: v["run"] for k, v in selected.items()} == result["selected"]
    gates = {k: promotion(v, k.rsplit("_", 1)[0], refs) for k, v in selected.items()}
    assert gates == result["promotion_gates"]
    prior = json.loads(Path("results/activation_screen_v1/result.json").read_text())
    rational = {
        base: json.loads(
            Path(f"results/runs/{prior['selected'][base + '_rational']}/metrics.json").read_text()
        )
        for base in ("calibrated_narrow", "blockshuffle")
    }
    shape_root = Path("results/affine_activation_shapes_v1")
    shape_result = json.loads((shape_root / "result.json").read_text())
    assert shape_result["status"] == "complete" and len(shape_result["cases"]) == 2
    shapes = {}
    for row in shape_result["cases"]:
        p = shape_root / row["recipe"] / "result.json"
        assert sha256(p) == row["result_sha256"]
        shape = json.loads(p.read_text())
        assert shape["checkpoint_sha256"] == hashes[shape["run"]]["checkpoint"]
        assert shape["source_checkpoint_unchanged"] and shape["common_trained_parameters_unchanged"]
        assert shape["run"] == selected[row["recipe"]]["run"]
        shapes[row["recipe"]] = shape
    data_path = Path("results/activation_data_fit_v1/result.json")
    data_fit = json.loads(data_path.read_text())
    source = Path("results/runs") / rational["blockshuffle"]["run"]
    assert data_fit["checkpoint_sha256"] == sha256(source / "checkpoint.pt")
    assert data_fit["data_hashes"] == rational["blockshuffle"]["data"]["files"]
    assert data_fit["source_checkpoint_unchanged"]
    promoted = [k for k, v in gates.items() if all(v.values())]
    lines = [
        "# Affine activation: retraining and mechanism ablation",
        "",
        f"**Frozen promotion: {', '.join(promoted) if promoted else 'NONE'}.** Six new seed-17, 200-step WikiText trials test whether a learned linear and constant correction can reproduce the rational model's gain at lower cost. All six completed with finite recorded diagnostics. This is selection, not convergence or independent replication.",
        "",
        "The activation is `SiLU(z) + 0.25*(tanh(theta_gain)*z + tanh(theta_bias))`, shared within eight channel groups and independent per layer. It adds **128 weights across eight layers**, for 2,801,792 FFN weights and 9,099,776 total: 70.3111% fewer FFN weights than full references. Both compressed bases preserve their initialization, optimizer calibration and zero-correction initial function. Native BF16 training only; BlockShuffle uses checkpoint recomputation.",
        "",
        "See the [frozen protocol](affine_activation_plan.md), [equations and origin-Jacobian proof](../src/learnable_activation_ffn/model.md), [preflight](../results/affine_activation_screen_v1/preflight.json), and [raw decisions](../results/affine_activation_screen_v1/result.json). The preflight reproduces all 26 old variants exactly and verifies archived control/data sources and the unchanged training-loop AST.",
        "",
        "## Selected final checkpoints",
        "",
        "| Recipe | Peak LR | NLL | Peak allocated MiB |",
        "|---|---:|---:|---:|",
    ]
    for k, m in {**refs, **{k + "_rational": v for k, v in rational.items()}, **selected}.items():
        lines.append(
            f"| {k} | {m['training']['learning_rate']:g} | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} |"
        )
    lines += [
        "",
        "## All new trials",
        "",
        "| Peak LR | Narrow + affine NLL | BlockShuffle + affine NLL |",
        "|---:|---:|---:|",
    ]
    for index, rate in enumerate(RATES):
        lines.append(
            f"| {rate:g} | {rows[AFFINE_RECIPES[0]][index]['validation_loss']:.6f} | {rows[AFFINE_RECIPES[1]][index]['validation_loss']:.6f} |"
        )
    lines += [
        "",
        "![All rates and selected memory](../results/plots/affine_activation_screen.png)",
        "",
        "## Frozen gates and effect sizes",
        "",
    ]
    comparisons = {}
    for key, m in selected.items():
        base = key.rsplit("_", 1)[0]
        comparisons[key] = {
            name: 100 * (m["validation_loss"] / reference["validation_loss"] - 1)
            for name, reference in {
                "own_base": refs[base],
                "own_rational": rational[base],
                "full_swiglu": refs["full_swiglu"],
                "full_gelu": refs["full_gelu"],
                "narrow": refs["calibrated_narrow"],
            }.items()
        }
        values = comparisons[key]
        failed = [name for name, passes in gates[key].items() if not passes]
        lines.append(
            f"**{key}:** NLL change {values['own_base']:+.4f}% versus selected unmodified base; {values['own_rational']:+.4f}% versus selected native rational; {values['full_swiglu']:+.4f}% versus full SwiGLU; {values['full_gelu']:+.4f}% versus full GELU. Failed gates: {', '.join(failed) if failed else 'none'}."
        )
        lines.append("")
    lines += [
        f"Grid-boundary winners: {', '.join(result['grid_boundary_winners']) or 'none'}. Rate selection uses final NLL, never the best intermediate step. All 322,688 validation targets are scored in each trial, including the last partial batch. Each recipe receives the same three-rate budget; historical architecture search remains unequal. Full-control optima were not bracketed. Serial speed measurements are not paired speed evidence.",
        "",
        "## What the learned shapes explain",
        "",
    ]
    for key, shape in shapes.items():
        gains = [x for layer in shape["layers"] for x in layer["shape"]["gain"]]
        biases = [x for layer in shape["layers"] for x in layer["shape"]["bias"]]
        lines += [
            f"{key}: learned gains range [{min(gains):.6f}, {max(gains):.6f}], offsets [{min(biases):.6f}, {max(biases):.6f}]. Resetting both theta vectors while retaining every other trained weight changes full-validation NLL from {shape['original_nll']:.6f} to {shape['reset_nll']:.6f} ({shape['reset_relative_nll_percent']:+.6f}%).",
            "",
        ]
    lines += [
        f"For the previous native rational BlockShuffle checkpoint, affine fits explain **{100 * data_fit['affine_energy_fraction']:.4f}%** of correction energy on actual training-prefix inputs, with fit-error RMS {data_fit['affine_fit_error_rms']:.8f} and correction RMS {data_fit['correction_rms']:.8f}. This uses {data_fit['sample_count']:,} deterministic samples across 64 layer/group pairs from the first 1,024 training tokens. The earlier uniform-grid energy fraction was 99.7684%. This uncentered energy statistic is not centered R-squared; the small prefix is not guaranteed representative. Inputs are BF16 projection outputs; corrections are FP32 before the activation's BF16 cast. [Raw data-weighted fit](../results/activation_data_fit_v1/result.json).",
        "",
        "A small fit error or small post-training reset cost does not prove why retraining succeeds or fails. The affine residual and rational residual have different parameterizations and optimization directions. The strict origin-Jacobian extension applies only to an isolated bias-free SwiGLU layer; it does not establish easier learning, universal data adaptation, or whole-network gradient stability.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.cli train --config configs/wikitext2_blockshuffle_affine_screen.json --cache data/wikitext2_v1 --learning-rate 0.0012",
        "uv run --extra compile --extra data python -m src.core.affine_activation_report",
        "```",
        "",
        "The audit refuses to overwrite fixed run names. Use a separate checkout/output convention for an independent repeat; the report verifies the retained original artifacts. No official WikiText test data were downloaded or scored.",
    ]
    Path("research/affine_activation_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(
        Path("results/affine_activation_summary.json"),
        {
            "selected": result["selected"],
            "gates": gates,
            "comparisons_percent": comparisons,
            "promoted": promoted,
            "run_hashes": hashes,
            "data_fit_sha256": sha256(data_path),
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    colors = ["#246a9a", "#b35035"]
    for key, color in zip(AFFINE_RECIPES, colors):
        axes[0].plot(
            [r * 1e3 for r in RATES],
            [m["validation_loss"] for m in rows[key]],
            "o-",
            color=color,
            label=key.replace("calibrated_", ""),
        )
    for key, style in (("full_gelu", ":"), ("full_swiglu", "--")):
        axes[0].axhline(refs[key]["validation_loss"], color="gray", linestyle=style, label=key)
    axes[0].set(
        xlabel="Peak learning rate (x 0.001)",
        ylabel="Final validation NLL",
        title="All six 200-step trials",
    )
    axes[0].legend(fontsize=8)
    memory = [refs["full_gelu"], refs["full_swiglu"], *selected.values()]
    axes[1].bar(
        range(4),
        [m["peak_allocated_vram_bytes"] / 2**20 for m in memory],
        color=["#aaa", "#777", *colors],
    )
    axes[1].set_xticks(
        range(4), ["Full GELU", "Full SwiGLU", "Narrow\n+ affine", "BlockShuffle\n+ affine"]
    )
    axes[1].axhline(
        1.1
        * min(refs[k]["peak_allocated_vram_bytes"] for k in ("full_gelu", "full_swiglu"))
        / 2**20,
        color="#942b2b",
        linestyle="--",
        label="Both-reference memory limit",
    )
    axes[1].set(ylabel="Peak allocated MiB", title="Selected native training recipes")
    axes[1].set_ylim(0, 850)
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    for suffix in ("png", "svg"):
        fig.savefig(f"results/plots/affine_activation_screen.{suffix}", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "selected": result["selected"],
                "promoted": promoted,
                "comparisons_percent": comparisons,
            }
        )
    )


if __name__ == "__main__":
    report()
