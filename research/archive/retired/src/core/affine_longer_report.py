"""Verified longer-budget results, diagnostics and learned affine functions."""

import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core.affine_longer import CRITICAL, PLAN, RECIPES, output_path, promotion, verify_trial
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/affine_longer_v2")
LABELS = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "calibrated_narrow": "Calibrated narrow",
    "blockshuffle": "BlockShuffle",
    "blockshuffle_affine": "BlockShuffle + affine",
}
COLORS = {
    "full_swiglu": "#7650a1",
    "full_gelu": "#888888",
    "calibrated_narrow": "#c38626",
    "blockshuffle": "#3780a0",
    "blockshuffle_affine": "#21754b",
}


def report():
    initial = json.loads((ROOT / "preflight.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    result = json.loads((ROOT / "result.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 5
    assert sha256(PLAN) == protocol["plan_sha256"] == result["plan_sha256"]
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
    rows = {}
    histories = {}
    rng = None
    hashes = {}
    for recipe in RECIPES:
        m, state = verify_trial(recipe, initial)
        assert all(
            m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        if rng is None:
            rng = state
        assert torch.equal(rng, state)
        p = output_path(recipe)
        trial = next(r for r in result["trials"] if r["recipe"] == recipe)
        assert sha256(p / "metrics.json") == trial["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == trial["checkpoint_sha256"]
        rows[recipe] = m
        histories[recipe] = [
            json.loads(line) for line in (p / "history.jsonl").read_text().splitlines()
        ]
        hashes[recipe] = {
            "metrics_sha256": sha256(p / "metrics.json"),
            "checkpoint_sha256": sha256(p / "checkpoint.pt"),
            "source_sha256": sha256(p / "source.zip"),
        }
    gates = promotion(rows)
    assert gates == result["gates"] and all(gates.values()) == result["earns_independent_seeds"]
    candidate = rows["blockshuffle_affine"]
    relative = {
        k: 100 * (candidate["validation_loss"] / v["validation_loss"] - 1)
        for k, v in rows.items()
        if k != "blockshuffle_affine"
    }
    lines = [
        "# Affine BlockShuffle at 800 WikiText steps",
        "",
        f"**Frozen promotion: {'PASS' if result['earns_independent_seeds'] else 'FAIL'}.** Affine BlockShuffle finishes at NLL {candidate['validation_loss']:.6f}, versus {rows['blockshuffle']['validation_loss']:.6f} for unmodified BlockShuffle and {rows['calibrated_narrow']['validation_loss']:.6f} for calibrated narrow. This is one seed at a longer fixed budget, not established convergence or independent replication.",
        "",
        f"The five recipes each train for 1,638,400 sampled tokens from the same pinned WikiText cache. Final validation uses all 322,688 targets. The candidate has {candidate['ffn_parameters']:,} FFN weights ({candidate['ffn_reduction_percent']:.4f}% fewer than full models), {candidate['total_parameters']:,} total weights ({candidate['total_reduction_percent']:.4f}% fewer), and adds only 128 weights to unmodified BlockShuffle. Projection, attention and embedding dimensions are unchanged.",
        "",
        "[Frozen plan](affine_activation_longer_plan.md), [raw result](../results/affine_longer_v2/result.json), [preflight](../results/affine_longer_v2/preflight.json), and [earlier short screen](affine_activation_results.md).",
        "",
        "## All five final checkpoints",
        "",
        "| Recipe | Peak LR | NLL | FFN weights | Peak MiB | Clipped steps | Train tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, m in rows.items():
        lines.append(
            f"| {LABELS[k]} | {m['training']['learning_rate']:g} | {m['validation_loss']:.6f} | {m['ffn_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.2f}% | {m['training_tokens_per_second']:.0f} |"
        )
    lines += [
        "",
        "Serial throughput is descriptive: these measurements do not establish a paired training-speed or inference-speed advantage. All execution is native BF16, with the existing BlockShuffle activation checkpointing. Matrix-work estimates exclude pointwise operations and do not imply measured speed.",
        "",
        "![Longer trajectories and memory](../results/plots/affine_longer.png)",
        "",
        "## Frozen gates",
        "",
        "| Gate | Decision |",
        "|---|---|",
    ]
    for k, v in gates.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        "Candidate relative NLL changes: "
        + "; ".join(f"{LABELS[k]} {v:+.4f}%" for k, v in relative.items())
        + ".",
        "",
        "Each peak learning rate was selected at 200 steps and frozen for this cohort. The warmup/cosine schedule is stretched to 800 steps and evaluations occur at steps 1, 200, 400, 600, 800. The 200-step and 800-step trajectories therefore use different schedules; their equal-step values are not continuation checkpoints. Most selected rates were at the search boundary, so longer-budget rate optima are not bracketed. No new rate search or best-intermediate-checkpoint selection is used.",
        "",
        "## Optimization and learned shapes",
        "",
    ]
    for k in ("blockshuffle", "blockshuffle_affine"):
        diag = json.loads((output_path(k) / "final_diagnostics.json").read_text())
        ffn = [v for name, v in diag.items() if name.endswith("ffn")]
        slopes = [v for name, v in diag.items() if name.endswith("learnable_activation")]
        if ffn:
            rms = [v["std"] for v in ffn]
            lines.append(
                f"{LABELS[k]}: final logged pre-clip gradient norm {rows[k]['final_gradient_norm']:.6f}; FFN-output standard deviation across layers ranges from {min(rms):.6f} to {max(rms):.6f} on the fixed diagnostic batch."
            )
        if slopes:
            lines.append(
                f"Sampled affine activation slopes have maximum absolute value {max(v['activation_slope_abs_max'] for v in slopes):.6f}; all activation/shape diagnostics are finite, and maximum control saturation fraction is {max(v['control_saturation_fraction'] for v in slopes):.6f}. This is sampled local evidence, not a whole-network gradient bound."
            )
        lines.append("")
    shapes = shape_plot(initial)
    lines += [
        "![Learned affine functions by layer](../results/plots/affine_longer_shapes.png)",
        "",
        "Each color identifies a layer; thin curves show its eight channel groups. Corrections start at zero. The 200-step and 800-step curves come from independently trained models with the same seed and different schedule lengths. The shape plot is a uniform-grid visualization, not a data-weighted mechanistic test. It cannot by itself explain a gain or failure.",
        "",
        "## Reproduction and qualification",
        "",
        "All five archived control source/checkpoint hashes, model dimensions, optimizer groups, initialization metadata and data hashes are verified. Initial full-validation NLL matches each source within 1e-7, and final sampling RNG states agree across all five new runs. All history points, gradients and initial/final diagnostics are finite. The training-loop AST and the shared execution sources remain fixed across the cohort.",
        "",
        "The first preflight stopped before training because a historical FLOP-counting change was mistaken for an execution change. Inspection showed that only the count formula changed to exclude activation parameters. Training/evaluation AST and all archived matrix-work counts reproduce. The failed launch and its explicit qualification are retained in [affine_longer_v1](../results/affine_longer_v1/qualification.json); measurements are in affine_longer_v2.",
        "",
    ]
    if result["earns_independent_seeds"]:
        lines.append(
            "This pass earns replication at seeds 29 and 43 with the same configurations, followed by TinyStories transfer and an activation-path ablation. It does not complete the gold research goal."
        )
    else:
        lines.append(
            "This failure stops independent-seed promotion under the frozen protocol. The short-screen win remains a short-screen result. Inspect which quality, memory or material-gain gate failed before designing another experiment; do not redefine acceptance around the earlier result."
        )
    lines += [
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.cli train --config configs/wikitext2_blockshuffle_affine_800.json --cache data/wikitext2_v1",
        "uv run --extra compile --extra data python -m src.core.affine_longer_report",
        "```",
        "",
        "The ordinary CLI creates a new run. The archival five-worker audit deliberately refuses to overwrite existing fixed run names. The official WikiText test split remains unfetched and unscored.",
    ]
    Path("research/affine_activation_longer_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(
        Path("results/affine_longer_summary.json"),
        {
            "gates": gates,
            "earns_independent_seeds": result["earns_independent_seeds"],
            "relative_nll_percent": relative,
            "run_hashes": hashes,
            "shape_plot": shapes,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for k, h in histories.items():
        axes[0].plot(
            [r["step"] for r in h],
            [r["validation_loss"] for r in h],
            "o-",
            color=COLORS[k],
            label=LABELS[k],
        )
    axes[0].set(
        xlabel="Optimizer step", ylabel="Validation NLL", title="Frozen 800-step trajectories"
    )
    axes[0].legend(fontsize=8)
    order = list(rows)
    axes[1].bar(
        range(5),
        [rows[k]["peak_allocated_vram_bytes"] / 2**20 for k in order],
        color=[COLORS[k] for k in order],
    )
    axes[1].set_xticks(range(5), [LABELS[k].replace(" ", "\n") for k in order], fontsize=8)
    limit = (
        1.1
        * min(rows[k]["peak_allocated_vram_bytes"] for k in ("full_gelu", "full_swiglu"))
        / 2**20
    )
    axes[1].axhline(limit, linestyle="--", color="#942b2b", label="Both-reference memory limit")
    axes[1].set(
        ylabel="Peak allocated MiB", title="Native training memory", ylim=(0, max(850, limit * 1.2))
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    for suffix in ("png", "svg"):
        fig.savefig(f"results/plots/affine_longer.{suffix}", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "gates": gates,
                "relative_nll_percent": relative,
                "earns_independent_seeds": result["earns_independent_seeds"],
            }
        )
    )


def shape_plot(initial):
    short = Path("results/runs") / initial["sources"]["blockshuffle_affine"]["run"]
    paths = {200: short, 800: output_path("blockshuffle_affine")}
    grid = np.linspace(-6, 6, 257)
    sigmoid = 1 / (1 + np.exp(-grid))
    derivative = sigmoid + grid * sigmoid * (1 - sigmoid)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), layout="constrained", sharey="row")
    records = {}
    for column, (steps, path) in enumerate(paths.items()):
        diag = json.loads((path / "final_diagnostics.json").read_text())
        records[steps] = []
        for layer in range(8):
            shape = diag[f"layer_{layer}.learnable_activation"]["shape"]
            a = np.asarray(shape["gain"])
            b = np.asarray(shape["bias"])
            records[steps].append({"layer": layer, "gain": a.tolist(), "bias": b.tolist()})
            for group in range(8):
                color = plt.cm.viridis(layer / 7)
                axes[0, column].plot(
                    grid, a[group] * grid + b[group], color=color, alpha=0.5, linewidth=0.8
                )
                axes[1, column].plot(
                    grid, derivative + a[group], color=color, alpha=0.4, linewidth=0.8
                )
        axes[0, column].axhline(
            0, color="black", linestyle="--", linewidth=1, label="Initial correction"
        )
        axes[0, column].set(
            title=f"{steps}-step learned corrections", ylabel="phi(z) - SiLU(z)", xlabel="z"
        )
        axes[1, column].plot(
            grid, derivative, color="black", linestyle="--", linewidth=1, label="Initial SiLU slope"
        )
        axes[1, column].set(
            title=f"{steps}-step activation derivatives", ylabel="d phi / dz", xlabel="z"
        )
        axes[1, column].legend(fontsize=8)
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, 7), cmap="viridis")
    fig.colorbar(scalar, ax=axes.ravel().tolist(), label="Layer", ticks=range(8), shrink=0.8)
    for suffix in ("png", "svg"):
        fig.savefig(f"results/plots/affine_longer_shapes.{suffix}", dpi=150)
    plt.close(fig)
    return {
        "grid": grid.tolist(),
        "layers_by_training_steps": records,
        "scope": "Uniform grid and recorded FP32 effective gains/biases; no new model evaluation or shape fitting.",
    }


if __name__ == "__main__":
    report()
