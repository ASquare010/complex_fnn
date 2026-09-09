"""Strict learned-activation report: all trials, execution costs and shape interventions."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.activation_screen import (
    RATES,
    RECIPES,
    configuration,
    controls,
    output_path,
    promotion,
)
from src.core.config import ModelConfig, TrainConfig
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import select_best

LABELS = {
    "calibrated_narrow_shifted": "Narrow + shifted Bezier",
    "calibrated_narrow_rational": "Narrow + rational",
    "blockshuffle_shifted": "BlockShuffle + shifted Bezier",
    "blockshuffle_rational": "BlockShuffle + rational",
}


def load(folder=Path("results/activation_screen_v1"), recipes=RECIPES):
    result = json.loads((folder / "result.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 3 * len(recipes)
    trial_map = {t["run"]: t for t in result["trials"]}
    rows, histories, hashes = {}, {}, {}
    common = None
    for recipe in recipes:
        rows[recipe] = []
        for rate in RATES:
            path = output_path(recipe, rate)
            m = json.loads((path / "metrics.json").read_text())
            mc, tc = configuration(recipe, rate)
            assert ModelConfig(**m["model"]) == mc and TrainConfig(**m["training"]) == tc
            assert m["training_tokens"] == 409600 and m["validation_tokens"] == 322688
            assert (
                m["ffn_parameters"] == mc.unique_ffn_parameters
                and m["total_parameters"] == mc.total_parameters
            )
            assert m["precision"] == "bf16" and m["data"]["dataset"] == "Salesforce/wikitext"
            assert sha256(path / "metrics.json") == trial_map[m["run"]]["metrics_sha256"]
            if common is None:
                common = m
            assert (
                m["data"]["files"] == common["data"]["files"]
                and m["environment"] == common["environment"]
            )
            # Files implementing the training computation stay frozen across all workers.
            for name in (
                "src/core/trainer.py",
                "src/core/transformer.py",
                "src/core/config.py",
                "src/core/optimization.py",
                "src/core/data.py",
                "src/core/benchmark.py",
                "src/core/diagnostics.py",
                "src/learnable_activation_ffn/__init__.py",
            ):
                assert (
                    m["provenance"]["source_files"][name]
                    == common["provenance"]["source_files"][name]
                )
            history = [
                json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()
            ]
            assert [r["step"] for r in history] == [1, 50, 100, 150, 200]
            assert all(
                math.isfinite(r[k])
                for r in history
                for k in ("validation_loss", "training_loss", "gradient_norm_pre_clip")
            )
            assert all(
                math.isfinite(v)
                for r in history
                for v in r["layer_gradient_norms_post_clip"].values()
            )
            assert abs(history[-1]["validation_loss"] - m["validation_loss"]) < 1e-12
            for stage in ("initial", "final"):
                diagnostics = json.loads((path / f"{stage}_diagnostics.json").read_text())
                assert len([k for k in diagnostics if k.endswith("learnable_activation")]) == 8
                assert all(
                    r["finite"] and r.get("slope_finite", True) for r in diagnostics.values()
                )
            with zipfile.ZipFile(path / "source.zip") as archive:
                assert all(
                    hashlib.sha256(archive.read(n)).hexdigest() == h
                    for n, h in m["provenance"]["source_files"].items()
                )
            rows[recipe].append(m)
            histories[m["run"]] = history
            hashes[m["run"]] = {
                "metrics": sha256(path / "metrics.json"),
                "checkpoint": sha256(path / "checkpoint.pt"),
                "source_archive": sha256(path / "source.zip"),
            }
    return result, rows, histories, hashes


def report():
    result, rows, histories, hashes = load()
    selected = {k: select_best(v) for k, v in rows.items()}
    refs = controls()
    assert {k: v["run"] for k, v in selected.items()} == result["selected"]
    gates = {k: promotion(v, k.rsplit("_", 1)[0], refs) for k, v in selected.items()}
    assert gates == result["promotion_gates"]
    assert {k: all(v.values()) for k, v in gates.items()} == result["earns_longer_training"]
    shape_root = Path("results/activation_shapes_v1")
    shape_result = json.loads((shape_root / "result.json").read_text())
    assert shape_result["status"] == "complete" and len(shape_result["cases"]) == 4
    shapes = {}
    for case in shape_result["cases"]:
        path = shape_root / case["recipe"] / "result.json"
        assert sha256(path) == case["result_sha256"]
        s = json.loads(path.read_text())
        assert s["run"] == selected[case["recipe"]]["run"] and s["validation_targets"] == 322688
        assert s["checkpoint_sha256"] == hashes[s["run"]]["checkpoint"]
        assert s["source_checkpoint_unchanged"] and s["common_trained_parameters_unchanged"]
        shapes[case["recipe"]] = s
    relative = {
        k: {ref: 100 * (v["validation_loss"] / m["validation_loss"] - 1) for ref, m in refs.items()}
        for k, v in selected.items()
    }
    promoted = [LABELS[k] for k, v in gates.items() if all(v.values())]
    winner = selected["blockshuffle_rational"]
    lead = (
        f"**BlockShuffle with a rational activation improves selected NLL to {winner['validation_loss']:.6f}: "
        f"{-relative['blockshuffle_rational']['blockshuffle']:.3f}% below its unmodified base and "
        f"{-relative['blockshuffle_rational']['calibrated_narrow']:.3f}% below calibrated narrow.** "
        f"It retains {winner['ffn_reduction_percent']:.4f}% fewer FFN weights. "
        f"Eager training peak is {winner['peak_allocated_vram_bytes'] / 2**20:.2f} MiB, so memory still blocks promotion."
    )
    lines = [
        "# Learnable activations on the compressed front-runners",
        "",
        lead,
        "",
        f"Frozen promotion result: **{', '.join(promoted) if promoted else 'NONE'}**. Twelve new trials compare two activation families on calibrated narrow SwiGLU and BlockShuffle. All completed with finite recorded diagnostics. This is a seed-17, 200-step WikiText selection screen, not convergence or a universal expressiveness result.",
        "",
        "Read the [domain guide](learnable_activation_domain.md), [equations and scoped proofs](../src/learnable_activation_ffn/model.md), [frozen plan](learnable_activation_plan.md), [raw decisions](../results/activation_screen_v1/result.json) and [preflight](../results/activation_screen_v1/preflight.json).",
        "",
        "## Selected final checkpoints",
        "",
        "| Recipe | Peak LR | NLL | Change vs own base | FFN weights | Peak MiB | Longer training |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("full_gelu", "full_swiglu", "calibrated_narrow", "blockshuffle"):
        m = refs[key]
        lines.append(
            f"| {key.replace('_', ' ')} | {m['training']['learning_rate']:g} | {m['validation_loss']:.6f} | reference | {m['ffn_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | reference |"
        )
    for key, m in selected.items():
        own = key.rsplit("_", 1)[0]
        lines.append(
            f"| {LABELS[key]} | {m['training']['learning_rate']:g} | {m['validation_loss']:.6f} | {relative[key][own]:+.4f}% | {m['ffn_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {'PASS' if all(gates[key].values()) else 'FAIL'} |"
        )
    lines += [
        "",
        "Negative NLL changes are improvements. Shifted curves add 576 weights across eight layers; rational curves add 320. Total weights are 9,100,224 and 9,099,968, respectively, versus 9,099,648 for either unmodified compressed base. FFN reduction remains above 70%.",
        "",
        "## Every allocated trial",
        "",
        "| Recipe | Peak LR | Final NLL | Peak MiB | Clipped steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, trials in rows.items():
        for m in trials:
            lines.append(
                f"| {LABELS[key]} | {m['training']['learning_rate']:g} | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% |"
            )
    lines += [
        "",
        "![Equal global-rate grids](../results/plots/learnable_activation_screen.png)",
        "",
        "## What blocked promotion",
        "",
    ]
    for key, decisions in gates.items():
        failed = [k.replace("_", " ") for k, v in decisions.items() if not v]
        lines.append(
            f"- **{LABELS[key]}:** {', '.join(failed) if failed else 'all frozen promotion gates pass'}."
        )
    lines += [
        "",
        "Grid-boundary winners: "
        + ", ".join(LABELS[k] for k in result["grid_boundary_winners"])
        + ". This grid was not extended after inspecting losses.",
        "",
        "## Execution costs",
        "",
        "| Selected recipe | Serial training tokens/s | Eager inference ms | Training peak / full GELU |",
        "|---|---:|---:|---:|",
    ]
    for key, m in selected.items():
        lines.append(
            f"| {LABELS[key]} | {m['training_tokens_per_second']:.0f} | {m['inference_latency_ms']:.3f} | {m['peak_allocated_vram_bytes'] / refs['full_gelu']['peak_allocated_vram_bytes']:.3f} |"
        )
    lines += [
        "",
        "These are serial descriptive timings, not a simultaneous paired speed comparison. New pointwise kernels, FP32 intermediates and activation/product recomputation add real costs. Narrow uses eager backward; BlockShuffle uses non-reentrant checkpointing. The old native SiLU-only derivative and fused serving kernels cannot evaluate these functions. Matrix work stays 5,603,328 FFN FLOPs per token; activation arithmetic and memory traffic are additional.",
        "",
        "## What the learned shapes do",
        "",
        "| Selected recipe | Grid correction RMS across all layers/groups | Largest absolute correction | NLL after reset | Reset cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, s in shapes.items():
        correction = np.array([r["correction"] for r in s["layers"]])
        lines.append(
            f"| {LABELS[key]} | {np.sqrt(np.mean(correction**2)):.6f} | {np.max(np.abs(correction)):.6f} | {s['reset_nll']:.6f} | {s['reset_relative_nll_percent']:+.4f}% |"
        )
    lines += [
        "",
        "![Learned corrections and derivatives](../results/plots/learnable_activation_shapes.png)",
        "",
        "The grid is uniform on [-6,6]; it is not weighted by the training preactivation distribution. Bands span all 64 layer/group curves, not statistical confidence intervals. Full recorded shape parameters and sampled real-token slopes are retained. Nonzero curve movement establishes that parameters learned, not that the richer family caused a training benefit.",
        "",
        "Reset sets every activation theta to zero in the selected trained checkpoint, restoring SiLU while keeping all other trained weights fixed. Original full-validation NLL reproduces before each intervention. Source checkpoints remain unchanged. This tests dependence of an already co-adapted checkpoint on its learned correction; it is not training a fixed-activation control or identifying the cause of any gain.",
        "",
        "## Evidence limits and next decision",
        "",
        "Both compressed bases start with the exact same common initial tensors, logits, loss and gradients as their respective controls on the real-token BF16 preflight. All 22 pre-existing model behaviors remain exact on the archived CPU cases. Projection/data/optimizer sources and assignments are preserved; richer activation recomputation is disclosed. Each recipe receives three rates and 409,600 tokens per run. The reused controls received the same three-rate budget; historical architecture search remains unequal.",
        "",
        "Validation covers 322,688 targets, including the final partial batch. Selection and reset use that same validation split; it is no longer unscored evidence for these choices. The official test remains unfetched. No three-seed or TinyStories-transfer claim follows from this cohort. Local activation derivative bounds are not a full-network stability or convergence proof. Novelty remains unverified.",
        "",
        (
            "Promoted recipes earn independent replication and simpler/fixed-coordinate controls before larger claims."
            if promoted
            else "No recipe earns the frozen longer-training budget. Preserve the learned shapes and isolate the limiting quality or execution cost before proposing another hypothesis; do not relax the gates after the result."
        ),
        "",
    ]
    if Path("results/activation_training_repeat_v1/result.json").exists():
        lines.extend(
            [
                "",
                "Execution follow-up: pointwise fusion passes the [isolated memory audit](activation_execution_results.md), but the [full training repeat](activation_training_repeat_results.md) fails quality. The original no-promotion decision remains.",
                "",
            ]
        )
    Path("research/learnable_activation_results.md").write_text(
        chr(10).join(lines), encoding="utf-8"
    )
    write_json(
        Path("results/learnable_activation_summary.json"),
        {
            "selected": selected,
            "relative_nll_percent": relative,
            "promotion_gates": gates,
            "hashes": hashes,
            "shape_results": {
                k: {
                    f: s[f]
                    for f in ("run", "original_nll", "reset_nll", "reset_relative_nll_percent")
                }
                for k, s in shapes.items()
            },
        },
    )
    plot_dir = Path("results/plots")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout="constrained")
    colors = {"shifted": "#a9561a", "rational": "#2479a3"}
    for ax, base, title in zip(
        axes, ("calibrated_narrow", "blockshuffle"), ("Calibrated narrow base", "BlockShuffle base")
    ):
        for family in ("shifted", "rational"):
            ms = rows[f"{base}_{family}"]
            ax.plot(
                np.array(RATES) * 1e3,
                [m["validation_loss"] for m in ms],
                marker="o",
                color=colors[family],
                label=family.title(),
            )
        control = [
            json.loads(
                Path(
                    f"results/runs/wikitext2_{base}_lr{round(r * 1e6)}_s17_200/metrics.json"
                ).read_text()
            )
            for r in RATES
        ]
        ax.plot(
            np.array(RATES) * 1e3,
            [m["validation_loss"] for m in control],
            marker="s",
            color="#333333",
            label="Unmodified base",
        )
        ax.axhline(
            refs["full_gelu"]["validation_loss"],
            color="#607949",
            ls="--",
            lw=1,
            label="Selected full GELU",
        )
        ax.set(
            title=title,
            xlabel="Peak learning rate (x 0.001)",
            ylabel="Final validation NLL (lower is better)",
        )
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    fig.suptitle("Learnable activations: all rates, seed 17, WikiText / 200 steps")
    for extension in ("png", "svg"):
        fig.savefig(plot_dir / f"learnable_activation_screen.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), layout="constrained")
    for row, key in enumerate(RECIPES):
        s = shapes[key]
        x = np.array(s["grid"])
        for col, field in enumerate(("correction", "derivative")):
            curves = np.array([r[field] for r in s["layers"]]).reshape(-1, len(x))
            ax = axes[row, col]
            color = colors[key.rsplit("_", 1)[1]]
            ax.fill_between(x, curves.min(0), curves.max(0), alpha=0.18, color=color)
            ax.plot(x, curves.mean(0), color=color, label="Mean over 64 shapes")
            if field == "derivative":
                sig = 1 / (1 + np.exp(-x))
                ax.plot(
                    x, sig + x * sig * (1 - sig), ls="--", color="#333333", label="SiLU derivative"
                )
            else:
                ax.axhline(0, color="#333333", ls="--", lw=1)
            ax.set(
                title=LABELS[key] + " / " + field,
                xlabel="Preactivation z",
                ylabel="Added activation" if col == 0 else "d phi / dz",
            )
            ax.grid(alpha=0.15)
            if row == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Selected learned shapes: band is layer/group range, not uncertainty")
    for extension in ("png", "svg"):
        fig.savefig(plot_dir / f"learnable_activation_shapes.{extension}", dpi=170)
    fig.savefig(
        Path(".cache") / "activation_shapes_preview.jpg", dpi=100, pil_kwargs={"quality": 72}
    )
    plt.close(fig)
    print(
        json.dumps(
            {
                "promoted": promoted,
                "selected_nll": {k: v["validation_loss"] for k, v in selected.items()},
                "relative_nll_percent": relative,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    report()
