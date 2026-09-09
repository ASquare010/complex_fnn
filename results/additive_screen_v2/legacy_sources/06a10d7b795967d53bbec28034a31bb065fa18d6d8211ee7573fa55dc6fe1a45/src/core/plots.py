"""Standalone figures and machine-readable learned curve diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.adaptive_cross_group_ffn import AdaptiveCoupledFFN
from src.affine_shared_ffn import AffineSharedFFN
from src.bezier_ffn import BezierActivation, QuadraticBezierActivation
from src.core.config import ModelConfig
from src.core.reproducibility import write_json
from src.core.transformer import Transformer


def curves(run: Path) -> None:
    """Show all cubic groups or 16 predetermined affine channels per layer."""
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**checkpoint["model_config"]))
    grid = torch.linspace(-6, 6, 241)
    kinds = (BezierActivation, AffineSharedFFN, QuadraticBezierActivation)
    initial = {
        name: module.curves(grid).numpy()
        for name, module in model.named_modules()
        if isinstance(module, kinds)
    }
    if not initial:
        return
    model.load_state_dict(checkpoint["model"])
    fig, axes = plt.subplots(len(initial), 2, figsize=(10, 3 * len(initial)), squeeze=False)
    records = {}
    for layer, (name, module) in enumerate(
        (n, m) for n, m in model.named_modules() if isinstance(m, kinds)
    ):
        final = module.curves(grid).numpy()
        before = initial[name]
        delta = final - before
        affine = isinstance(module, AffineSharedFFN)
        indices = np.linspace(0, final.shape[1] - 1, min(16, final.shape[1]), dtype=int)
        baseline = "SiLU" if affine else "GELU"
        if isinstance(module, QuadraticBezierActivation):
            baseline = "SiLU" if module.residual else "Initial quadratic"
        axes[layer, 0].plot(grid, before[:, 0], color="black", linestyle="--", label=baseline)
        axes[layer, 0].plot(grid, final[:, indices])
        axes[layer, 1].plot(grid, delta[:, indices])
        detail = "; value fixed to 1" if affine else ""
        axes[layer, 0].set_title(f"Layer {layer}: functions{detail}")
        axes[layer, 1].set_title(f"Layer {layer}: changes from {baseline}")
        axes[layer, 0].legend()
        record = {
            "displayed_channels": indices.tolist(),
            "max_curve_change": float(np.abs(delta).max()),
            "channel_curve_std_mean": float(final.std(axis=1).mean()),
            "grid": grid.tolist(),
            "initial": before[:, indices].tolist(),
            "final": final[:, indices].tolist(),
            "curve_columns": "displayed_channels; all channel coefficients are saved below",
        }
        if affine:
            record["interpretation"] = (
                "Effective scalar gate; value projection fixed to one. Not the whole FFN."
            )
            record["coefficients"] = {}
            for key, parameter, bound in (
                ("input_scale_delta", module.scale_delta, 0.5),
                ("input_shift", module.shift_delta, 1.0),
                ("output_scale_delta", module.output_delta, 0.5),
            ):
                bounded = (bound * parameter.tanh()).detach().numpy()
                record["coefficients"][key] = {
                    "raw": parameter.detach().tolist(),
                    "bounded": bounded.tolist(),
                    "bound": bound,
                    "saturation_fraction": float((np.abs(bounded) > 0.98 * bound).mean()),
                }
        else:
            control = (0.5 * module.theta.tanh()).detach().numpy()
            record.update(
                theta=module.theta.detach().tolist(),
                bounded_deltas=control.tolist(),
                control_saturation_fraction=float((np.abs(control) > 0.49).mean()),
            )
        if isinstance(module, QuadraticBezierActivation):
            record["controls"] = module.controls().detach().tolist()
            record["interpretation"] = "Scalar activation before branch products or value gating"
        records[name] = record
    for ax in axes.flatten():
        ax.set_xlabel("Preactivation")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "learned_curves.png", dpi=140)
    plt.close(fig)
    write_json(run / "learned_curves.json", records)


def coupling_curves(run: Path) -> None:
    """Plot the learned linear neighbor contribution, not a full FFN response."""
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, AdaptiveCoupledFFN)]
    if not modules:
        return
    grid = torch.linspace(-6, 6, 121)
    fig, axes = plt.subplots(len(modules), 2, figsize=(10, 3 * len(modules)), squeeze=False)
    records = {}
    for i, (name, module) in enumerate(modules):
        coefficient = module.coefficients().detach().numpy()
        values = module.curves(grid).numpy()
        indices = np.linspace(0, len(coefficient) - 1, min(16, len(coefficient)), dtype=int)
        axes[i, 0].hist(coefficient, bins=30)
        axes[i, 0].axvline(0, color="black", linestyle="--", label="Initial zero")
        axes[i, 0].set(
            title=f"Layer {i}: learned coupling", xlabel="Coefficient", ylabel="Channels"
        )
        axes[i, 1].plot(grid, values[:, indices])
        axes[i, 1].axhline(0, color="black", linestyle="--", label="Initial zero")
        axes[i, 1].set(
            title=f"Layer {i}: neighbor response, own value = 0",
            xlabel="Neighbor value",
            ylabel="Mixed value contribution",
        )
        records[name] = {
            "raw_theta": module.mix_theta.detach().tolist(),
            "coefficients": coefficient.tolist(),
            "bound": 0.75,
            "saturation_fraction": float((np.abs(coefficient) > 0.735).mean()),
            "grid": grid.tolist(),
            "final_curves": values[:, indices].tolist(),
            "initial_curves": np.zeros_like(values[:, indices]).tolist(),
            "displayed_channels": indices.tolist(),
            "interpretation": "Linear value-mixer slice with own value zero; excludes SiLU, other features and output projection.",
        }
    for ax in axes.flatten():
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "learned_curves.png", dpi=140)
    plt.close(fig)
    write_json(run / "learned_curves.json", records)


def overview(root: Path) -> None:
    paths = sorted((root / "runs").glob("*/metrics.json"))
    cohorts = {}
    from src.core.report import cohort

    for p in paths:
        record = json.loads(p.read_text())
        cohorts.setdefault(cohort(record), []).append((p.parent, record))
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    for key, entries in cohorts.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        grouped = {}
        for run, metrics in entries:
            label = metrics["model"]["variant"]
            if metrics["model"]["hidden"]:
                label += f" h={metrics['model']['hidden']}"
            if metrics["model"]["variant"].startswith("structured"):
                label += f" g={metrics['model']['groups']}"
            grouped.setdefault(label, []).append((run, metrics))
            if label.startswith("bezier") or "affine" in label or "quadratic" in label:
                curves(run)
            if "adaptive" in label:
                coupling_curves(run)
        for index, (label, runs) in enumerate(grouped.items()):
            histories = [
                [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
                for run, _ in runs
            ]
            steps = [r["step"] for r in histories[0]]
            values = np.array([[r["validation_loss"] for r in history] for history in histories])
            mean, std = (
                values.mean(0),
                values.std(0, ddof=1) if len(runs) > 1 else np.zeros(len(steps)),
            )
            color = f"C{index}"
            axes[0].plot(steps, mean, label=label, color=color)
            axes[0].fill_between(steps, mean - std, mean + std, alpha=0.12, color=color)
            axes[1].errorbar(
                runs[0][1]["ffn_parameters"] / 1e6,
                mean[-1],
                yerr=std[-1],
                fmt=["o", "s", "^", "D", "v"][index % 5],
                color=color,
                label=label,
                capsize=4,
            )
        axes[0].set(
            xlabel="Optimizer steps",
            ylabel="Validation NLL",
            title="Mean with sample SD across seeds",
        )
        axes[0].legend(fontsize=8)
        axes[1].set(
            xlabel="FFN parameters, all layers (millions)",
            ylabel="Final validation NLL",
            title="Quality versus FFN size",
        )
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots / f"screening_{key}.png", dpi=160)
        plt.close(fig)
    function_path = root / "functions_s17.json"
    if function_path.exists():
        rows = json.loads(function_path.read_text())["rows"]
        tasks = list(dict.fromkeys(r["task"] for r in rows))
        variants = list(dict.fromkeys(r["variant"] for r in rows))
        errors = np.array(
            [
                [
                    next(
                        r["heldout_normalized_mse"]
                        for r in rows
                        if r["task"] == task and r["variant"] == variant
                    )
                    for variant in variants
                ]
                for task in tasks
            ]
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        from matplotlib.colors import LogNorm

        plot = ax.imshow(errors, norm=LogNorm(), cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(variants)), variants, rotation=25, ha="right")
        ax.set_yticks(range(len(tasks)), tasks)
        for i in range(len(tasks)):
            for j in range(len(variants)):
                ax.text(
                    j,
                    i,
                    f"{errors[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if errors[i, j] > 0.5 else "black",
                )
        ax.set_title("Held-out normalized MSE (lower is better); one seed, 300 steps")
        fig.colorbar(plot, ax=ax, label="Normalized MSE, logarithmic colors")
        fig.tight_layout()
        fig.savefig(plots / "functions_s17.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    overview(args.root)
