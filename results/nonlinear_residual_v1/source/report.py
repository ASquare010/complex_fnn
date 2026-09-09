"""Post-audit H099 figures and compact diagnostics; never changes fitted evidence."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/nonlinear_residual_v1")
TASKS = ("quadratic", "cubic", "product", "piecewise")


def read(path):
    return json.loads(Path(path).read_text())


def run():
    assert read(ROOT / "audit_process.json")["status"] == "PASS"
    result = read(ROOT / "result.json")
    raw = read(ROOT / "raw_result.json")
    rows = [r for r in raw["rows"] if r["selected"]]
    summary = result["summary"]
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    forms = (
        "full_gelu",
        "full_swiglu",
        "full_relu2",
        "narrow_gelu",
        "narrow_swiglu",
        "narrow_relu2",
        "starrelu",
        "even_rational",
        "fixed_mix",
        "learned_mix",
    )
    colors = ["#9aa4b2"] * 3 + ["#3567a8"] * 4 + ["#7e57a3"] * 2 + ["#d76b23"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, task in zip(axes.flat, TASKS, strict=True):
        means = [summary["per_task"][f][task]["mean"] for f in forms]
        errors = [summary["per_task"][f][task]["sample_sd"] for f in forms]
        ax.barh(forms, means, xerr=errors, color=colors, capsize=2)
        ax.invert_yaxis()
        ax.axvline(result["zero_mse"][task], color="black", ls=":", label="zero predictor")
        affine = np.mean([r["reporting_mse"] for r in result["affine"] if r["task"] == task])
        ax.axvline(affine, color="#a62f3b", ls="--", label="affine fit")
        ax.set(title=task, xlabel="Reporting MSE (mean +/- sample SD, three seeds)")
        ax.set_xlim(left=0)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("H099: nonlinear residual learning at 300 updates, batch 256")
    fig.savefig(plots / "comparison.png", dpi=130)
    plt.close(fig)

    selected = {}
    for task in TASKS:
        selected[task] = [
            read(ROOT / "cells" / r["label"] / "training.json")
            for r in rows
            if r["task"] == task and r["form"] == "learned_mix"
        ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    curve_diagnostics = {}
    z = np.linspace(-4, 4, 301)
    s = z / (1 + np.abs(z))
    for col, task in enumerate(TASKS):
        runs = selected[task]
        values = np.array([[d["shape"]["alpha"] for d in r["diagnostics"]] for r in runs])
        steps = [d["step"] for d in runs[0]["diagnostics"]]
        curve_diagnostics[task] = {
            "steps": steps,
            "alpha_by_seed_step_group": values.tolist(),
            "seed_order": [r["seed"] for r in runs],
            "selected_diagnostics": [r["diagnostics"] for r in runs],
        }
        for group in range(4):
            mean = values[:, :, group].mean(0)
            sd = values[:, :, group].std(0, ddof=1)
            axes[0, col].plot(steps, mean, marker=".", label=f"group {group + 1}")
            axes[0, col].fill_between(steps, mean - sd, mean + sd, alpha=0.12)
        for i, step in enumerate(steps):
            alpha = values[:, i, :].mean()
            axes[1, col].plot(z, z * s * (1 + alpha * s), label=f"step {step}")
        axes[0, col].set(title=task, xlabel="Update", ylabel="Learned alpha")
        axes[1, col].set(xlabel="Preactivation z", ylabel="Activation (seed/group mean)")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Four shared shape parameters learn; movement alone is not a quality win")
    fig.savefig(plots / "learned_shapes.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    convergence = {}
    for col, task in enumerate(TASKS):
        convergence[task] = {}
        for form in ("narrow_gelu", "narrow_swiglu", "starrelu", "fixed_mix", "learned_mix"):
            runs = [
                read(ROOT / "cells" / r["label"] / "training.json")
                for r in rows
                if r["task"] == task and r["form"] == form
            ]
            loss = np.array([[h["loss"] for h in r["history"]] for r in runs])
            grad = np.array([[h["preclip_norm"] for h in r["history"]] for r in runs])
            # Non-overlapping 10-step means avoid an endpoint padding artifact.
            loss_mean = loss.reshape(3, 30, 10).mean((0, 2))
            grad_mean = grad.reshape(3, 30, 10).mean((0, 2))
            convergence[task][form] = {
                "loss_10step_mean": loss_mean.tolist(),
                "preclip_norm_10step_mean": grad_mean.tolist(),
            }
            axes[0, col].plot(np.arange(1, 31) * 10, loss_mean, label=form)
            axes[1, col].plot(np.arange(1, 31) * 10, grad_mean)
        axes[0, col].set(title=task, yscale="log", xlabel="Update", ylabel="Training MSE")
        axes[1, col].set(yscale="log", xlabel="Update", ylabel="Preclip gradient norm")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Selected-rate training trajectories: three-seed, ten-step means")
    fig.savefig(plots / "training.png", dpi=130)
    plt.close(fig)
    export = {
        "scope": "Post-audit descriptive export; no new training or selection",
        "curves": curve_diagnostics,
        "convergence": convergence,
        "all_values_finite": all(math.isfinite(r["reporting_mse"]) for r in rows),
    }
    with (ROOT / "diagnostics.json").open("x") as stream:
        json.dump(export, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "plots": [p.as_posix() for p in sorted(plots.glob("*.png"))],
                "diagnostics": (ROOT / "diagnostics.json").as_posix(),
            }
        )
    )


if __name__ == "__main__":
    run()
