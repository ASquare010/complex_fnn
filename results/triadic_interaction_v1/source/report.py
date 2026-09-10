"""Post-audit H100 figures and descriptive feature-alignment measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from results.triadic_interaction_v1.source.model import FORMS, TASKS, InteractionFFN

ROOT = Path("results/triadic_interaction_v1")
torch.set_num_threads(4)


def read(path):
    return json.loads(Path(path).read_text())


def align(weight, basis):
    w = weight.double()
    squared = (w @ basis).square() / w.square().sum(1, keepdim=True)
    energy = squared.sum(1)
    return {
        "mean_subspace_fraction": energy.mean().item(),
        "median_subspace_fraction": energy.median().item(),
        "top16_mean_subspace_fraction": energy.topk(16).values.mean().item(),
        "max_squared_cosine_by_teacher_direction": squared.max(0).values.tolist(),
    }


def main():
    assert read(ROOT / "audit_process.json")["status"] == "PASS"
    result = read(ROOT / "result.json")
    rows = [r for r in read(ROOT / "raw_result.json")["rows"] if r["selected"]]
    data = torch.load(ROOT / "data.pt", map_location="cpu", weights_only=True)
    basis = data["basis"]
    del data
    alignment, trajectories, diagnostics = [], {}, []
    for row in rows:
        training = read(ROOT / "cells" / row["label"] / "training.json")
        diagnostics.append({"label": row["label"], "samples": training["diagnostics"]})
        trajectories[row["label"]] = {
            key: np.array([h[key] for h in training["history"]]).reshape(30, 10).mean(1).tolist()
            for key in ("loss", "preclip_norm")
        }
        if row["form"] == "feature_aware":
            continue
        initial = InteractionFFN(row["form"], row["seed"], row["task"])
        saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["model"]
        for name in ("up", "gate", "third"):
            projection = getattr(initial, name)
            if projection is not None:
                alignment.append(
                    {
                        "task": row["task"],
                        "form": row["form"],
                        "seed": row["seed"],
                        "projection": name,
                        "initial": align(projection.weight, basis),
                        "final": align(saved[name + ".weight"], basis),
                    }
                )
    export = {
        "scope": "Post-audit descriptive diagnostic; teacher basis used only after fitting",
        "limitation": "Alignment is not a causal intervention; maxima also depend on width",
        "alignment": alignment,
        "trajectories_10step_means": trajectories,
        "sampled_diagnostics": diagnostics,
    }
    with (ROOT / "diagnostics.json").open("x") as stream:
        json.dump(export, stream, indent=2)
        stream.write("\n")
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    summary = result["summary"]
    forms = [f for f in FORMS if f != "feature_aware"]
    colors = [
        "#b2b9c5"
        if f.startswith("full")
        else "#d76b23"
        if f == "cp3"
        else "#823e96"
        if f == "bounded_cp3"
        else "#3567a8"
        for f in forms
    ]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, task in zip(axes.flat, TASKS, strict=True):
        ax.barh(
            forms,
            [summary["per_task"][f][task]["mean"] for f in forms],
            xerr=[summary["per_task"][f][task]["sample_sd"] for f in forms],
            color=colors,
            capsize=2,
        )
        ax.invert_yaxis()
        ax.axvline(result["zero_mse"][task], ls=":", color="black", label="zero predictor")
        ax.set(title=task, xlabel="Reporting MSE (mean +/- sample SD, three seeds)", xlim=(0, None))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("H100: raw and bounded three-way interactions on hidden directions")
    fig.savefig(plots / "comparison.png", dpi=125)
    plt.close(fig)

    chosen = ("narrow_gelu", "starrelu", "cubic_ridge", "cp2", "cp3", "bounded_cp3")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
    for ax, task in zip(axes, TASKS, strict=True):
        for form in chosen:
            values = [
                a
                for a in alignment
                if a["task"] == task and a["form"] == form and a["projection"] == "up"
            ]
            a, b = [
                np.mean([v[phase]["mean_subspace_fraction"] for v in values])
                for phase in ("initial", "final")
            ]
            ax.plot([0, 1], [a, b], marker="o", label=form)
        ax.axhline(16 / 384, color="black", ls=":", lw=0.8)
        ax.set(
            title=task,
            xticks=[0, 1],
            xticklabels=["Initial", "Final"],
            ylabel="Mean weight energy in target subspace",
            ylim=(0, 1),
        )
    axes[0].legend(fontsize=7)
    fig.suptitle("First projection only; known target basis used after training")
    fig.savefig(plots / "alignment.png", dpi=125)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for col, task in enumerate(TASKS):
        for form in chosen:
            subset = [r for r in rows if r["task"] == task and r["form"] == form]
            for row_index, key in enumerate(("loss", "preclip_norm")):
                mean = np.mean([trajectories[r["label"]][key] for r in subset], 0)
                axes[row_index, col].plot(np.arange(1, 31) * 10, mean, label=form)
                axes[row_index, col].set(
                    xlabel="Update",
                    yscale="log",
                    ylabel="Training MSE" if key == "loss" else "Preclip gradient norm",
                )
        axes[0, col].set_title(task)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Selected-rate trajectories: three-seed, ten-step means")
    fig.savefig(plots / "training.png", dpi=125)
    plt.close(fig)
    print(
        json.dumps(
            {
                "alignment_records": len(alignment),
                "diagnostic_runs": len(diagnostics),
                "plots": [p.as_posix() for p in sorted(plots.glob("*.png"))],
            }
        )
    )


if __name__ == "__main__":
    main()
