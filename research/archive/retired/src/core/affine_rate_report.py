"""Verify and present the complete two-rate activation comparison."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.core.affine_longer import CRITICAL, verify_trial
from src.core.affine_longer import output_path as original_path
from src.core.affine_rate_control import (
    CASES,
    PLAN,
    RATES,
    output_path,
    same_rate_comparisons,
    verify_new,
)
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import promotion

ROOT = Path("results/affine_rate_control_v1")


def report():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 2
    assert result["plan_sha256"] == protocol["plan_sha256"] == sha256(PLAN)
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    initial = json.loads(Path("results/affine_longer_v2/preflight.json").read_text())
    references, matrix, paths = {}, {"600": {}, "1200": {}}, {}
    sampling = None
    for recipe in ("full_swiglu", "full_gelu", "calibrated_narrow", *CASES):
        m, rng = verify_trial(recipe, initial)
        references[recipe] = m
        if sampling is None:
            sampling = rng
        assert torch.equal(sampling, rng)
        assert (
            sha256(original_path(recipe) / "metrics.json")
            == protocol["source_hashes"][recipe]["metrics"]
        )
        assert (
            sha256(original_path(recipe) / "checkpoint.pt")
            == protocol["source_hashes"][recipe]["checkpoint"]
        )
        if recipe in CASES:
            rate = str(round(m["training"]["learning_rate"] * 1e6))
            matrix[rate][recipe] = m
            paths[rate, recipe] = original_path(recipe)
    for recipe in CASES:
        m, rng = verify_new(recipe, initial, references[recipe])
        assert torch.equal(sampling, rng)
        t = next(t for t in result["trials"] if t["recipe"] == recipe)
        assert sha256(output_path(recipe) / "metrics.json") == t["metrics_sha256"]
        assert sha256(output_path(recipe) / "checkpoint.pt") == t["checkpoint_sha256"]
        assert all(
            m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in (*CRITICAL, "src/core/affine_rate_control.py")
        )
        rate = str(round(RATES[recipe] * 1e6))
        matrix[rate][recipe] = m
        paths[rate, recipe] = output_path(recipe)
    comparisons = same_rate_comparisons(matrix)
    quality = {
        rate: {r: promotion({**references, "blockshuffle": m}) for r, m in pair.items()}
        for rate, pair in matrix.items()
    }
    assert comparisons == result["same_rate_comparisons"] and quality == result["reference_gates"]
    robust = all(v["at_least_point_two_percent_better"] for v in comparisons.values())
    assert robust == result["material_benefit_at_both_rates"]
    lines = [
        "# Same-rate control for affine BlockShuffle",
        "",
        f"**Frozen two-rate robustness decision: {'PASS' if robust else 'FAIL'}.** The two missing 800-step seed-17 cells complete the plain/affine by 0.0006/0.0012 comparison. This addresses the learning-rate confound in the selected-recipe result. It remains one-seed evidence, with a separate gate-backward execution difference.",
        "",
        "[Frozen H043 plan](affine_activation_rate_control_plan.md), [raw decisions](../results/affine_rate_control_v1/result.json), and [three-seed selected recipes](affine_activation_replication_results.md).",
        "",
        "| Peak LR | Plain NLL | Affine NLL | Affine minus plain | Relative change | At least 0.2% better |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for rate, c in comparisons.items():
        lines.append(
            f"| {int(rate) / 1e6:.4f} | {c['plain_nll']:.9f} | {c['affine_nll']:.9f} | {c['affine_minus_plain_nll']:+.9f} | {c['relative_affine_change_percent']:+.4f}% | {'PASS' if c['at_least_point_two_percent_better'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## All four cells and quality/memory decisions",
        "",
        "| Rate | Architecture | Training peak MiB | Clip fraction | All reference gates | Evidence |",
        "|---|---|---:|---:|---|---|",
    ]
    for rate, pair in matrix.items():
        for recipe, m in pair.items():
            new = int(rate) == round(RATES[recipe] * 1e6)
            lines.append(
                f"| {int(rate) / 1e6:.4f} | {recipe} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.3f}% | {'PASS' if all(quality[rate][recipe].values()) else 'FAIL'} | {'New H043' if new else 'Original H039'} |"
            )
    lines += [
        "",
        "Reference gates require >=70% fewer FFN weights, <=1% relative NLL cost against both full controls, beating calibrated narrow, and allocated training peak <=1.1 times each full control. Same-rate material benefit is assessed separately above.",
        "",
        "![Same-rate comparison and trajectories](../results/plots/affine_rate_control.png)",
        "",
        "## Interpretation",
        "",
    ]
    if robust:
        lines += [
            "Affine improves by at least 0.2% at both tested rates. A global rate change alone does not account for these same-rate recipe differences. The benefit still cannot be assigned solely to the learned shape: plain uses the custom native SiLU/product backward; affine uses checkpointed native autograd. A same-rate, same-recomputation plain control is the next minimal mechanism test."
        ]
    else:
        lines += [
            "The >=0.2% benefit does not hold at both rates. Retain the H041 result as a comparison of selected recipes and qualify claims of activation-specific gain. Do not infer that 128 learned coefficients alone caused the three-seed difference. Inspect both rate-specific effects and the distinct recomputation/backward paths before adding gain/bias variants."
        ]
    lines += [
        "",
        "All four cells have exact final sampling-RNG equality. New trials change only peak LR within their source recipe; initial NLL, optimizer groups, initialization, critical computation hashes, validation targets, source archives and checkpoint/config metadata are verified. Each trial uses 1,638,400 sampled tokens and all 322,688 validation targets. Official test stays unscored. Rates and hypotheses were frozen before the two new results; no convergence or multi-seed same-rate claim is made.",
        "",
        "TinyStories transfer, gain/bias retraining, deployment timing and broader novelty/convergence checks remain separate. The successful post-training removal experiment is not a substitute for these training controls.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.affine_rate_report",
        "```",
    ]
    Path("research/affine_activation_rate_control_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    summary = {
        "same_rate_comparisons": comparisons,
        "reference_gates": quality,
        "material_benefit_at_both_rates": robust,
        "sources_verified": True,
        "source_hashes": {
            p.name: {
                "metrics": sha256(p / "metrics.json"),
                "checkpoint": sha256(p / "checkpoint.pt"),
            }
            for p in paths.values()
        },
    }
    write_json(Path("results/affine_rate_control_summary.json"), summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    colors = {"blockshuffle": "#277da1", "blockshuffle_affine": "#20754e"}
    labels = {"blockshuffle": "Plain", "blockshuffle_affine": "Affine"}
    for recipe in CASES:
        axes[0].plot(
            [0, 1],
            [matrix[r][recipe]["validation_loss"] for r in ("600", "1200")],
            "o-",
            color=colors[recipe],
            label=labels[recipe],
        )
        for rate, style in (("600", "--"), ("1200", "-")):
            history = [
                json.loads(line)
                for line in (paths[rate, recipe] / "history.jsonl").read_text().splitlines()
            ]
            h = [r for r in history if r["step"] >= 200]
            axes[1].plot(
                [r["step"] for r in h],
                [r["validation_loss"] for r in h],
                style,
                color=colors[recipe],
                marker="o",
                label=f"{labels[recipe]} LR {int(rate) / 1e6:.4f}",
            )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["0.0006", "0.0012"],
        xlabel="Global peak learning rate",
        ylabel="Final full-validation NLL",
        title="Same-rate comparison; seed 17",
    )
    axes[1].set(
        xlabel="Optimizer step", ylabel="Full-validation NLL", title="Frozen 800-step schedules"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    for extension in ("png", "svg"):
        fig.savefig(f"results/plots/affine_rate_control.{extension}", dpi=160)
    plt.close(fig)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    report()
