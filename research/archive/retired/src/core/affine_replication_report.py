"""Verified three-seed WikiText results with every paired comparison retained."""

import json
import statistics
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.affine_longer import CRITICAL, RECIPES, output_path, promotion
from src.core.affine_longer_report import COLORS, LABELS
from src.core.affine_replication import PLAN, PREVIOUS, load_three_seeds, paired_statistics
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/affine_replication_v1")


def report():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 10
    assert result["plan_sha256"] == protocol["plan_sha256"] == sha256(PLAN)
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        import hashlib

        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    initial = json.loads((PREVIOUS / "preflight.json").read_text())
    rows, rng = load_three_seeds(initial)
    gates = {seed: promotion(m) for seed, m in rows.items()}
    assert {str(s): g for s, g in gates.items()} == result["gates_by_seed"]
    assert {str(s): g for s, g in rng.items()} == result["sampling_rng_hashes"]
    assert all(all(g.values()) for g in gates.values()) == result["replication_passes"]
    hashes = {}
    for seed, models in rows.items():
        for recipe, m in models.items():
            path = output_path(recipe, seed)
            if seed != 17:
                trial = next(
                    t for t in result["trials"] if t["seed"] == seed and t["recipe"] == recipe
                )
                assert sha256(path / "metrics.json") == trial["metrics_sha256"]
                assert sha256(path / "checkpoint.pt") == trial["checkpoint_sha256"]
                assert all(
                    m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                    for n in (*CRITICAL, "src/core/affine_longer.py")
                )
            hashes[path.name] = {
                "metrics": sha256(path / "metrics.json"),
                "checkpoint": sha256(path / "checkpoint.pt"),
                "source_archive": sha256(path / "source.zip"),
            }
    paired = {
        recipe: paired_statistics(
            [
                rows[s]["blockshuffle_affine"]["validation_loss"]
                - rows[s][recipe]["validation_loss"]
                for s in rows
            ]
        )
        for recipe in RECIPES
        if recipe != "blockshuffle_affine"
    }
    assert paired == result["paired_statistics"]
    summaries = {}
    for recipe in RECIPES:
        values = [rows[s][recipe] for s in rows]
        losses = [v["validation_loss"] for v in values]
        summaries[recipe] = {
            "mean_nll": statistics.mean(losses),
            "sample_sd_nll": statistics.stdev(losses),
            "ffn_parameters": values[0]["ffn_parameters"],
            "total_parameters": values[0]["total_parameters"],
            "mean_peak_mib": statistics.mean(
                v["peak_allocated_vram_bytes"] / 2**20 for v in values
            ),
            "max_peak_mib": max(v["peak_allocated_vram_bytes"] / 2**20 for v in values),
            "mean_clip_fraction": statistics.mean(v["clipped_step_fraction"] for v in values),
            "mean_serial_train_tokens_per_second": statistics.mean(
                v["training_tokens_per_second"] for v in values
            ),
        }
    candidate = summaries["blockshuffle_affine"]
    relative = {
        k: 100 * (candidate["mean_nll"] / v["mean_nll"] - 1)
        for k, v in summaries.items()
        if k != "blockshuffle_affine"
    }
    qualification = []
    if Path("results/affine_rate_control_v1/result.json").exists():
        qualification = [
            "**Later same-rate qualification:** [H043](affine_activation_rate_control_results.md) finds only 0.0193% affine benefit at LR 0.0012 in seed 17, versus 0.337% at LR 0.0006. The selected-recipe advantage below compares different rates and is not an isolated activation gain. The [corrected three-seed result](affine_rate_replication_results.md) finds only 0.101% mean affine benefit and 2/3 wins, failing material promotion.",
            "",
        ]
    lines = [
        "# Affine BlockShuffle: three-seed WikiText replication",
        "",
        *qualification,
        f"**Frozen replication decision: {'PASS' if result['replication_passes'] else 'FAIL'}.** Mean affine BlockShuffle NLL is {candidate['mean_nll']:.6f} +/- {candidate['sample_sd_nll']:.6f} sample SD at 800 steps, seeds 17, 29 and 43. Every seed and gate is retained below. This is a fixed small-model, fixed-token-budget result, not established convergence or broad SOTA evidence.",
        "",
        "The candidate uses 2,801,792 FFN weights and 9,099,776 total, adding only 128 weights to unmodified BlockShuffle. Full models use 9,437,184 FFN weights and 15,735,168 total: 70.3111% fewer FFN weights and 42.1692% fewer total weights. All five architectures use the same attention, vocabulary and depth. The two new seeds add ten training runs, 16,384,000 sampled tokens, with no architecture or rate changes.",
        "",
        "[Frozen replication plan](affine_activation_replication_plan.md), [earlier 800-step cohort](affine_activation_longer_results.md), [raw decisions](../results/affine_replication_v1/result.json), and [preflight](../results/affine_replication_v1/preflight.json).",
        "",
        "## Every final NLL",
        "",
        "| Recipe | Seed 17 | Seed 29 | Seed 43 | Mean +/- sample SD | Max peak MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for recipe, v in summaries.items():
        losses = [rows[s][recipe]["validation_loss"] for s in rows]
        lines.append(
            f"| {LABELS[recipe]} | {losses[0]:.6f} | {losses[1]:.6f} | {losses[2]:.6f} | {v['mean_nll']:.6f} +/- {v['sample_sd_nll']:.6f} | {v['max_peak_mib']:.2f} |"
        )
    lines += [
        "",
        "Candidate relative change against each mean: "
        + "; ".join(f"{LABELS[k]} {v:+.4f}%" for k, v in relative.items())
        + ".",
        "",
        "![All seed outcomes and paired differences](../results/plots/affine_replication.png)",
        "",
        "## Frozen gates, separately by seed",
        "",
        "| Gate | Seed 17 | Seed 29 | Seed 43 |",
        "|---|---|---|---|",
    ]
    for name in gates[17]:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join("PASS" if gates[s][name] else "FAIL" for s in rows)
            + " |"
        )
    lines += [
        "",
        "## Paired evidence and its limits",
        "",
        "Differences below are affine candidate minus comparator NLL; negative favors affine. Intervals are exploratory paired-mean 95% Student-t intervals (df=2), conditional on approximately normal independent seed differences. The first seed was used for selection and three samples cannot validate the normality assumption. These are not unqualified confirmatory intervals.",
        "",
        "| Comparator | Paired differences (17,29,43) | Mean difference | Exploratory 95% t interval | One-sided sign p |",
        "|---|---|---:|---|---:|",
    ]
    for recipe, v in paired.items():
        ds = ", ".join(f"{x:+.6f}" for x in v["differences"])
        ci = v["student_t_95_interval"]
        lines.append(
            f"| {LABELS[recipe]} | {ds} | {v['mean']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {v['one_sided_sign_p']:.3f} |"
        )
    lines += [
        "",
        "The exact sign test concerns the paired median and excludes exact ties. Three wins out of three give one-sided p=0.125; that alone does not meet a 0.05 threshold. Frozen per-seed engineering gates and statistical significance are separate claims. Method references: [NIST Student-t table](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm) and [NIST sign test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm).",
        "",
        "## Optimization and execution",
        "",
        "| Recipe | Mean clipped steps | Mean allocated peak MiB | Mean serial train tokens/s |",
        "|---|---:|---:|---:|",
    ]
    for recipe, v in summaries.items():
        lines.append(
            f"| {LABELS[recipe]} | {100 * v['mean_clip_fraction']:.3f}% | {v['mean_peak_mib']:.2f} | {v['mean_serial_train_tokens_per_second']:.0f} |"
        )
    lines += [
        "",
        "All recorded gradients, activations and learned slopes are finite. This does not prove global nonvanishing gradients or well-conditioned full-network Jacobians. Serial timing is descriptive; rotated worker order does not create simultaneous paired speed measurements. The known full-product and correction-only compiler training failures remain separate; all runs here use native execution.",
        "",
        "## Verification and next requirements",
        "",
        "Each new configuration differs only in seed. Tiny CPU checks preserve all common non-FFN initial parameters across recipes and preserve affine/base initial logits, loss and common gradients exactly. Actual full-validation initial NLL agrees for affine and unmodified BlockShuffle within each seed. Final sampling RNG states agree across recipes within a seed and differ across seeds. All source archives, checkpoints, model/optimizer metadata, token counts and frozen data hashes are verified. No test split was downloaded or scored.",
        "",
        "All models train for 1,638,400 sampled tokens and score the same 322,688 validation targets. Rates were selected at 200 steps; most were boundary winners and were not retuned for 800 steps or the new seeds. Historical architecture search, selected seed 17, the small tokenizer/model and repeated use of validation data constrain generalization claims.",
        "",
    ]
    lines.append(
        "The original frozen pass earned transfer and gain/bias studies. The later same-rate correction above takes precedence: its material-benefit gate fails, so those additional activation experiments are not promoted on the confounded selected-rate advantage."
        if result["replication_passes"]
        else "The replication fails its frozen gates. Preserve the earlier positive results while identifying the failing seed/constraint before selecting another experiment."
    )
    lines += [
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.cli train --config configs/wikitext2_blockshuffle_affine_800.json --cache data/wikitext2_v1 --seed 29",
        "uv run --extra compile --extra data python -m src.core.affine_replication_report",
        "```",
        "",
        "The ordinary CLI creates a fresh run; archival audit names are protected against overwrite.",
    ]
    Path("research/affine_activation_replication_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(
        Path("results/affine_replication_summary.json"),
        {
            "replication_passes": result["replication_passes"],
            "summaries": summaries,
            "relative_mean_nll_percent": relative,
            "paired_statistics": paired,
            "gates_by_seed": gates,
            "run_hashes": hashes,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout="constrained")
    for index, recipe in enumerate(RECIPES):
        ys = [rows[s][recipe]["validation_loss"] for s in rows]
        axes[0].scatter(index + np.array([-0.08, 0, 0.08]), ys, color=COLORS[recipe], s=34)
        axes[0].errorbar(
            index,
            summaries[recipe]["mean_nll"],
            yerr=summaries[recipe]["sample_sd_nll"],
            color="black",
            marker="_",
            capsize=5,
        )
    axes[0].set_xticks(range(5), [LABELS[k].replace(" ", "\n") for k in RECIPES], fontsize=8)
    axes[0].set(ylabel="Final validation NLL", title="Three seeds; mean +/- sample SD")
    for index, (recipe, v) in enumerate(paired.items()):
        mean = v["mean"]
        lo, hi = v["student_t_95_interval"]
        axes[1].errorbar(
            mean,
            index,
            xerr=[[mean - lo], [hi - mean]],
            color=COLORS[recipe],
            marker="o",
            capsize=4,
        )
        axes[1].scatter(v["differences"], [index] * 3, color=COLORS[recipe], marker="x", s=35)
    axes[1].axvline(0, color="gray", linestyle="--")
    axes[1].set_yticks(range(len(paired)), [LABELS[k] for k in paired], fontsize=8)
    axes[1].set(
        xlabel="Affine minus comparator NLL", title="Paired mean and exploratory 95% t interval"
    )
    axes[0].grid(axis="y", alpha=0.15)
    axes[1].grid(axis="x", alpha=0.15)
    for suffix in ("png", "svg"):
        fig.savefig(f"results/plots/affine_replication.{suffix}", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "replication_passes": result["replication_passes"],
                "relative_mean_nll_percent": relative,
                "paired_statistics": paired,
            }
        )
    )


if __name__ == "__main__":
    report()
