"""Strict report for the twelve-trial WikiText tuning screen."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.config import ModelConfig, TrainConfig
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import (
    RATES,
    RECIPES,
    configuration,
    output_path,
    promotion,
    select_best,
)

LABELS = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "calibrated_narrow": "Calibrated narrow",
    "blockshuffle": "BlockShuffle",
}


def load(root=Path("results")):
    folder = root / "wikitext_screen_v1"
    result = json.loads((folder / "result.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 12
    records, histories, hashes = {}, {}, {}
    common = None
    for recipe in RECIPES:
        records[recipe] = []
        for rate in RATES:
            path = root / "runs" / output_path(recipe, rate).name
            m = json.loads((path / "metrics.json").read_text())
            mc, tc = configuration(recipe, rate)
            assert mc == ModelConfig(**m["model"]) and tc == TrainConfig(**m["training"])
            assert m["training_tokens"] == 409600 and m["validation_tokens"] == 322688
            assert m["data"]["dataset"] == "Salesforce/wikitext" and m["precision"] == "bf16"
            assert (
                m["ffn_parameters"] == mc.unique_ffn_parameters
                and m["total_parameters"] == mc.total_parameters
            )
            if common is None:
                common = m
            assert (
                m["data"]["files"] == common["data"]["files"]
                and m["environment"] == common["environment"]
            )
            assert math.isfinite(m["validation_loss"]) and math.isfinite(m["final_gradient_norm"])
            assert 0 <= m["clipped_step_fraction"] <= 1
            history = [
                json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()
            ]
            assert [r["step"] for r in history] == [1, 50, 100, 150, 200]
            assert all(
                math.isfinite(r["validation_loss"])
                and math.isfinite(r["training_loss"])
                and math.isfinite(r["gradient_norm_pre_clip"])
                for r in history
            )
            assert abs(history[-1]["validation_loss"] - m["validation_loss"]) < 1e-12
            for stage in ("initial", "final"):
                diagnostics = json.loads((path / f"{stage}_diagnostics.json").read_text())
                assert all(v["finite"] for v in diagnostics.values())
            with zipfile.ZipFile(path / "source.zip") as archive:
                assert all(
                    hashlib.sha256(archive.read(name)).hexdigest() == value
                    for name, value in m["provenance"]["source_files"].items()
                )
            hashes[m["run"]] = {
                "metrics": sha256(path / "metrics.json"),
                "checkpoint": sha256(path / "checkpoint.pt"),
                "source_archive": sha256(path / "source.zip"),
            }
            records[recipe].append(m)
            histories[m["run"]] = history
    return result, records, histories, hashes


def report(root=Path("results")):
    result, records, histories, hashes = load(root)
    selected = {k: select_best(v) for k, v in records.items()}
    assert {k: m["run"] for k, m in selected.items()} == result["selected"]
    gates = promotion(selected)
    gates["all_trial_activation_diagnostics_finite"] = True
    assert gates == result["promotion_gates"] and all(gates.values()) == result["earns_800_steps"]
    candidate = selected["blockshuffle"]
    paired = {
        k: 100 * (candidate["validation_loss"] / selected[k]["validation_loss"] - 1)
        for k in ("full_swiglu", "full_gelu", "calibrated_narrow")
    }
    summary = {
        "result_sha256": sha256(root / "wikitext_screen_v1/result.json"),
        "artifact_hashes": hashes,
        "selected": {
            k: {
                field: m[field]
                for field in (
                    "run",
                    "validation_loss",
                    "ffn_parameters",
                    "total_parameters",
                    "peak_allocated_vram_bytes",
                    "clipped_step_fraction",
                    "training_cache_exposure_ratio",
                )
            }
            | {"peak_learning_rate": m["training"]["learning_rate"]}
            for k, m in selected.items()
        },
        "candidate_relative_nll_percent": paired,
        "promotion_gates": gates,
        "earns_800_steps": result["earns_800_steps"],
        "grid_boundary_winners": result["grid_boundary_winners"],
    }
    lines = [
        "# WikiText-2: equal-budget transfer screen",
        "",
        f"The frozen promotion decision is **{'PASS' if result['earns_800_steps'] else 'FAIL'}**. Every recipe received the same three-rate, 200-step tuning budget on the new corpus. Selection uses final complete-validation NLL. Read the [frozen plan](wikitext_screen_plan.md), [source/cache design](broader_corpus_design.md), [raw cohort](../results/wikitext_screen_v1/result.json) and [cache reconstruction](../results/verification/wikitext_reconstruction_v1.json).",
        "",
        "## Selected recipe comparison",
        "",
        "| Recipe | Selected peak LR | Final NLL | Unique FFN weights | Total weights | Peak allocated MiB | Clipped steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in selected.items():
        lines.append(
            f"| {LABELS[key]} | {m['training']['learning_rate']:.4g} | {m['validation_loss']:.6f} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% |"
        )
    lines += [
        "",
        f"Selected BlockShuffle NLL changes: {paired['full_swiglu']:+.4f}% versus full SwiGLU, {paired['full_gelu']:+.4f}% versus full GELU, and {paired['calibrated_narrow']:+.4f}% versus calibrated narrow. Its FFN/total reductions remain 70.3125% / 42.17%. These compare explicitly different FFN initialization and optimizer treatments under equal new global-rate search budgets.",
        "",
        "## Every allocated trial",
        "",
        "| Recipe | Peak LR | Initial NLL | Final NLL | Final pre-clip gradient norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for recipe in RECIPES:
        for m in records[recipe]:
            lines.append(
                f"| {LABELS[recipe]} | {m['training']['learning_rate']:.4g} | {m['initial_validation_loss']:.6f} | {m['validation_loss']:.6f} | {m['final_gradient_norm']:.4f} |"
            )
    lines += ["", "## Frozen gates", "", "| Check | Decision |", "|---|---|"]
    for key, value in gates.items():
        lines.append(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |")
    boundary = ", ".join(LABELS[k] for k in result["grid_boundary_winners"]) or "None"
    lines += [
        "",
        f"Grid-boundary winners: {boundary}. A boundary winner means this grid did not bracket the optimum. The grid is not expanded after seeing results. All twelve histories, initial/final diagnostics, checkpoints and source archives are retained.",
        "",
        "![All rates and selected learning curves](../results/plots/wikitext_screen.png)",
        "",
        "## What this evidence establishes",
        "",
        "The official raw train/validation files match their pinned published content hashes. Top-level headings delimit 629 training units and 60 validation units, with subsection headings preserved. Five normalized exact train duplicates are removed. The training-only 4,096-entry byte BPE produces 3,083,650 training tokens and 322,802 validation tokens with no unknown-token occurrences. A second cache construction using the archived tokenizer matched every stored file hash.",
        "",
        "Validation covers 2,521 context-128 windows and exactly 322,688 targets, including a final batch of nine windows. Preflight compared the actual concatenated target stream against the source tokens, actual common initial weights and parameter counts, real-token finite backward, and unchanged shared training/attention/data sources. Every run uses 409,600 sampled training tokens, about 0.133 cache exposures. This is far from a convergence study.",
        "",
        "This is one seed and selection on the same complete validation split used for choosing rates. The official test split is still unfetched and unscored. Equal new tuning does not erase the earlier unequal architecture search. These BPE NLLs cannot be compared to TinyStories scores or published word-level WikiText perplexities. Sequential native inference logs do not establish a paired serving advantage. No post-training floor or fused kernel changes were used in training.",
        "",
        "Next decision: "
        + (
            "The candidate earned a separately frozen 800-step comparison with all controls. Longer training, three seeds and a locked test-set evaluation are still required before claiming broader-data success."
            if result["earns_800_steps"]
            else "The frozen candidate does not earn the 800-step cohort. Inspect which quality, memory or optimization gate failed before defining another hypothesis; do not silently relax the threshold."
        ),
        "",
        "Current efficient-FFN comparators, convergence, longer-context behavior, full-network stability and verified novelty remain outstanding.",
    ]
    colors = {
        "full_swiglu": "#505a6b",
        "full_gelu": "#7656a8",
        "calibrated_narrow": "#cc792b",
        "blockshuffle": "#087e8b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for key in RECIPES:
        axes[0].plot(
            [m["training"]["learning_rate"] for m in records[key]],
            [m["validation_loss"] for m in records[key]],
            marker="o",
            label=LABELS[key],
            color=colors[key],
        )
        m = selected[key]
        h = histories[m["run"]]
        axes[1].plot(
            [r["step"] for r in h],
            [r["validation_loss"] for r in h],
            marker="o",
            ms=3,
            label=LABELS[key],
            color=colors[key],
        )
    axes[0].set(
        xlabel="Peak global learning rate",
        ylabel="Final complete-validation NLL",
        title="All twelve equal-budget trials",
        xscale="log",
    )
    axes[0].set_xticks(RATES, ["0.0003", "0.0006", "0.0012"])
    axes[0].minorticks_off()
    axes[1].set(
        xlabel="Training step",
        ylabel="Complete-validation NLL",
        title="Selected rate for each recipe; one seed",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(root / "plots" / f"wikitext_screen.{extension}", dpi=100, bbox_inches="tight")
    plt.close(fig)
    write_json(root / "wikitext_screen_summary.json", summary)
    Path("research/wikitext_screen_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
