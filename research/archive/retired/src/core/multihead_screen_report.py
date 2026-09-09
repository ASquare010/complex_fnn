"""Verify and report every cell of the frozen H046 multi-head screen."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.multihead_screen import (
    COMPUTATION,
    PLAN,
    RATES,
    RECIPES,
    candidate_gates,
    output_path,
    verify_archive,
    verify_trial,
)
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import RECIPES as CONTROLS
from src.core.wikitext_screen import output_path as control_path
from src.core.wikitext_screen import select_best

ROOT = Path("results/multihead_screen_v2")
LABELS = {
    "multihead_reference": "Multi-head normal",
    "multihead_calibrated": "Multi-head calibrated",
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "calibrated_narrow": "Calibrated narrow",
    "blockshuffle": "Plain BlockShuffle",
}


def report():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    initial = json.loads((ROOT / "preflight.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 6
    assert sha256(PLAN) == protocol["plan_sha256"] == result["plan_sha256"]
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    baseline = json.loads((ROOT / "baseline_initial.json").read_text())
    assert baseline["status"] == "PASS" and all(
        baseline["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
        for n in COMPUTATION
    )
    rows, paths = {}, {}
    for recipe in CONTROLS:
        rows[recipe] = []
        for rate in RATES:
            path = control_path(recipe, rate)
            m = json.loads((path / "metrics.json").read_text())
            frozen = next(t for t in initial["controls"] if t["run"] == path.name)
            assert sha256(path / "metrics.json") == frozen["metrics_sha256"]
            assert sha256(path / "checkpoint.pt") == frozen["checkpoint_sha256"]
            assert sha256(path / "source.zip") == frozen["source_archive_sha256"]
            verify_archive(path, m)
            rows[recipe].append(m)
            paths[m["run"]] = path
    diagnostics = {}
    for recipe in RECIPES:
        rows[recipe] = []
        for rate in RATES:
            path = output_path(recipe, rate)
            m = verify_trial(recipe, rate, initial)
            frozen = next(t for t in result["trials"] if t["run"] == path.name)
            assert sha256(path / "metrics.json") == frozen["metrics_sha256"]
            assert sha256(path / "checkpoint.pt") == frozen["checkpoint_sha256"]
            assert all(
                m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
                for n in (*COMPUTATION, "src/core/multihead_screen.py")
            )
            rows[recipe].append(m)
            paths[m["run"]] = path
            d = json.loads((path / "final_diagnostics.json").read_text())
            routed = [v for v in d.values() if "routing_finite" in v]
            diagnostics[m["run"]] = {
                "entropy_range": [
                    min(v["routing_normalized_entropy_mean"] for v in routed),
                    max(v["routing_normalized_entropy_mean"] for v in routed),
                ],
                "max_saturation": max(v["routing_saturation_fraction"] for v in routed),
                "min_mass": min(v["routing_weight_sum_mean"] for v in routed),
                "clip_fraction": m["clipped_step_fraction"],
                "training_tokens_per_second_descriptive": m["training_tokens_per_second"],
            }
    selected = {r: select_best(v) for r, v in rows.items()}
    references = {r: selected[r] for r in CONTROLS}
    decisions = {r: candidate_gates(selected[r], references) for r in RECIPES}
    same_rate = {
        r: {
            str(round(rate * 1e6)): {
                c: 100 * (rows[r][i]["validation_loss"] / rows[c][i]["validation_loss"] - 1)
                for c in CONTROLS
            }
            for i, rate in enumerate(RATES)
        }
        for r in RECIPES
    }
    assert (
        decisions == result["decisions"] and same_rate == result["same_rate_relative_nll_percent"]
    )
    assert {r: selected[r]["run"] for r in RECIPES} == result["selected"]
    assert {r: selected[r]["run"] for r in CONTROLS} == result["selected_controls"]
    lines = [
        "# H046: Multi-head FFN validation screen",
        "",
        "**Both local adaptations fail quality and memory promotion.** Calibrating initialization helps at every tested rate, but the best calibrated result still trails all four selected controls. This is a small-head, 200-step result, not a refutation of the published architecture at its larger dimensions or training budgets.",
        "",
        "[Frozen protocol](multihead_screen_plan.md), [raw decisions](../results/multihead_screen_v2/result.json), [initialization qualification](multihead_integration_results.md), [budget derivation](multihead_budget_geometry.md).",
        "",
        "All trials use WikiText-2, width 384, eight layers, context 128, batch 16, seed 17, native BF16, 409,600 sampled training tokens and all 322,688 validation targets. Six new trials use 2,457,600 training tokens in total. Official test is unscored. Each initialization receives three rates independently; twelve frozen control trials are reused. Historical architecture-search effort is unequal.",
        "",
        "| Recipe | NLL at 0.0003 | NLL at 0.0006 | NLL at 0.0012 | Selected rate | FFN weights | Total weights | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r, cells in rows.items():
        s = selected[r]
        values = " | ".join(f"{m['validation_loss']:.6f}" for m in cells)
        lines.append(
            f"| {LABELS[r]} | {values} | {s['training']['learning_rate']:.4f} | {s['ffn_parameters']:,} | {s['total_parameters']:,} | {s['peak_allocated_vram_bytes'] / 2**20:.2f} |"
        )
    lines += [
        "",
        "## Same-rate differences",
        "",
        "Positive values mean worse candidate NLL. The complete rate matrix prevents selection from hiding a rate-specific failure.",
        "",
        "| Candidate | Rate | vs full SwiGLU | vs full GELU | vs narrow | vs BlockShuffle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r, rates in same_rate.items():
        for rate, cells in rates.items():
            values = " | ".join(f"{cells[c]:+.4f}%" for c in CONTROLS)
            lines.append(f"| {LABELS[r]} | {int(rate) / 1e6:.4f} | {values} |")
    lines += ["", "## Gates and diagnosis", "", "| Gate | Normal | Calibrated |", "|---|---|---|"]
    for gate in decisions[RECIPES[0]]["gates"]:
        values = " | ".join("PASS" if decisions[r]["gates"][gate] else "FAIL" for r in RECIPES)
        lines.append(f"| {gate.replace('_', ' ')} | {values} |")
    lines += [
        "",
        "Both winners are at the upper grid boundary; neither optimum is bracketed. No post-result rate extension or automatic 800-step promotion is performed. Normal-versus-calibrated initialization changes parameter scales and optimization geometry, so the benefit cannot be assigned solely to output variance.",
        "",
        "| Candidate | Rate | Final layer entropy range (nats) | Maximum sigmoid saturation | Clipped steps | Training tokens/s (descriptive) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in RECIPES:
        for m in rows[r]:
            d = diagnostics[m["run"]]
            lo, hi = d["entropy_range"]
            lines.append(
                f"| {LABELS[r]} | {m['training']['learning_rate']:.4f} | {lo:.4f} to {hi:.4f} | {100 * d['max_saturation']:.3f}% | {100 * d['clip_fraction']:.1f}% | {d['training_tokens_per_second_descriptive']:,.0f} |"
            )
    lines += [
        "",
        "Routing diagnostics sample the first 256 tokens per layer outside timed training. All recorded activations, routing statistics and gradient norms are finite. This does not provide a lower gradient bound or establish whole-network stability. Training throughput is recorded for transparency; the archived controls were timed in different sessions, so no causal speed comparison is made. No FlashMHF kernel is implemented.",
        "",
        "![All rates and selected training curves](../results/plots/multihead_screen.png)",
        "",
        "## Verification and decision",
        "",
        "All six checkpoint/config counts, histories, data hashes, archives, optimizer metadata and final sampling RNG match the frozen protocol. The final sampling state equals all twelve controls. Current-code initial validation losses reproduced all four controls within 1e-7. The first baseline worker crashed during PyTorch import before evaluation or candidate training; its explicitly recorded, source-identical continuation completed all six trials without retries. Both records remain retained. Root cause of the intermittent native import failures remains unresolved.",
        "",
        "Retain this small-budget parallel adaptation as negative local evidence. The budget derivation identifies a separately testable simplification: one headwise SwiGLU, no router, twice the head width, and exactly the BlockShuffle parameter budget. It halves the parallel model's hidden feature count. That changes several capacity dimensions, so any gain would be a recipe comparison rather than an isolated router ablation. The stronger plain 800-step result and the corrected 0.101% three-seed activation benefit are unchanged.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.multihead_screen_report",
        "```",
        "",
    ]
    Path("research/multihead_screen_results.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for r, cells in rows.items():
        axes[0].plot(RATES, [m["validation_loss"] for m in cells], "o-", label=LABELS[r])
        s = selected[r]
        history = [
            json.loads(v) for v in (paths[s["run"]] / "history.jsonl").read_text().splitlines()
        ]
        h = [v for v in history if v["step"] >= 50]
        axes[1].plot(
            [v["step"] for v in h], [v["validation_loss"] for v in h], "o-", label=LABELS[r]
        )
    axes[0].set(
        xlabel="Peak learning rate",
        ylabel="Final full-validation NLL",
        title="All frozen rate cells",
        xticks=RATES,
    )
    axes[1].set(
        xlabel="Optimizer step",
        ylabel="Full-validation NLL",
        title="Selected rates; steps 50 to 200",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(f"results/plots/multihead_screen.{ext}", dpi=150)
    plt.close(fig)
    summary = {
        "status": "FAIL",
        "selected_nll": {r: m["validation_loss"] for r, m in selected.items()},
        "decisions": decisions,
        "diagnostics": diagnostics,
        "all_artifacts_verified": True,
        "plan_sha256": sha256(PLAN),
        "result_sha256": sha256(ROOT / "result.json"),
    }
    write_json(Path("results/multihead_screen_summary.json"), summary)
    print(
        json.dumps({"status": summary["status"], "selected_nll": summary["selected_nll"]}, indent=2)
    )


if __name__ == "__main__":
    report()
