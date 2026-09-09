"""Verify and present every H048 optimizer-grid result, including numerical failures."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.optimizer_bracket import (
    CRITICAL,
    PLAN,
    RATES,
    RECIPES,
    numerical_failure,
    output_path,
    same_rate_comparisons,
    selected_decision,
    verify_run,
)
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import select_best

ROOT = Path("results/optimizer_bracket_v1")
LABELS = {
    "full_swiglu": "Full SwiGLU",
    "full_gelu": "Full GELU",
    "calibrated_narrow": "Calibrated narrow",
    "blockshuffle": "Plain BlockShuffle",
}


def load_verified():
    result = json.loads((ROOT / "result.json").read_text())
    initial = json.loads((ROOT / "preflight.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 8
    assert result["plan_sha256"] == protocol["plan_sha256"] == sha256(PLAN)
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    baseline = json.loads((ROOT / "baseline_initial.json").read_text())
    assert baseline["status"] == "PASS" and len(baseline["rows"]) == 4
    assert all(
        abs(r["initial_nll"] - r["reference_initial_nll"]) < 1e-7 and r["targets"] == 322688
        for r in baseline["rows"]
    )
    assert all(
        baseline["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
        for n in CRITICAL
    )
    matrix = {}
    for r in RECIPES:
        ref = initial["references"][r]
        p = output_path(r, RATES[0])
        assert sha256(p / "checkpoint.pt") == ref["checkpoint_sha256"]
        assert sha256(p / "source.zip") == ref["source_archive_sha256"]
        matrix[r] = [verify_run(r, RATES[0], initial)]
    for t in result["trials"]:
        p = output_path(t["recipe"], t["rate"])
        if t["status"] == "NUMERICAL_FAIL":
            assert numerical_failure(p) and sha256(p / "failure.json") == t["failure_sha256"]
            continue
        m = verify_run(t["recipe"], t["rate"], initial)
        assert sha256(p / "metrics.json") == t["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == t["checkpoint_sha256"]
        assert all(
            m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        matrix[t["recipe"]].append(m)
    assert selected_decision(matrix) == result["decision"]
    assert same_rate_comparisons(matrix) == result["same_rate_relative_nll_percent"]
    return result, initial, protocol, matrix


def report():
    result, initial, protocol, matrix = load_verified()
    decision = result["decision"]
    selected = {r: select_best(cells) for r, cells in matrix.items()}
    same_rate = result["same_rate_relative_nll_percent"]
    late = {}
    for r, m in selected.items():
        h = [
            json.loads(v)
            for v in (Path("results/runs") / m["run"] / "history.jsonl").read_text().splitlines()
        ]
        n600 = next(v["validation_loss"] for v in h if v["step"] == 600)
        late[r] = {
            "step600_nll": n600,
            "step800_nll": m["validation_loss"],
            "relative_final_200_step_improvement_percent": 100 * (1 - m["validation_loss"] / n600),
        }
    passed = decision["full_promotion_passes"]
    lines = [
        "# H048: Extended 800-step optimizer comparison",
        "",
        f"**Selected-recipe local promotion: {'PASS' if passed else 'FAIL'}.** The complete frozen grid gives each recipe rates 0.0012, 0.0024 and 0.0048. This tests stronger global-rate tuning at one seed and one duration; it does not establish convergence or a globally optimal rate.",
        "",
        "[Frozen protocol](optimizer_bracket_plan.md), [raw decisions](../results/optimizer_bracket_v1/result.json), [prior corrected three-seed comparison](affine_rate_replication_results.md).",
        "",
        "Eight new trials were allocated 800 steps each; four completed 0.0012 references are reused. Each complete trial trains 1,638,400 sampled tokens and scores all 322,688 validation targets. The new budget is at most 13,107,200 training tokens. Width 384, eight layers, context 128, batch 16, seed 17 and native BF16 are fixed. Official test remains unscored. Model code, data order, initialization, optimizer group treatment and recomputation are unchanged; only global peak LR differs within each recipe.",
        "",
        "## All primary rate cells",
        "",
        "| Recipe | NLL at 0.0012 | NLL at 0.0024 | NLL at 0.0048 | Selected rate | Grid position |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r, cells in matrix.items():
        entries = []
        for rate in RATES:
            m = next((m for m in cells if m["training"]["learning_rate"] == rate), None)
            entries.append(f"{m['validation_loss']:.6f}" if m else "NUMERICAL FAIL")
        s = selected[r]
        lines.append(
            f"| {LABELS[r]} | {' | '.join(entries)} | {s['training']['learning_rate']:.4f} | {decision['grid_positions'][r].replace('_', ' ')} |"
        )
    lines += [
        "",
        "Primary selection excludes the older plain 0.0006 cell and every affine/headwise trial. All historical results remain retained. Each primary recipe receives the same three global rates, but previous architecture/init/optimizer exploration effort differs. Narrow keeps width-calibrated down initialization/LR and product decay; BlockShuffle keeps factor LR calibration, parameter decay and native gate recomputation. Increasing global LR also increases AdamW's per-step decay amount.",
        "",
        "## Same-rate and selected-recipe differences",
        "",
        "Positive values mean worse BlockShuffle NLL. Comparing independently selected recipes asks a different question from comparing the same global rate.",
        "",
        "| Comparison | vs full SwiGLU | vs full GELU | vs calibrated narrow |",
        "|---|---:|---:|---:|",
    ]
    order = ("full_swiglu", "full_gelu", "calibrated_narrow")
    for rate, cells in same_rate.items():
        values = " | ".join(
            f"{cells[r]:+.4f}%" if cells[r] is not None else "unavailable" for r in order
        )
        lines.append(f"| Same rate {int(rate) / 1e6:.4f} | {values} |")
    values = " | ".join(f"{decision['selected_relative_nll_percent'][r]:+.4f}%" for r in order)
    lines.append(f"| Independently selected rates | {values} |")
    lines += ["", "| Local promotion gate | Decision |", "|---|---|"]
    for k, v in decision["gates"].items():
        lines.append(f"| {k.replace('_', ' ')} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        f"Separate >=0.2% margin over selected narrow: **{'PASS' if decision['narrow_margin_at_least_point_two_percent'] else 'FAIL'}**. These thresholds are engineering gates, not statistical significance tests.",
        "",
        "## Resources and late training progress",
        "",
        "| Selected recipe | FFN weights | Total weights | FFN forward matrix FLOPs/token | Peak MiB | Clipped steps | Training tokens/s (descriptive) | NLL improvement, steps 600-800 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r, m in selected.items():
        lines.append(
            f"| {LABELS[r]} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['ffn_matrix_forward_flops_per_token']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% | {m['training_tokens_per_second']:,.0f} | {late[r]['relative_final_200_step_improvement_percent']:+.4f}% |"
        )
    lines += [
        "",
        "The [decay-only accounting](optimizer_rate_geometry.md) quantifies the fixed recipes' different represented-map shrinkage without claiming a measured cause. Matrix counts exclude nonlinearities, norms, loss and optimizer work. Timing is retained for transparency, but references and new trials were measured in different sessions; there is no paired training-speed claim. A declining final segment is evidence of remaining learning during this schedule, not a measurement of converged quality. Three discrete rates do not prove a continuous optimum.",
        "",
        "![Rate response and selected trajectories](../results/plots/optimizer_bracket.png)",
        "",
        "## Every new trial",
        "",
        "| Recipe | Rate | Outcome | Final NLL | Peak MiB | Clip fraction | Evidence |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for t in result["trials"]:
        p = Path("results/runs") / t["run"]
        if t["status"] == "NUMERICAL_FAIL":
            lines.append(
                f"| {LABELS[t['recipe']]} | {t['rate']:.4f} | NUMERICAL FAIL | unavailable | unavailable | unavailable | [failure](../{p.as_posix()}/failure.json) |"
            )
        else:
            m = next(m for m in matrix[t["recipe"]] if m["run"] == t["run"])
            lines.append(
                f"| {LABELS[t['recipe']]} | {t['rate']:.4f} | COMPLETE | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% | [metrics](../{p.as_posix()}/metrics.json) |"
            )
    lines += [
        "",
        "## Verification and next decision",
        "",
        "Current computation matches the passing 220-test snapshot. All 30 prior-model CPU snapshots retain exact parameters, logits, loss, gradients and matrix-work counts. Historical registry/factory/count/diagnostic additions are explicitly qualified; shared training, data, optimizer, dense/BlockShuffle/grouped implementations and execution ASTs match reference archives. All four current initial validation NLLs reproduce the old values within 1e-7. Every complete trial's data, source archive, config, optimizer metadata, schedule, finite diagnostics, checkpoint weights/counts and final sampling RNG verify. The complete validation target stream includes the final nine-window batch.",
        "",
    ]
    failures = sorted(ROOT.glob("failure_attempt_*.json"))
    lines.append(
        f"Numerically failed grid cells: {result['numerical_failures']}. Preserved infrastructure audit failures: {len(failures)}. Any continuation records are separate from the frozen protocol; completed or numerically failed cells are never rerun. No final score is assigned to an incomplete process failure."
    )
    lines.append("")
    if passed:
        lines.append(
            "The compressed plain recipe survives this expanded rate comparison. All selected recipes are unchanged at 0.0012, so their existing seed-29/43 measurements remain the historical independent-initialization evidence; repeating those same runs would not add a new control. This is not a new independent replication of the higher-rate ranking. Every primary winner is on the lower boundary. A separately frozen lower-rate comparison is still needed before treating the global rate as bracketed, followed by longer-budget testing. This pass does not establish convergence, broad superiority, a new primitive or a whole-network gradient guarantee."
        )
    elif decision["quality_passes"]:
        lines.append(
            "Quality survives, but the resource requirement fails. Preserve the quality evidence and require a separate execution investigation before promotion; do not infer that lower parameter count gives lower practical memory or latency."
        )
    else:
        lines.append(
            "Expanded tuning removes the selected-recipe quality pass. Qualify the earlier 0.0012 result as a same-rate, fixed-budget finding. Do not promote the older comparison as evidence of superiority over the better-tuned full/narrow controls. Reconsider optimization and representation evidence before another architecture variant."
        )
    lines += [
        "",
        "The corrected 0.101% affine activation gain is measured at 0.0012 across three seeds. This grid does not test affine at higher rates or attribute any new rate gain to activation shape.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.optimizer_bracket_report",
        "```",
        "",
    ]
    Path("research/optimizer_bracket_results.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for r, cells in matrix.items():
        cells = sorted(cells, key=lambda m: m["training"]["learning_rate"])
        axes[0].plot(
            [m["training"]["learning_rate"] for m in cells],
            [m["validation_loss"] for m in cells],
            "o-",
            label=LABELS[r],
        )
        s = selected[r]
        h = [
            json.loads(v)
            for v in (Path("results/runs") / s["run"] / "history.jsonl").read_text().splitlines()
        ]
        h = [v for v in h if v["step"] >= 200]
        axes[1].plot(
            [v["step"] for v in h],
            [v["validation_loss"] for v in h],
            "o-",
            label=f"{LABELS[r]} @ {s['training']['learning_rate']:.4f}",
        )
    axes[0].set(
        xlabel="Global peak learning rate",
        ylabel="Final full-validation NLL",
        title="All primary rates; seed 17",
        xticks=RATES,
    )
    axes[1].set(
        xlabel="Optimizer step", ylabel="Full-validation NLL", title="Selected 800-step recipes"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(f"results/plots/optimizer_bracket.{ext}", dpi=150)
    plt.close(fig)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "decision": decision,
        "same_rate_relative_nll_percent": same_rate,
        "late_progress": late,
        "selected_nll": {r: m["validation_loss"] for r, m in selected.items()},
        "all_artifacts_verified": True,
        "new_completed_trials": 8 - result["numerical_failures"],
        "numerical_failures": result["numerical_failures"],
        "infrastructure_failures": len(failures),
        "plan_sha256": sha256(PLAN),
        "result_sha256": sha256(ROOT / "result.json"),
    }
    write_json(Path("results/optimizer_bracket_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report()
