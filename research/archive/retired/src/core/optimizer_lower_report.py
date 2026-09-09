"""Verify the completed H049 four-rate comparison and its three new controls."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.optimizer_bracket_report import LABELS
from src.core.optimizer_lower import (
    CRITICAL,
    PLAN,
    RATES,
    decisions,
    numerical_failure,
    output_path,
    preflight,
    same_rates,
    verify_lower,
)
from src.core.reproducibility import sha256, write_json
from src.core.wikitext_screen import select_best

ROOT = Path("results/optimizer_lower_v1")


def load_verified():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    initial = json.loads((ROOT / "preflight.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 3
    assert sha256(PLAN) == protocol["plan_sha256"] == result["plan_sha256"]
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    checked, matrix = preflight()
    assert checked == initial
    for t in result["trials"]:
        p = output_path(t["recipe"], RATES[0])
        if t["status"] == "NUMERICAL_FAIL":
            assert numerical_failure(p) and sha256(p / "failure.json") == t["failure_sha256"]
            continue
        m = verify_lower(t["recipe"], initial["prior_initial"])
        assert (
            sha256(p / "metrics.json") == t["metrics_sha256"]
            and sha256(p / "checkpoint.pt") == t["checkpoint_sha256"]
        )
        assert all(
            m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        matrix[t["recipe"]].append(m)
    assert (
        decisions(matrix) == result["decision"]
        and same_rates(matrix) == result["same_rate_relative_nll_percent"]
    )
    return result, initial, matrix


def report():
    result, initial, matrix = load_verified()
    decision = result["decision"]
    selected = {r: select_best(v) for r, v in matrix.items()}
    passed = decision["full_promotion_passes"]
    lines = [
        "# H049: Complete four-rate optimizer comparison",
        "",
        f"**Local gold gates: {'PASS' if passed else 'FAIL'}.** Three new lower-rate dense controls complete a sixteen-cell grid with equal four-rate selection per recipe. H048's original three-rate result remains unchanged; the old plain 0.0006 cell is explicitly added here.",
        "",
        "[Frozen H049 plan](optimizer_lower_plan.md), [raw decisions](../results/optimizer_lower_v1/result.json), [H048 higher-rate result](optimizer_bracket_results.md), [decay-only accounting](optimizer_rate_geometry.md).",
        "",
        "## Every rate and selected recipe",
        "",
        "| Recipe | NLL at 0.0006 | NLL at 0.0012 | NLL at 0.0024 | NLL at 0.0048 | Selected rate | Grid position |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r, cells in matrix.items():
        values = []
        for rate in RATES:
            m = next((m for m in cells if m["training"]["learning_rate"] == rate), None)
            values.append(f"{m['validation_loss']:.6f}" if m else "NUMERICAL FAIL")
        s = selected[r]
        lines.append(
            f"| {LABELS[r]} | {' | '.join(values)} | {s['training']['learning_rate']:.4f} | {decision['grid_positions'][r].replace('_', ' ')} |"
        )
    lines += [
        "",
        "All trials use width 384, eight layers, context 128, batch 16, seed 17, BF16 and 800 steps, with 1,638,400 sampled training tokens and all 322,688 validation targets. H049 adds 4,915,200 tokens across three complete allocations; thirteen earlier cells are reused. Only global peak LR changes within a recipe. Initialization, optimizer/decay calibration, recomputation, model dimensions, data order and schedule length remain fixed. Official test stays unscored. Historical architecture/optimizer search effort is unequal.",
        "",
        "An interior winner has measured grid points on both sides. This is a discrete bracket, not proof of a global continuous optimum. A single-seed rate search also does not establish convergence or rate ranking across seeds.",
        "",
        "## Same-rate versus selected-recipe comparisons",
        "",
        "Positive numbers mean worse BlockShuffle NLL. A missing numerical cell has no final score.",
        "",
        "| Comparison | vs full SwiGLU | vs full GELU | vs calibrated narrow |",
        "|---|---:|---:|---:|",
    ]
    order = ("full_swiglu", "full_gelu", "calibrated_narrow")
    for rate, cells in result["same_rate_relative_nll_percent"].items():
        values = " | ".join(
            f"{cells[r]:+.4f}%" if cells[r] is not None else "unavailable" for r in order
        )
        lines.append(f"| Same LR {int(rate) / 1e6:.4f} | {values} |")
    values = " | ".join(f"{decision['selected_relative_nll_percent'][r]:+.4f}%" for r in order)
    lines.append(f"| Selected rates | {values} |")
    lines += ["", "| Frozen gate | Decision |", "|---|---|"]
    for k, v in decision["gates"].items():
        lines.append(f"| {k.replace('_', ' ')} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        f"Separate >=0.2% selected-narrow margin: **{'PASS' if decision['narrow_margin_at_least_point_two_percent'] else 'FAIL'}**. These are engineering gates, not significance tests.",
        "",
        "## New controls and retained resource evidence",
        "",
        "| New lower-rate control | NLL | Peak MiB | Clipped steps | Training tokens/s (descriptive) | Evidence |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for t in result["trials"]:
        if t["status"] == "NUMERICAL_FAIL":
            lines.append(
                f"| {LABELS[t['recipe']]} | NUMERICAL FAIL | unavailable | unavailable | unavailable | [failure](../results/runs/{t['run']}/failure.json) |"
            )
        else:
            m = next(m for m in matrix[t["recipe"]] if m["run"] == t["run"])
            lines.append(
                f"| {LABELS[t['recipe']]} | {m['validation_loss']:.6f} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% | {m['training_tokens_per_second']:,.0f} | [metrics](../results/runs/{m['run']}/metrics.json) |"
            )
    lines += [
        "",
        "| Selected recipe | FFN weights | Total weights | FFN matrix FLOPs/token | Peak MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for r, m in selected.items():
        lines.append(
            f"| {LABELS[r]} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['ffn_matrix_forward_flops_per_token']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} |"
        )
    lines += [
        "",
        "Matrix work excludes nonlinearities and other non-matrix operations. Cross-session training timings are descriptive, not paired speed measurements. Fewer parameters do not by themselves establish faster training or inference.",
        "",
        "![All four rates and selected trajectories](../results/plots/optimizer_lower.png)",
        "",
        "## Verification and interpretation",
        "",
        "All thirteen retained cells verify before new training. The original plain 0.0006 checkpoint and metrics match their H039 hashes. H048's source archives, twelve cells and initial-function reproductions remain intact. Current computation matches the 223-test snapshot; no NN, trainer, data, optimizer or diagnostic code changes in this round. Each new run independently reproduces its reference initial NLL within 1e-7. Source archives, all eight data hashes, exact validation stream, schedules, optimizer metadata, finite layer/gradient histories, checkpoint counts/weights and final sampling states verify.",
        "",
    ]
    failures = len(list(ROOT.glob("failure_attempt_*.json")))
    lines.append(
        f"Numerical cell failures: {result['numerical_failures']}; preserved infrastructure audit failures: {failures}. Completed and numerically failed trials are never rerun."
    )
    lines.append("")
    if passed and decision["all_selected_rates_unchanged_at_0012"]:
        lines.append(
            "The stronger plain recipe survives the completed four-rate comparison, with every recipe still selected at 0.0012. Its existing seed-29/43 results remain the historical independent-initialization evidence; there is no reason to repeat identical runs. The newly expanded rate ranking itself has only seed-17 evidence. The next step is a separately frozen longer-budget comparison with the full and calibrated narrow controls, preserving the selected optimizer treatments. Late loss reductions in the 800-step curves show why a fixed-budget win is not converged superiority."
        )
    elif passed:
        lines.append(
            "The local gates survive, but selected rates change. Freeze independent-seed controls for changed recipes before a replication claim. A boundary winner remains unbracketed, and no continuous optimum or convergence claim follows."
        )
    else:
        lines.append(
            "The lower controls remove local promotion. Qualify the older comparison as a finding within its rate grid; do not treat it as superiority over the newly selected controls. Preserve the complete grid and investigate the remaining quality/resource gap before further promotion."
        )
    lines += [
        "",
        "The learnable affine activation's corrected 0.101% mean benefit at 0.0012 is unchanged. These optimizer controls do not make it a material gain, prove a new nonlinear primitive or establish a whole-network gradient lower bound.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.optimizer_lower_report",
        "```",
        "",
    ]
    Path("research/optimizer_lower_results.md").write_text("\n".join(lines), encoding="utf-8")
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
        title="Complete four-rate grid; seed 17",
        xticks=RATES,
    )
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[1].set(
        xlabel="Optimizer step", ylabel="Full-validation NLL", title="Selected 800-step recipes"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(f"results/plots/optimizer_lower.{ext}", dpi=150)
    plt.close(fig)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "decision": decision,
        "selected_nll": {r: m["validation_loss"] for r, m in selected.items()},
        "same_rate_relative_nll_percent": result["same_rate_relative_nll_percent"],
        "all_artifacts_verified": True,
        "new_completed_trials": 3 - result["numerical_failures"],
        "numerical_failures": result["numerical_failures"],
        "infrastructure_failures": failures,
        "plan_sha256": sha256(PLAN),
        "result_sha256": sha256(ROOT / "result.json"),
    }
    write_json(Path("results/optimizer_lower_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report()
