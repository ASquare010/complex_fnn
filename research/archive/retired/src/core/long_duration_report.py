"""Verify H050 duration evidence without upgrading a late plateau into convergence."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.core.long_duration import (
    CRITICAL,
    PLAN,
    RECIPES,
    decisions,
    numerical_failure,
    output_path,
    preflight,
    verify_trial,
)
from src.core.optimizer_bracket_report import LABELS
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/long_duration_v1")


def load_verified():
    result = json.loads((ROOT / "result.json").read_text())
    initial = json.loads((ROOT / "preflight.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == 4
    assert sha256(PLAN) == result["plan_sha256"] == protocol["plan_sha256"]
    assert preflight() == initial
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    sampling = json.loads((ROOT / "sampling/result.json").read_text())
    assert sampling["status"] == "PASS" and sampling["generator_device"] == "cuda"
    assert sampling["optimizer_updates"] == sampling["validation_targets_scored"] == 0
    assert (
        sampling["training_batch_calls"] == 3200
        and sampling["data_hashes"] == initial["data_hashes"]
    )
    assert sampling["after_800_batches_sha256"] == initial["sampling_800_sha256"]
    assert all(
        sampling["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
        for n in CRITICAL
    )
    assert sha256(ROOT / "sampling/states.pt") == sampling["states_sha256"]
    states = torch.load(ROOT / "sampling/states.pt", map_location="cpu", weights_only=True)
    for count in (800, 3200):
        assert (
            hashlib.sha256(states[f"after_{count}_batches"].numpy().tobytes()).hexdigest()
            == sampling[f"after_{count}_batches_sha256"]
        )
    rows, trajectories = {}, {}
    for t in result["trials"]:
        p = output_path(t["recipe"])
        if t["status"] == "NUMERICAL_FAIL":
            assert numerical_failure(p) and sha256(p / "failure.json") == t["failure_sha256"]
            continue
        m, tr = verify_trial(t["recipe"], initial, sampling)
        assert (
            sha256(p / "metrics.json") == t["metrics_sha256"]
            and sha256(p / "checkpoint.pt") == t["checkpoint_sha256"]
        )
        assert all(
            m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        rows[t["recipe"]] = m
        trajectories[t["recipe"]] = tr
    assert decisions(rows) == result["decision"] and trajectories == result["trajectory_screens"]
    assert result["all_late_plateau_screens_pass"] == (
        len(trajectories) == 4
        and all(v["late_plateau_screen_passes"] for v in trajectories.values())
    )
    return result, initial, rows, trajectories


def report():
    result, initial, rows, trajectories = load_verified()
    d = result["decision"]
    passed = d["full_promotion_passes"]
    plateau = result["all_late_plateau_screens_pass"]
    short = {
        r: json.loads((Path("results/runs") / v["run"] / "metrics.json").read_text())
        for r, v in initial["references"].items()
    }
    lines = [
        "# H050: Fixed-recipe 3,200-step comparison",
        "",
        f"**Local quality/memory promotion: {'PASS' if passed else 'FAIL'}. Operational late-plateau screen: {'PASS' if plateau else 'FAIL'}.** Neither outcome is a proof of optimal convergence. This round trains the four selected recipes for four times the earlier duration, with unchanged peak rate and optimizer treatment.",
        "",
        "[Frozen protocol](long_duration_plan.md), [raw decisions](../results/long_duration_v1/result.json), [complete 800-step rate grid](optimizer_lower_results.md).",
        "",
        "All models use width 384, eight layers, context 128, batch 16, vocabulary 4096, seed 17, native BF16 and peak LR 0.0012. Only steps and proportional logging change from the selected short recipes: 3,200 steps and logging every 800. The relative schedule stretches to 320 warmup steps and cosine decay to 0.1 peak. Each complete trial trains 6,553,600 sampled tokens, about 2.125 cache-token exposures; replacement sampling is not an ordered epoch. Four allocations total 26,214,400 tokens. Every evaluation scores all 322,688 validation targets. Official test stays unscored.",
        "",
        "## Final endpoints and resources",
        "",
        "| Recipe | Earlier 800-step NLL | New 3,200-step NLL | FFN weights | Total weights | Peak MiB | Clipped steps | Training tokens/s (descriptive) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in RECIPES:
        m = rows.get(r)
        if m is None:
            lines.append(
                f"| {LABELS[r]} | {short[r]['validation_loss']:.6f} | NUMERICAL FAIL | unavailable | unavailable | unavailable | unavailable | unavailable |"
            )
        else:
            lines.append(
                f"| {LABELS[r]} | {short[r]['validation_loss']:.6f} | {m['validation_loss']:.6f} | {m['ffn_parameters']:,} | {m['total_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% | {m['training_tokens_per_second']:,.0f} |"
            )
    lines += [
        "",
        "Earlier and new final endpoints use different token budgets and schedule lengths. The 800-step point inside a new run is not equivalent to a separately trained 800-step model: warmup/decay differ. Within the four new runs, duration and data order match. Source/checkpoint metadata retain logical matrix FLOPs, parameter bytes and optimizer-state bytes; these do not imply a training-speed win. Cross-session throughput is descriptive.",
        "",
        "## Frozen final-step gates",
        "",
    ]
    if d["all_four_complete"]:
        lines += ["| Gate | Decision |", "|---|---|"]
        for k, v in d["gates"].items():
            lines.append(f"| {k.replace('_', ' ')} | {'PASS' if v else 'FAIL'} |")
        lines += ["", "| BlockShuffle relative final NLL | Difference |", "|---|---:|"]
        for r, v in d["relative_nll_percent"].items():
            lines.append(f"| vs {LABELS[r]} | {v:+.4f}% |")
        lines += [
            "",
            f"Separate >=0.2% narrow margin: **{'PASS' if d['narrow_margin_at_least_point_two_percent'] else 'FAIL'}**. Positive relative NLL means worse. These are engineering gates, not statistical significance tests.",
        ]
    else:
        lines.append(
            "At least one recipe lacks a complete finite final score. The four-way promotion is ineligible; preserve each numerical failure without assigning it a fabricated NLL."
        )
    lines += [
        "",
        "## Late trajectory evidence",
        "",
        "The frozen diagnostic requires absolute NLL change from 2,400 to 3,200 steps <=0.2%, and final NLL <=0.2% above the best recorded endpoint, for every recipe. This is a limited operational screen, not a convergence theorem or proof that more training/rate tuning cannot help.",
        "",
        "| Recipe | Relative NLL change, last 800 steps | Final cost over best recorded endpoint | Late screen |",
        "|---|---:|---:|---|",
    ]
    for r, t in trajectories.items():
        lines.append(
            f"| {LABELS[r]} | {t['relative_last_800_step_nll_change_percent']:+.4f}% | {t['relative_final_cost_over_best_recorded_percent']:+.4f}% | {'PASS' if t['late_plateau_screen_passes'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Negative last-interval changes indicate continued improvement during the schedule. The primary quality decision always uses the final endpoint, not whichever intermediate validation loss was best.",
        "",
        "![Within-duration learning and reference endpoint comparison](../results/plots/long_duration.png)",
        "",
        "## Verification",
        "",
        "H049's complete sixteen-cell grid verifies and retains an interior 0.0012 winner for every recipe. Current computation matches the passing 225-test snapshot, with no NN, trainer, data, optimizer or diagnostic source change. A separate read-only GPU worker reconstructs 3,200 CUDA sampler calls: the first batch matches the retained qualification hash, its 800-call state matches every selected old checkpoint, and all complete new checkpoints match its 3,200-call state. CPU and CUDA generators are not substituted for each other. That worker performs zero optimizer updates and scores no validation targets.",
        "",
        "All eight immutable data hashes, the complete validation stream including the final nine-window batch, exact source archives, checkpoint counts/weights and configurations verify. Each new initial full-validation loss and first pre-update training loss reproduce their reference within 1e-7; the first optimizer update then differs because warmup is stretched. Full histories, actual schedules, optimizer metadata and finite gradients/layer statistics verify. No bitwise long-trajectory or full-network gradient lower-bound claim is made.",
        "",
    ]
    failures = len(list(ROOT.glob("failure_attempt_*.json")))
    lines.append(
        f"Numerically failed cells: {result['numerical_failures']}; preserved infrastructure audit failures: {failures}. Every training log, failure, checkpoint and source record remains retained. Completed or numerically failed runs are not repeated."
    )
    lines += ["", "## Next decision", ""]
    if passed:
        lines.append(
            "The compressed recipe retains the local quality/memory gates at this longer fixed duration. Independent-seed longer controls must be frozen before a replication claim. The rate grid was selected at 800 steps; duration-specific rate ranking is not established. A failed late-plateau screen requires continued convergence qualification rather than a claim that the quality is final."
        )
    elif d["quality_passes"]:
        lines.append(
            "Longer-duration quality passes but the resource requirement does not. Preserve the quality evidence and require a separate execution investigation before promotion."
        )
    else:
        lines.append(
            "This longer fixed recipe does not retain the local quality pass. Qualify the shorter-budget advantage and inspect optimization/representation evidence before promotion. This is not a family-wide rejection: the longer-duration optimum and independent-seed behavior remain unmeasured."
        )
    lines += [
        "",
        "The earlier same-rate affine activation result is unchanged; no learnable-activation variant is added to this duration comparison. No new primitive, universal optimality, state-of-the-art superiority or breakthrough is established by these local controls.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.long_duration_report",
        "```",
        "",
    ]
    Path("research/long_duration_results.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for r, m in rows.items():
        h = [json.loads(v) for v in (output_path(r) / "history.jsonl").read_text().splitlines()]
        h = [v for v in h if v["step"] >= 800]
        axes[0].plot(
            [v["step"] for v in h], [v["validation_loss"] for v in h], "o-", label=LABELS[r]
        )
        axes[1].plot(
            [800, 3200], [short[r]["validation_loss"], m["validation_loss"]], "o-", label=LABELS[r]
        )
    axes[0].set(
        xlabel="Optimizer step",
        ylabel="Full-validation NLL",
        title="Within new 3,200-step schedules",
    )
    axes[1].set(
        xlabel="Separate schedule duration (steps)",
        ylabel="Final full-validation NLL",
        title="Final endpoints; different training budgets",
        xticks=[800, 3200],
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        if rows:
            ax.legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(f"results/plots/long_duration.{ext}", dpi=150)
    plt.close(fig)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "decision": d,
        "final_nll": {r: m["validation_loss"] for r, m in rows.items()},
        "trajectory_screens": trajectories,
        "all_late_plateau_screens_pass": plateau,
        "all_artifacts_verified": True,
        "new_completed_trials": len(rows),
        "numerical_failures": result["numerical_failures"],
        "infrastructure_failures": failures,
        "plan_sha256": sha256(PLAN),
        "result_sha256": sha256(ROOT / "result.json"),
    }
    write_json(Path("results/long_duration_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report()
