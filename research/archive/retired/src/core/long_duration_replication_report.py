"""Verify the frozen three-seed duration comparison, retaining every failed gate."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.long_duration_replication import (
    CRITICAL,
    ORDER,
    PLAN,
    RECIPES,
    SEEDS,
    output_path,
    preflight,
    summarize,
    verify_sampling,
    verify_seed17,
    verify_trial,
)
from src.core.optimizer_bracket import numerical_failure
from src.core.optimizer_bracket_report import LABELS
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/long_duration_replication_v1")


def load_verified():
    result = json.loads((ROOT / "result.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    initial = json.loads((ROOT / "preflight.json").read_text())
    assert result["status"] == "complete" and len(result["trials"]) == len(ORDER) == 8
    assert result["plan_sha256"] == protocol["plan_sha256"] == sha256(PLAN)
    assert preflight() == initial
    assert [(t["seed"], t["recipe"]) for t in result["trials"]] == list(ORDER)
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.read(PLAN.as_posix()) == PLAN.read_bytes()
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in protocol["provenance"]["source_files"].items()
        )
    transport_path = ROOT / "worker_transport_qualification_v1.json"
    transport = json.loads(transport_path.read_text())
    resume = json.loads((ROOT / "resume_2.json").read_text())
    assert sha256(transport_path) == resume["transport_qualification_sha256"]
    assert sha256(ROOT / "failure_attempt_1.json") == transport["failure_sha256"]
    assert (
        sha256(ROOT / "wikitext2_full_gelu_lr1200_s29_3200.attempt_1.log")
        == transport["failed_worker_log_sha256"]
    )
    assert sha256(Path("src/core/frozen_train_worker.py")) == transport["minimal_worker_sha256"]
    assert (
        sha256(Path("results/verification/long_duration_replication_resume_v1.py"))
        == transport["dispatcher_sha256"]
    )
    assert transport["protocol_sha256"] == sha256(ROOT / "protocol.json")
    assert transport["plan_sha256"] == sha256(PLAN)
    assert all(
        protocol["provenance"]["source_files"][n] == h == sha256(Path(n))
        for n, h in transport["critical_source_hashes"].items()
    )
    _, _, seed17, tr17 = verify_seed17()
    rows, trajectories = {17: seed17}, {"17": tr17}
    samplers = {s: verify_sampling(ROOT, s, initial, protocol) for s in SEEDS[1:]}
    for t in result["trials"]:
        s, r = t["seed"], t["recipe"]
        p = output_path(r, s)
        if t["status"] == "NUMERICAL_FAIL":
            assert numerical_failure(p) and sha256(p / "failure.json") == t["failure_sha256"]
            continue
        assert t["status"] == "COMPLETE"
        assert sha256(p / "metrics.json") == t["metrics_sha256"]
        assert sha256(p / "checkpoint.pt") == t["checkpoint_sha256"]
        m, tr = verify_trial(r, s, initial, samplers[s])
        assert (
            m["provenance"]["source_files"]["src/core/frozen_train_worker.py"]
            == transport["minimal_worker_sha256"]
        )
        assert all(
            m["provenance"]["source_files"][n] == protocol["provenance"]["source_files"][n]
            for n in CRITICAL
        )
        rows.setdefault(s, {})[r] = m
        trajectories.setdefault(str(s), {})[r] = tr
    assert summarize(rows) == result["summary"]
    assert trajectories == result["trajectory_screens"]
    assert result["all_late_plateau_screens_pass"] == (
        sum(len(v) for v in trajectories.values()) == 12
        and all(t["late_plateau_screen_passes"] for v in trajectories.values() for t in v.values())
    )
    return result, rows, trajectories


def report():
    result, rows, trajectories = load_verified()
    summary = result["summary"]
    passed = summary["all_seeds_pass_primary_gates"]
    lines = [
        "# H051: Three-seed 3,200-step comparison",
        "",
        f"**Primary quality/memory replication: {'PASS in every seed' if passed else 'FAIL'}. Late-plateau qualification: {'PASS' if result['all_late_plateau_screens_pass'] else 'FAIL'}.** The primary rule requires every seed individually; a passing average cannot rescue a failed seed. These are fixed-duration results, not converged or globally tuned superiority.",
        "",
        "[Frozen plan](long_duration_replication_plan.md), [raw evidence](../results/long_duration_replication_v1/result.json), [retained seed-17 comparison](long_duration_results.md).",
        "",
        "Seeds 29/43 add eight fresh trials, retaining all four seed-17 outcomes. Each uses width 384, eight layers, attention heads 6, context 128, vocabulary 4096, batch 16, native BF16, peak LR 0.0012, 320-step warmup and cosine decay to 0.1 peak. Only seed changes from H050. Each new model trains 6,553,600 sampled tokens; eight allocations total 52,428,800. Every evaluation scores all 322,688 validation targets. The primary endpoint is final step 3,200. Official test is unscored.",
        "",
        "## Every final endpoint",
        "",
        "| Seed | Recipe | NLL | FFN weights | Peak MiB | Clipped steps | Training tokens/s (descriptive) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in SEEDS:
        for r in RECIPES:
            if r not in rows.get(s, {}):
                lines.append(f"| {s} | {LABELS[r]} | numerical failure | - | - | - | - |")
                continue
            m = rows[s][r]
            lines.append(
                f"| {s} | {LABELS[r]} | {m['validation_loss']:.6f} | {m['ffn_parameters']:,} | {m['peak_allocated_vram_bytes'] / 2**20:.2f} | {100 * m['clipped_step_fraction']:.1f}% | {m['training_tokens_per_second']:,.0f} |"
            )
    lines += [
        "",
        "The complete source records retain total weights, logical matrix FLOPs, parameter and optimizer-state bytes. These counts do not imply a training-speed win. Throughput spans sessions and is descriptive; no paired speed claim follows.",
        "",
        "## Per-seed gates",
        "",
        "| Seed | Primary gates | NLL vs full SwiGLU | vs full GELU | vs narrow | >=0.2% narrow margin |",
        "|---|---|---:|---:|---:|---|",
    ]
    for s in SEEDS:
        d = summary["per_seed"][str(s)]
        if not d["all_four_complete"]:
            lines.append(f"| {s} | FAIL: incomplete | - | - | - | ineligible |")
            continue
        rel = d["relative_nll_percent"]
        lines.append(
            f"| {s} | {'PASS' if d['full_promotion_passes'] else 'FAIL'} | {rel['full_swiglu']:+.4f}% | {rel['full_gelu']:+.4f}% | {rel['calibrated_narrow']:+.4f}% | {'PASS' if d['narrow_margin_at_least_point_two_percent'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Positive relative NLL means worse. The original gates require >=70% fewer FFN weights, <=1% NLL cost versus both full controls, strictly beating narrow, finite verified diagnostics, and allocated peak <=1.1 times each full control. The 0.2% narrow margin is a separate, predeclared diagnostic.",
        "",
    ]
    for s in SEEDS:
        d = summary["per_seed"][str(s)]
        if d["all_four_complete"]:
            failed = [k.replace("_", " ") for k, v in d["gates"].items() if not v]
            if failed:
                lines.append(f"Seed {s} failed: {', '.join(failed)}.")
    if summary["all_twelve_complete"]:
        lines += [
            "",
            "## Three-seed means and paired differences",
            "",
            "| Recipe | Mean NLL | Sample SD | Maximum peak MiB |",
            "|---|---:|---:|---:|",
        ]
        for r in RECIPES:
            lines.append(
                f"| {LABELS[r]} | {summary['mean_nll'][r]:.6f} | {summary['sample_sd_nll'][r]:.6f} | {max(rows[s][r]['peak_allocated_vram_bytes'] for s in SEEDS) / 2**20:.2f} |"
            )
        lines += [
            "",
            "| Plain relative to control | Mean NLL difference | Paired SD | Exploratory 95% t interval | Strict seed wins | Relative mean NLL |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for r, p in summary["paired"].items():
            lo, hi = p["exploratory_t95_interval"]
            lines.append(
                f"| {LABELS[r]} | {p['mean']:+.6f} | {p['sample_sd']:.6f} | [{lo:+.6f}, {hi:+.6f}] | {p['strict_candidate_wins']}/3 | {summary['relative_mean_nll_percent'][r]:+.4f}% |"
            )
        lines += [
            "",
            f"Separate >=0.2% mean narrow margin: **{'PASS' if summary['mean_narrow_margin_at_least_point_two_percent'] else 'FAIL'}**. Each paired difference is candidate minus control at the same seed. Intervals use n=3, df=2 and t=4.302653. These small-sample summaries are exploratory, not proof of universal superiority, equivalence or independently replicated rate ranking.",
        ]
    lines += [
        "",
        "## Late trajectory evidence",
        "",
        "| Seed | Recipe | Last-800-step NLL change | Final cost over best endpoint | Late screen |",
        "|---|---|---:|---:|---|",
    ]
    for s in SEEDS:
        for r, tr in trajectories.get(str(s), {}).items():
            lines.append(
                f"| {s} | {LABELS[r]} | {tr['relative_last_800_step_nll_change_percent']:+.4f}% | {tr['relative_final_cost_over_best_recorded_percent']:+.4f}% | {'PASS' if tr['late_plateau_screen_passes'] else 'FAIL'} |"
            )
    lines += [
        "",
        "The operational screen requires absolute last-800-step NLL change <=0.2% and final NLL <=0.2% above the best recorded endpoint. Negative change means continued improvement. Passing this screen still cannot prove optimal convergence. No 800-step checkpoint inside these stretched schedules is treated as equivalent to a separately trained shorter model.",
        "",
        "![Longer trajectories in each independent seed](../results/plots/long_duration_replication.png)",
        "",
        "## Verification and retained failures",
        "",
        "H050's four outcomes, the twelve selected 800-step reference cells, all immutable data hashes and the complete validation stream verify. Current computation matches the passing 227-test snapshot. Separate CUDA workers reproduce each new seed's retained 800-batch RNG state and reconstruct its 3,200-batch state; they perform no optimizer updates or validation scoring. Every completed checkpoint matches its sampler state. Initial validation NLL and first pre-update training loss reproduce the corresponding seed's short reference within 1e-7. Exact configurations, schedules, parameter/FLOP counts, optimizer/init metadata, source archives, finite weights/gradients/layer diagnostics and all checkpoint hashes verify.",
        "",
    ]
    starts = []
    for p in Path("results/verification").glob("long_duration_replication_launcher_v*.json"):
        r = json.loads(p.read_text())
        if r.get("returncode", 0) != 0:
            starts.append(p.as_posix())
    failures = len(list(ROOT.glob("failure_attempt_*.json")))
    lines += [
        f"Numerical cells: {result['numerical_failures']}. Preserved audit failures: {failures}. Failed launcher processes: {len(starts)} (including the audit failure propagated to its launcher; these counts overlap). The first launch crashed in Python inspect.cleandoc during Matplotlib import, before creating any audit directory or sampling/training output. Its [explicit source-identical continuation](../results/verification/long_duration_replication_startup_qualification_v1.json) retains the fatal log. The first training worker also crashed during Matplotlib import before creating a run. Its qualified continuation dispatches the identical frozen configuration to a [minimal worker](../src/core/frozen_train_worker.py) that directly calls the unchanged trainer and imports no report/plot driver. The original audit and its verification functions remain exact; the operational dispatcher change is explicitly recorded. No root cause or permanent fix is claimed. Numerical and infrastructure failures are distinct; every attempted run remains retained.",
        "",
        "## Next decision",
        "",
    ]
    lines.append(
        "All seeds retain the original local gates at this fixed duration. Preserve this replicated result while investigating duration-specific tuning, continued learning, broader scale/comparators and novelty before a breakthrough claim."
        if passed
        else "The single-seed H050 local pass does not replicate under the frozen all-seed rule. Preserve the earlier 800-step evidence with its duration limits. Investigate duration-specific optimization and representation before further promotion; this failure does not reject the whole structured family."
    )
    lines += [
        "",
        "The corrected same-rate affine activation benefit remains only 0.101% at 800 steps; this study adds no activation variant and supplies no new activation-specific evidence. No universal-optimality, converged-SOTA, novel-primitive or full-network nonvanishing-gradient claim follows.",
        "",
        "```powershell",
        "uv run --extra compile --extra data python -m src.core.long_duration_replication_report",
        "```",
        "",
    ]
    Path("research/long_duration_replication_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, s in zip(axes, SEEDS):
        for color_index, r in enumerate(RECIPES):
            if r not in rows.get(s, {}):
                continue
            h = [
                json.loads(v)
                for v in (output_path(r, s) / "history.jsonl").read_text().splitlines()
            ]
            h = [v for v in h if v["step"] >= 800]
            ax.plot(
                [v["step"] for v in h],
                [v["validation_loss"] for v in h],
                "o-",
                label=LABELS[r],
                color=f"C{color_index}",
            )
        ax.set(
            title=f"Seed {s}: 3,200-step schedule",
            xlabel="Optimizer step",
            ylabel="Full-validation NLL",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(f"results/plots/long_duration_replication.{ext}", dpi=150)
    plt.close(fig)
    output = {
        "status": "PASS" if passed else "FAIL",
        "summary": summary,
        "trajectory_screens": trajectories,
        "all_late_plateau_screens_pass": result["all_late_plateau_screens_pass"],
        "all_artifacts_verified": True,
        "new_completed_trials": sum(t["status"] == "COMPLETE" for t in result["trials"]),
        "numerical_failures": result["numerical_failures"],
        "audit_failures": failures,
        "failed_launches": starts,
        "plan_sha256": sha256(PLAN),
        "result_sha256": sha256(ROOT / "result.json"),
    }
    write_json(Path("results/long_duration_replication_summary.json"), output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    report()
