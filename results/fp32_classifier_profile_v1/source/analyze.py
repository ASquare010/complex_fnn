"""Apply the frozen H114 gates to all fixtures, with no outlier removal."""

import csv
import gzip
import io
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("results/fp32_classifier_profile_v1")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def describe(values):
    return dict(
        n=len(values),
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values) if len(values) > 1 else None,
        min=min(values),
        max=max(values),
    )


def table(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))


def analyze():
    result, audit = read(ROOT / "result.json"), read(ROOT / "completion_audit.json")
    assert result["status"] == "COMPLETE" and audit["passed"]
    audit_rows = {r["label"]: r for r in audit["rows"]}
    cases = result["cases"]
    rows, timings = [], []
    for case in cases:
        f = case["fixture"]
        ref = next(
            c for c in cases if c["fixture"]["label"] == f["label"] and c["policy"] == "block_bf16"
        )
        a, ar = audit_rows[case["label"]], audit_rows[ref["label"]]
        error = a["probe_error_to_native_fp32"]
        row = dict(
            label=case["label"],
            fixture=f["label"],
            scope=f["scope"],
            dataset=f["dataset"],
            seed=f["seed"],
            source_policy=f["source_policy"],
            source_step=f["source_step"],
            policy=case["policy"],
            batch=f["batch"],
            context=f["context"],
            total_parameters=case["parameter_count"],
            ffn_parameters=case["ffn_parameters"],
            peak_job_mib=case["peak_job_allocated_bytes"] / 2**20,
            peak_reserved_mib=case["peak_job_reserved_bytes"] / 2**20,
            training_peak_mib=case["peak_training_allocated_bytes"] / 2**20,
            memory_ratio=case["peak_job_allocated_bytes"] / ref["peak_job_allocated_bytes"],
            wall_median_ms=case["timing"]["wall_update_ms"]["median"],
            wall_ratio=case["timing"]["wall_update_ms"]["median"]
            / ref["timing"]["wall_update_ms"]["median"],
            event_median_ms=case["timing"]["event_update_ms"]["median"],
            event_ratio=case["timing"]["event_update_ms"]["median"]
            / ref["timing"]["event_update_ms"]["median"],
            timing_stability_ratio=case["timing_stability_ratio"],
            reference_timing_stability_ratio=ref["timing_stability_ratio"],
            initial_native_nll=a["source_native_validation"]["nll"],
            final_native_nll=a["final_native_validation"]["nll"],
            final_nll_ratio=a["final_native_validation"]["nll"]
            / ar["final_native_validation"]["nll"],
            initial_loss_relative_error=error["relative_loss_difference"],
            global_gradient_relative_error=error["global_relative_l2"],
            max_tensor_gradient_relative_error=error["max_tensor_relative_l2"],
            all_finite=case["weights_gradients_moments_finite"],
            clip_fraction=case["clip_fraction"],
        )
        rows.append(row)
        for record in case["records"]:
            timings.append(
                dict(
                    label=case["label"],
                    scope=f["scope"],
                    policy=case["policy"],
                    **{
                        k: record[k]
                        for k in (
                            "step",
                            "warmup",
                            "loss",
                            "preclip_norm",
                            "wall_update_ms",
                            "event_update_ms",
                            "event_forward_ms",
                            "event_backward_ms",
                            "event_optimizer_ms",
                            "peak_allocated_bytes",
                            "peak_reserved_bytes",
                            "tokens_hash",
                            "targets_hash",
                        )
                    },
                )
            )
    metrics = (
        "peak_job_mib",
        "peak_reserved_mib",
        "memory_ratio",
        "wall_median_ms",
        "wall_ratio",
        "event_median_ms",
        "event_ratio",
        "timing_stability_ratio",
        "final_native_nll",
        "final_nll_ratio",
        "global_gradient_relative_error",
        "max_tensor_gradient_relative_error",
        "clip_fraction",
    )
    groups, decisions, by_seed = [], [], []
    for scope in dict.fromkeys(r["scope"] for r in rows):
        for policy in ("block_bf16", "chunks_bf16", "block_fp32", "chunks_fp32"):
            peers = [r for r in rows if r["scope"] == scope and r["policy"] == policy]
            groups.append(
                dict(
                    scope=scope,
                    policy=policy,
                    starting_states=len(peers),
                    training_seeds=len({p["seed"] for p in peers}),
                    metrics={k: describe([p[k] for p in peers]) for k in metrics},
                )
            )
            for seed in dict.fromkeys(r["seed"] for r in peers):
                selected = [p for p in peers if p["seed"] == seed]
                by_seed.append(
                    dict(
                        scope=scope,
                        policy=policy,
                        seed=seed,
                        starting_states=len(selected),
                        metrics={k: st.mean(p[k] for p in selected) for k in metrics},
                    )
                )
        peers = [r for r in rows if r["scope"] == scope and r["policy"] == "chunks_fp32"]
        gates = dict(
            all_finite=all(p["all_finite"] for p in peers),
            initial_loss_fidelity=all(p["initial_loss_relative_error"] <= 1e-6 for p in peers),
            initial_global_gradient_fidelity=all(
                p["global_gradient_relative_error"] <= 0.002 for p in peers
            ),
            initial_per_tensor_gradient_fidelity=all(
                p["max_tensor_gradient_relative_error"] <= 0.02 for p in peers
            ),
            all_memory_ratios_at_most_0_85=all(p["memory_ratio"] <= 0.85 for p in peers),
            median_wall_ratio_at_most_1_25=st.median(p["wall_ratio"] for p in peers) <= 1.25,
            all_reference_and_candidate_timing_stable=all(
                max(p["timing_stability_ratio"], p["reference_timing_stability_ratio"]) <= 1.25
                for p in peers
            ),
            every_native_nll_ratio_at_most_1_01=all(p["final_nll_ratio"] <= 1.01 for p in peers),
        )
        decisions.append(
            dict(
                scope=scope,
                gates=gates,
                passes_original_measured_gates=all(gates.values()),
                qualifies_fresh_lm_experiment=all(gates.values())
                and audit["full_gradient_replay_qualified"],
                gradient_replay_audit_hold=not audit["full_gradient_replay_qualified"],
                failed_gates=[k for k, v in gates.items() if not v],
            )
        )
    summary = dict(
        status="COMPLETE",
        study="H114",
        starting_states=14,
        cases=56,
        optimizer_updates=result["optimizer_updates"],
        profile_backward_passes=result["profile_backward_passes"],
        audit_backward_passes=audit["additional_backward_passes"],
        training_targets=result["training_targets"],
        elapsed_seconds=result["elapsed_seconds"],
        groups=groups,
        by_seed=by_seed,
        decisions=decisions,
        qualified_scopes=[d["scope"] for d in decisions if d["qualifies_fresh_lm_experiment"]],
        all_phase_peaks_recorded=True,
        correlated_starting_policies=True,
        full_controls_have_one_training_seed=True,
        fresh_training_runs=0,
        long_term_quality_proved=False,
        new_architecture=False,
        broad_goal_achieved=False,
        independent_audit={k: v for k, v in audit.items() if k not in ("rows", "boundaries")},
    )
    write(ROOT / "summary.json", summary)
    table(ROOT / "metrics.csv.gz", rows)
    table(ROOT / "updates.csv.gz", timings)
    print(
        json.dumps(dict(decisions=decisions, elapsed_seconds=result["elapsed_seconds"]), indent=2)
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "progress":
        files = sorted((ROOT / "runs").glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
        print("Completed cases:", len(files))
        for path in files[-4:]:
            row = read(path)
            print(
                row["label"],
                "MiB",
                row["peak_job_allocated_bytes"] / 2**20,
                "wall_ms",
                row["timing"]["wall_update_ms"]["median"],
                "stability",
                row["timing_stability_ratio"],
            )
    else:
        analyze()
