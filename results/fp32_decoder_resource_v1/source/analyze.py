"""All-fixture resource comparisons and prospective gates; no scientific execution."""

import csv
import gzip
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    return dict(
        n=len(values),
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values) if len(values) > 1 else None,
        min=min(values),
        max=max(values),
    )


def table(name, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (ROOT / name).write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))


def run():
    result, audit, protocol = [
        read(ROOT / name) for name in ("result.json", "audit.json", "protocol.json")
    ]
    assert result["status"] == "COMPLETE"
    rows, updates = [], []
    for case in result["cases"]:
        peers = {
            c["policy"]: c
            for c in result["cases"]
            if c["fixture"]["label"] == case["fixture"]["label"]
        }
        base = peers["bf16_default_native"]
        native = peers[case["policy"].replace("chunks", "native")]
        verified = next(r for r in audit["rows"] if r["label"] == case["label"])
        pair = next(
            (r for r in audit["paired_initial_errors"] if r["label"] == case["label"]), None
        )
        row = dict(
            label=case["label"],
            fixture=case["fixture"]["label"],
            scope=case["fixture"]["scope"],
            policy=case["policy"],
            matched_reference=native["policy"],
            total_parameters=case["parameter_count"],
            ffn_parameters=case["ffn_parameters"],
            peak_job_mib=case["peak_job_allocated_bytes"] / 2**20,
            peak_reserved_mib=case["peak_job_reserved_bytes"] / 2**20,
            peak_training_mib=case["peak_training_allocated_bytes"] / 2**20,
            peak_phase="|".join(
                p["phase"]
                for p in case["memory_phases"]
                if p["peak_allocated_bytes"] == case["peak_job_allocated_bytes"]
            ),
            wall_update_ms=case["timing"]["wall_update_ms"]["median"],
            event_update_ms=case["timing"]["event_update_ms"]["median"],
            event_forward_ms=case["timing"]["event_forward_ms"]["median"],
            event_backward_ms=case["timing"]["event_backward_ms"]["median"],
            event_optimizer_ms=case["timing"]["event_optimizer_ms"]["median"],
            timing_stability=case["timing_stability_ratio"],
            reference_stability=base["timing_stability_ratio"],
            native_stability=native["timing_stability_ratio"],
            final_nll=case["final_validation"]["nll"],
            memory_ratio_bf16=case["peak_job_allocated_bytes"] / base["peak_job_allocated_bytes"],
            memory_ratio_native=case["peak_job_allocated_bytes"]
            / native["peak_job_allocated_bytes"],
            wall_ratio_bf16=case["timing"]["wall_update_ms"]["median"]
            / base["timing"]["wall_update_ms"]["median"],
            wall_ratio_native=case["timing"]["wall_update_ms"]["median"]
            / native["timing"]["wall_update_ms"]["median"],
            nll_ratio_bf16=case["final_validation"]["nll"] / base["final_validation"]["nll"],
            nll_ratio_native=case["final_validation"]["nll"] / native["final_validation"]["nll"],
            paired_loss_error=None if pair is None else pair["loss_relative_error"],
            paired_global_gradient_error=None
            if pair is None
            else pair["error"]["global_relative_l2"],
            paired_max_tensor_error=None
            if pair is None
            else pair["error"]["max_tensor_relative_l2"],
            audited_gradient_replay=None
            if verified["replay"] is None
            else verified["replay"]["passed"],
            gradient_replay_global=None
            if verified["replay"] is None
            else verified["replay"]["error"]["global_relative_l2"],
            gradient_replay_bitwise=None
            if verified["replay"] is None
            else verified["replay"]["bitwise_equal"],
            native_score_relative_error=verified["final_score_relative_error"],
            all_finite=case["weights_gradients_moments_finite"],
            clip_fraction=case["clip_fraction"],
            loop_wall_seconds=case["training_loop_wall_seconds"],
        )
        rows.append(row)
        for r in case["records"]:
            updates.append(
                dict(
                    label=case["label"],
                    policy=case["policy"],
                    **{
                        k: r[k]
                        for k in (
                            "step",
                            "absolute_step",
                            "warmup",
                            "loss",
                            "preclip_norm",
                            "wall_update_ms",
                            "event_update_ms",
                            "event_forward_ms",
                            "event_backward_ms",
                            "event_optimizer_ms",
                            "tokens_hash",
                            "targets_hash",
                            "peak_allocated_bytes",
                            "peak_reserved_bytes",
                        )
                    },
                )
            )
    metric_keys = (
        "peak_job_mib",
        "peak_reserved_mib",
        "peak_training_mib",
        "wall_update_ms",
        "event_update_ms",
        "memory_ratio_bf16",
        "memory_ratio_native",
        "wall_ratio_bf16",
        "wall_ratio_native",
        "nll_ratio_bf16",
        "nll_ratio_native",
        "final_nll",
        "clip_fraction",
    )
    groups, decisions = [], []
    for scope in dict.fromkeys(r["scope"] for r in rows):
        for policy in protocol["policies"]:
            peers = [r for r in rows if r["scope"] == scope and r["policy"] == policy]
            groups.append(
                dict(
                    scope=scope,
                    policy=policy,
                    fixtures=len(peers),
                    metrics={k: describe([p[k] for p in peers]) for k in metric_keys},
                )
            )
            if not policy.endswith("chunks"):
                continue
            native_peers = [
                r
                for r in rows
                if r["scope"] == scope and r["policy"] == policy.replace("chunks", "native")
            ]
            gates = dict(
                finite=all(p["all_finite"] for p in peers),
                paired_loss=all(p["paired_loss_error"] <= 1e-6 for p in peers),
                paired_global_gradient=all(
                    p["paired_global_gradient_error"] <= 0.002 for p in peers
                ),
                paired_tensor_gradient=all(p["paired_max_tensor_error"] <= 0.02 for p in peers),
                memory_bf16=all(p["memory_ratio_bf16"] <= 0.85 for p in peers),
                memory_native=all(p["memory_ratio_native"] <= 0.85 for p in peers),
                wall_bf16=st.median(p["wall_ratio_bf16"] for p in peers) <= 1.25,
                wall_native=st.median(p["wall_ratio_native"] for p in peers) <= 1.25,
                timing_stability=all(
                    max(p["timing_stability"], p["reference_stability"], p["native_stability"])
                    <= 1.25
                    for p in peers
                ),
                short_quality_bf16=all(p["nll_ratio_bf16"] <= 1.01 for p in peers),
                short_quality_native=all(p["nll_ratio_native"] <= 1.01 for p in peers),
                gradient_replay=all(p["audited_gradient_replay"] for p in peers + native_peers),
                independent_audit=audit["passed"],
            )
            decisions.append(
                dict(
                    scope=scope,
                    policy=policy,
                    gates=gates,
                    qualifies_broader_replication=all(gates.values()),
                    fresh_training_allocated=False,
                    decision="PROMISING"
                    if all(gates.values())
                    else "ELIMINATED from this resource gate",
                )
            )
    pareto = []
    for fixture in protocol["fixtures"]:
        peers = [r for r in rows if r["fixture"] == fixture["label"]]
        keys = ("peak_job_mib", "wall_update_ms", "final_nll")
        nondominated = [
            a["policy"]
            for a in peers
            if not any(
                all(b[k] <= a[k] for k in keys) and any(b[k] < a[k] for k in keys) for b in peers
            )
        ]
        pareto.append(dict(fixture=fixture["label"], raw_nondominated_policies=nondominated))
    summary = dict(
        status="COMPLETE",
        cases=30,
        optimizer_updates=result["optimizer_updates"],
        training_targets=result["training_targets"],
        profile_backward_passes=result["profile_backward_passes"],
        qualification_backward_passes=24,
        audit_backward_passes=24,
        elapsed_seconds=result["elapsed_seconds"],
        recorded_memory_phases=sum(len(c["memory_phases"]) for c in result["cases"]),
        groups=groups,
        decisions=decisions,
        pareto_by_fixture=pareto,
        pareto_note="Raw 50-update NLL is not a statistically established quality difference.",
        qualified_pairs=[
            dict(scope=d["scope"], policy=d["policy"])
            for d in decisions
            if d["qualifies_broader_replication"]
        ],
        all_audits_passed=audit["passed"],
        native_validation_scores=36,
        bitwise_gradient_replays=audit["bitwise_gradient_replays"],
        maximum_gradient_replay_global=max(
            r["replay"]["error"]["global_relative_l2"]
            for r in audit["rows"]
            if r["replay"] is not None
        ),
        maximum_native_score_relative_error=max(
            [r["final_score_relative_error"] for r in audit["rows"]]
            + [r["maximum_relative_error"] for r in audit["initial_native_scores"]]
        ),
        all_phase_peaks_recorded=result["all_phase_peaks_recorded"],
        fresh_training_runs=0,
        broad_goal_achieved=False,
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    table("metrics.csv.gz", rows)
    table("updates.csv.gz", updates)
    print(
        json.dumps(dict(qualified_pairs=summary["qualified_pairs"], decisions=decisions), indent=2)
    )


if __name__ == "__main__":
    run()
