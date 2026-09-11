"""Apply the frozen conditional interpretation to every original and repeated run."""

import csv
import gzip
import io
import itertools
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/training_variability_v1")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def describe(values: list[float]) -> dict:
    return dict(
        n=len(values),
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values),
        min=min(values),
        max=max(values),
        range=max(values) - min(values),
    )


def table(name: str, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (ROOT / name).write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))


def collect(protocol: dict, result: dict) -> list[dict]:
    previous = read(Path(protocol["previous_root"]) / "result.json")
    return [
        c for c in previous["cases"] if c["label"] in protocol["original_case_labels"]
    ] + result["cases"]


def run() -> None:
    assert (ROOT / "study_exit.txt").read_text().strip() == "0"
    assert (ROOT / "audit_exit.txt").read_text().strip() == "0"
    protocol, result, audit = [
        read(ROOT / name) for name in ("protocol.json", "result.json", "audit.json")
    ]
    cases = collect(protocol, result)
    assert len(cases) == 6
    metrics, curves, updates = [], [], []
    for case in cases:
        repeat = case["fixture"].get("repetition", 0)
        common = dict(
            label=case["label"],
            policy=case["policy"],
            repetition=repeat,
            origin="H117" if repeat == 0 else "H118",
            seed=101,
        )
        metrics.append(
            dict(
                **common,
                final_nll=case["final_validation"]["nll"],
                total_parameters=case["parameter_count"],
                ffn_parameters=case["ffn_parameters"],
                peak_job_mib=case["peak_job_allocated_bytes"] / 2**20,
                peak_reserved_mib=case["peak_job_reserved_bytes"] / 2**20,
                wall_update_ms=case["timing"]["wall_update_ms"]["median"],
                timing_stability=case["timing_stability_ratio"],
                clip_fraction=case["clip_fraction"],
                finite=case["weights_gradients_moments_finite"],
            )
        )
        scores = (
            [(0, case["initial_validation"])]
            + [(r["step"], r["validation"]) for r in case["intermediate_checkpoints"]]
            + [(800, case["final_validation"])]
        )
        for step, score in scores:
            curves.append(
                dict(
                    **common,
                    corpus="wikitext2",
                    step=step,
                    nll=score["nll"],
                    targets=score["targets"],
                    order_sha256=score["order_sha256"],
                )
            )
        for record in case["records"]:
            updates.append(
                dict(
                    **common,
                    **{
                        k: record[k]
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
    groups = {
        p: describe([m["final_nll"] for m in metrics if m["policy"] == p])
        for p in protocol["policies"]
    }
    native, chunk = [groups[p] for p in protocol["policies"]]
    paired = []
    for repeat in (0, 1, 2):
        peers = {r["policy"]: r for r in metrics if r["repetition"] == repeat}
        n, c = [peers[p] for p in protocol["policies"]]
        paired.append(
            dict(
                repetition=repeat,
                native_nll=n["final_nll"],
                chunk_nll=c["final_nll"],
                ratio=c["final_nll"] / n["final_nll"],
                gap=c["final_nll"] - n["final_nll"],
                memory_ratio=c["peak_job_mib"] / n["peak_job_mib"],
                time_ratio=c["wall_update_ms"] / n["wall_update_ms"],
            )
        )
    overlap = max(native["min"], chunk["min"]) <= min(native["max"], chunk["max"])
    repeated_material = chunk["min"] > 1.01 * native["max"]
    difference, within_range = (
        chunk["median"] - native["median"],
        max(native["range"], chunk["range"]),
    )
    label = (
        "UNQUALIFIED"
        if not audit["passed"]
        else (
            "REPEATED MATERIAL DISADVANTAGE"
            if repeated_material
            else "OVERLAPPING REPEAT VARIATION"
            if overlap
            else "INCONCLUSIVE"
        )
    )
    divergence = []
    for policy in protocol["policies"]:
        peers = [c for c in cases if c["policy"] == policy]
        for left, right in itertools.combinations(peers, 2):
            differing = [
                a["step"]
                for a, b in zip(left["records"], right["records"], strict=True)
                if a["loss"] != b["loss"]
            ]
            divergence.append(
                dict(
                    policy=policy,
                    left=left["label"],
                    right=right["label"],
                    first_different_recorded_training_loss=None if not differing else differing[0],
                    different_training_loss_records=len(differing),
                    final_model_bitwise_equal=left["final_model_hash"] == right["final_model_hash"],
                )
            )
    summary = dict(
        status="COMPLETE",
        classification=label,
        all_audits_passed=audit["passed"],
        new_executions=4,
        original_executions=2,
        distinct_training_seeds=1,
        new_optimizer_updates=3200,
        original_optimizer_updates=1600,
        new_training_targets=13107200,
        new_backward_passes=3208,
        selection="WikiText seed101 selected after H117 quality failure; conditional diagnosis",
        groups=groups,
        paired=paired,
        intervals_overlap=overlap,
        repeated_material_disadvantage=repeated_material,
        median_gap=difference,
        maximum_within_policy_range=within_range,
        gap_over_range=None if within_range == 0 else difference / within_range,
        within_range_covers_original_gap=within_range >= abs(paired[0]["gap"]),
        new_paired_ratios_above_1pct=[p["ratio"] > 1.01 for p in paired[1:]],
        divergence=divergence,
        recorded_memory_phases=sum(len(c["memory_phases"]) for c in result["cases"]),
        elapsed_seconds=result["elapsed_seconds"],
        original_gate_requalified=False,
        broad_goal_achieved=False,
        new_changed_recipe_allocation=False,
    )
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    table("metrics.csv.gz", metrics)
    table("curves.csv.gz", curves)
    table("updates.csv.gz", updates)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "classification",
                    "groups",
                    "paired",
                    "median_gap",
                    "maximum_within_policy_range",
                    "within_range_covers_original_gap",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
