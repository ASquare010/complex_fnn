"""Publish all fixed outcomes, including failed numerical gates and recovery."""

import csv
import gzip
import io
import json
import statistics as st

from results.adam_update_sensitivity_v1.source.prepare import ROOT, read


def stats(values):
    return dict(
        n=len(values),
        min=min(values),
        median=st.median(values),
        mean=st.mean(values),
        max=max(values),
        sample_variance=st.variance(values) if len(values) > 1 else None,
    )


def run():
    p, r, a, audit = [
        read(ROOT / n) for n in ("protocol.json", "result.json", "analysis.json", "audit.json")
    ]
    assert (ROOT / "audit_exit.txt").read_text().strip() == "0"
    cross = [x for x in a["pairs"] if x["epsilon"] == 1e-8 and not x["same_policy"]]
    summary = dict(
        status="COMPLETE",
        strong_near_zero_amplification=a["strong_near_zero_amplification"],
        all_numerical_gates_passed=audit["passed"],
        cuda_numerical_gates_passed=all(
            x["passed"] for x in audit["rows"] if x["label"].startswith("cuda__")
        ),
        cpu_numerical_gates_passed=all(
            x["passed"] for x in audit["rows"] if x["label"].startswith("cpu__")
        ),
        cross_policy={
            k: stats([x[k] for x in cross])
            for k in (
                "clipped_distance",
                "direction_distance",
                "amplification",
                "near_zero_error_share",
                "opposite_nonzero_signs",
            )
        },
        conditioning=a["conditioning"],
        numerical_errors={
            device: {
                "clip_max": max(
                    x["clipped_error"]["distance"]
                    for x in audit["rows"]
                    if x["label"].startswith(device)
                ),
                "parameter_relative_max": max(
                    x["parameter_error"]["distance"]
                    for x in audit["rows"]
                    if x["label"].startswith(device)
                ),
                "parameter_absolute_max": max(
                    x["parameter_error"]["max_absolute"]
                    for x in audit["rows"]
                    if x["label"].startswith(device)
                ),
            }
            for device in ("cpu", "cuda")
        },
        new_disposable_optimizer_steps=12,
        original_stage_steps=6,
        recovery_steps=6,
        language_training_updates=0,
        forwards=0,
        backwards=0,
        validation_scores=0,
        distinct_training_seeds=1,
        selected_failure_diagnosis=True,
        frozen_sources=len(p["sources"]),
        maintained_files=len(p["maintained_files"]),
        tensor_artifacts=24,
        cuda_memory_intervals=24,
        cuda_zero_boundaries=7,
        original_stage_failed=True,
        original_stage_wall_seconds=None,
        recovery_wall_seconds=r["recovery_wall_seconds"],
        original_gate_requalified=False,
        changed_recipe_promoted=False,
        broad_goal_achieved=False,
    )
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    rows = []
    for pair in a["pairs"]:
        row = {k: v for k, v in pair.items() if not isinstance(v, list)}
        for key in ("bin_counts", "bin_squared_errors", "bin_error_shares"):
            row.update({f"{key}_{i}": v for i, v in enumerate(pair[key])})
        rows.append(row)
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (ROOT / "direction_pairs.csv.gz").write_bytes(gzip.compress(out.getvalue().encode(), mtime=0))
    for name in ("result.json", "analysis.json", "audit.json", "study_failure.json"):
        (ROOT / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
    for path in ROOT.glob("*.log"):
        if path.name != "summarize.log":
            path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
