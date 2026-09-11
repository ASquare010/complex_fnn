"""Verify H119 evidence integrity without hiding its failed numerical gate."""

import csv
import gzip
import io
import itertools
import json
import math
from pathlib import Path

from results.adam_update_sensitivity_v1.source.prepare import ROOT, hashes, read, sha
from results.adam_update_sensitivity_v1.source.summarize import stats


def run():
    p, recovery, result, a, audit, summary = [
        read(ROOT / n)
        for n in (
            "protocol.json",
            "recovery_protocol.json",
            "result.json",
            "analysis.json",
            "audit.json",
            "summary.json",
        )
    ]
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(p[field])
    assert len(p["sources"]) == 145 and len(p["maintained_files"]) == 61
    hashes(recovery["files"])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    prior = read(p["previous_receipt"])
    assert prior["status"] == "PASS" and not prior["broad_goal_achieved"]
    for name, digest in prior["files"].items():
        preserved = p["before_documents"].get(name, {}).get("preserved_path", name)
        assert sha(preserved) == digest, name
    for key in ("tensor_hashes", "local_metric_hashes"):
        hashes(prior[key])
    for name, info in prior["packed"].items():
        assert sha(name) == info["sha256"]
    for info in p["before_documents"].values():
        assert sha(info["preserved_path"]) == info["sha256"]
    assert result["cases"][:6] == recovery["cpu_cases"]
    assert len(result["cases"]) == 12 and len(audit["rows"]) == 12
    assert result["completed_original_steps"] == result["recovery_steps"] == 6
    assert result["disposable_optimizer_steps"] == summary["new_disposable_optimizer_steps"] == 12
    assert recovery["total_disposable_steps"] == 12
    assert result["original_stage_wall_seconds"] is None
    assert result["original_failure_preserved"] and summary["original_stage_failed"]
    for key in (
        "language_training_updates",
        "forwards",
        "backwards",
        "training_targets",
        "validation_scores",
    ):
        assert result[key] == p[key] == 0
    for record in (p, result, a, audit, summary):
        assert not record["broad_goal_achieved"]
    assert not summary["original_gate_requalified"] and not summary["changed_recipe_promoted"]
    assert audit["optimizer_steps"] == audit["forwards"] == audit["backwards"] == 0
    assert not audit["cuda_initialized"]
    tensors, local_metrics = {}, {}
    all_gates = []
    for i, c in enumerate(result["cases"]):
        assert c["device"] == ("cpu" if i < 6 else "cuda")
        assert c["input_label"] == p["cases"][i % 6]["label"]
        assert c["initial_model_hash"] == p["initial_model_hash"]
        assert c["source_gradient_sha256"] == p["cases"][i % 6]["probe"]["sha256"]
        assert c["parameters"] == 9099648 and c["finite"] and c["state_steps_all_one"]
        assert c["disposable_optimizer_steps"] == 1
        assert c["forwards"] == c["backwards"] == c["language_training_updates"] == 0
        record = ROOT / "runs" / c["label"] / "result.json"
        assert read(record) == c
        local_metrics[record.as_posix()] = sha(record)
        for field in ("clipped", "updated"):
            tensors[c[field]["path"]] = c[field]["sha256"]
        row = next(row for row in audit["rows"] if row["label"] == c["label"])
        parameter_pass = (
            row["parameter_error"]["distance"] <= p["parameter_relative_tolerance"]
            and row["parameter_error"]["max_absolute"] <= p["parameter_absolute_tolerance"]
        )
        moment_pass = all(m["distance"] <= 1e-6 for m in row["moment_errors"])
        clip_pass = row["clipped_error"]["distance"] <= p["clip_relative_tolerance"]
        assert row["passed"] == (parameter_pass and moment_pass and clip_pass)
        assert parameter_pass and moment_pass
        assert clip_pass == (c["device"] == "cuda")
        all_gates.append(row["passed"])
        if c["device"] == "cpu":
            assert c["memory_phases"] is None
        else:
            assert len(c["memory_phases"]) == 4
            for m in c["memory_phases"]:
                assert m["peak_allocated_bytes"] >= m["live_allocated_bytes"] > 0
                assert (
                    m["peak_reserved_bytes"]
                    >= m["live_reserved_bytes"]
                    >= m["live_allocated_bytes"]
                )
    hashes(tensors)
    assert len(tensors) == 24
    assert audit["passed"] == summary["all_numerical_gates_passed"] == all(all_gates) is False
    assert summary["cuda_numerical_gates_passed"] and not summary["cpu_numerical_gates_passed"]
    assert len(result["boundaries"]) == 7
    assert all(b == dict(allocated=0, reserved=0) for b in result["boundaries"])
    labels = [c["label"] for c in p["cases"]]
    policies = {c["label"]: c["policy"] for c in p["cases"]}
    expected_pairs = set(itertools.combinations(labels, 2))
    assert len(a["pairs"]) == audit["direction_pairs_checked"] == 45
    for eps in p["epsilons"]:
        rows = [r for r in a["pairs"] if r["epsilon"] == eps]
        assert len(rows) == 15 and {(r["left"], r["right"]) for r in rows} == expected_pairs
        for row in rows:
            assert row["same_policy"] == (policies[row["left"]] == policies[row["right"]])
            assert sum(row["bin_counts"]) == 9099648
            assert math.isclose(sum(row["bin_error_shares"]), 1.0, abs_tol=1e-12)
            assert math.isclose(
                row["near_zero_error_share"], sum(row["bin_error_shares"][:2]), abs_tol=1e-12
            )
            assert row["amplification"] == row["direction_distance"] / row["clipped_distance"]
    baseline = [r for r in a["pairs"] if r["epsilon"] == 1e-8 and not r["same_policy"]]
    assert len(baseline) == 9
    strong = all(
        r["amplification"] >= p["amplification_min"]
        and r["near_zero_error_share"] >= p["near_zero_share_min"]
        for r in baseline
    )
    assert (
        strong
        == a["strong_near_zero_amplification"]
        == summary["strong_near_zero_amplification"]
        is True
    )
    for key, values in summary["cross_policy"].items():
        assert stats([r[key] for r in baseline]) == values
    assert summary["conditioning"] == a["conditioning"]
    for eps in p["epsilons"][1:]:
        rows = [r for r in a["pairs"] if r["epsilon"] == eps and not r["same_policy"]]
        ratios = [
            r["direction_distance"] / b["direction_distance"]
            for r, b in zip(rows, baseline, strict=True)
        ]
        distortions = [d["distance"] for d in a["distortions"] if d["epsilon"] == eps]
        assert len(distortions) == 6
        c = next(c for c in a["conditioning"] if c["epsilon"] == eps)
        assert c["max_error_ratio"] == max(ratios) and c["max_direction_distortion"] == max(
            distortions
        )
        assert (
            c["passed"]
            == (
                max(ratios) <= p["conditioning_ratio_max"]
                and max(distortions) <= p["distortion_max"]
            )
            is False
        )
    assert len(a["native_pairs"]) == audit["native_pairs_checked"] == 15
    assert len(a["device_pairs"]) == audit["device_pairs_checked"] == 6
    assert len(a["distortions"]) == audit["distortions_checked"] == 12
    table = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((ROOT / "direction_pairs.csv.gz").read_bytes()).decode())
        )
    )
    assert len(table) == 45
    for row, item in zip(table, a["pairs"], strict=True):
        for key, value in item.items():
            if isinstance(value, list):
                assert [float(row[f"{key}_{i}"]) for i in range(4)] == value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                assert float(row[key]) == value
            else:
                assert row[key] == str(value)
    stages = (
        "prepare",
        "prepare_recovery",
        "recover",
        "analyze",
        "prepare_audit_recovery",
        "audit",
        "summarize",
        "plot",
        "report",
        "publish",
    )
    assert (ROOT / "study_exit.txt").read_text().strip() == "1"
    assert all((ROOT / f"{s}_exit.txt").read_text().strip() == "0" for s in stages)
    packed = {}
    # Retain all ended stage logs, including report/plot/publication and the failure.
    for path in ROOT.glob("*.log"):
        output = path.with_suffix(".log.gz")
        if not output.exists():
            output.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    for path in ROOT.glob("*.gz"):
        raw = gzip.decompress(path.read_bytes())
        original = path.with_suffix("")
        if original.exists():
            assert raw == original.read_bytes()
        packed[path.as_posix()] = dict(sha256=sha(path), unpacked_bytes=len(raw))
    for name in ("result.json", "analysis.json", "audit.json"):
        path = ROOT / name
        local_metrics[path.as_posix()] = sha(path)
    files = [
        *ROOT.glob("source/*"),
        *ROOT.glob("*_exit.txt"),
        *[
            ROOT / n
            for n in (
                "protocol.json",
                "recovery_protocol.json",
                "audit_protocol.json",
                "summary.json",
                "environment.json",
                "study_failure.json",
            )
        ],
        Path(__file__),
        *[Path(n) for n in p["before_documents"]],
        *[
            Path("research") / n
            for n in (
                "adam_update_sensitivity_plan.md",
                "adam_update_sensitivity_recovery_plan.md",
                "adam_update_sensitivity_results.md",
                "figures/adam_update_sensitivity.png",
            )
        ],
    ]
    receipt = dict(
        status="PASS",
        meaning="Evidence integrity and faithful accounting; CPU clipping numerical gate remains FAIL",
        scientific_numerical_audit_passed=False,
        cuda_numerical_gates_passed=True,
        strong_near_zero_amplification=True,
        low_distortion_epsilon_remedies_passed=False,
        new_disposable_optimizer_steps=12,
        language_training_updates=0,
        forwards=0,
        backwards=0,
        original_failure_preserved=True,
        broad_goal_achieved=False,
        files={
            f.relative_to(Path.cwd()).as_posix() if f.is_absolute() else f.as_posix(): sha(f)
            for f in files
            if f.is_file()
        },
        tensor_hashes=tensors,
        local_metric_hashes=local_metrics,
        packed=packed,
    )
    target = Path("results/verification/adam_update_sensitivity_final_v1.json")
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            dict(
                status="PASS",
                numerical_audit="FAIL_CPU_CLIPPING",
                maintained_hashes=61,
                frozen_sources=145,
                tensor_artifacts=24,
                disposable_steps=12,
                language_training_updates=0,
                broad_goal_achieved=False,
            )
        )
    )


if __name__ == "__main__":
    run()
