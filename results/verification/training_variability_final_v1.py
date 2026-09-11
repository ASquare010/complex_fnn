"""Verify H118 accounting, prior evidence and fixed conditional interpretation."""

import csv
import gzip
import io
import itertools
import json
import math
import statistics as st
from pathlib import Path

from results.verification import fp32_training_replication_final_v1 as previous_checks

ROOT = Path("results/training_variability_v1")
read, sha, hashes = previous_checks.read, previous_checks.sha, previous_checks.hashes


def table(name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(gzip.decompress((ROOT / name).read_bytes()).decode())))


def run() -> None:
    protocol = read(ROOT / "protocol.json")
    for key in ("sources", "maintained_files", "input_hashes"):
        hashes(protocol[key])
    assert len(protocol["sources"]) == 137 and len(protocol["maintained_files"]) == 61
    prior_path = Path("results/verification/fp32_training_replication_final_v1.json")
    assert sha(prior_path) == protocol["previous_receipt_sha256"]
    prior = read(prior_path)
    for name, digest in prior["files"].items():
        preserved = protocol["before_documents"].get(name, {}).get("preserved_path", name)
        assert sha(preserved) == digest, name
    for key in ("tensor_hashes", "local_metric_hashes"):
        hashes(prior[key])
    for name, info in prior["packed"].items():
        assert sha(name) == info["sha256"]
    for info in protocol["before_documents"].values():
        assert sha(info["preserved_path"]) == info["sha256"]
    assert prior["status"] == "PASS" and not prior["two_corpus_component_qualified"]
    hashes(read(ROOT / "audit_protocol.json")["files"])
    result, audit, summary = [
        read(ROOT / name) for name in ("result.json", "audit.json", "summary.json")
    ]
    previous_root = Path(protocol["previous_root"])
    old_result, old_audit = [read(previous_root / name) for name in ("result.json", "audit.json")]
    originals = [c for c in old_result["cases"] if c["label"] in protocol["original_case_labels"]]
    cases = originals + result["cases"]
    assert len(originals) == 2 and len(result["cases"]) == 4 and len(cases) == 6
    assert len({c["label"] for c in cases}) == 6
    assert len({c["initial_model_hash"] for c in cases}) == 1
    for key in (
        "source_checkpoint_sha256",
        "initial_optimizer_hash",
        "initial_sampler_hash",
        "data_order_hash",
        "final_sampler_hash",
    ):
        assert len({c[key] for c in cases}) == 1, key
    assert all(c["fixture"]["seed"] == 101 and c["fixture"]["source_step"] == 0 for c in cases)
    assert len(result["boundaries"]) == 5 and len(audit["boundaries"]) == 6
    assert all(
        b == dict(allocated=0, reserved=0) for b in result["boundaries"] + audit["boundaries"]
    )
    assert result["output_root_restored"] and result["selected_failure_diagnosis"]
    assert (
        result["optimizer_updates"]
        == protocol["optimizer_updates"]
        == summary["new_optimizer_updates"]
        == 3200
    )
    assert (
        result["training_targets"]
        == protocol["training_targets"]
        == summary["new_training_targets"]
        == 13107200
    )
    assert (
        result["profile_backward_passes"] == 3204
        and result["new_qualification_backward_passes"] == 0
    )
    assert summary["new_backward_passes"] == 3208 and summary["original_optimizer_updates"] == 1600
    assert audit["native_validation_scores"] == 13 and audit["gradient_replay_backward_passes"] == 4
    assert (
        audit["trained_checkpoints_verified"] == 12 and audit["training_batches_verified"] == 3200
    )
    assert audit["regenerated_initializations"] == 1 and audit["optimizer_updates"] == 0
    assert all(not p["broad_goal_achieved"] for p in (protocol, result, audit, summary))
    assert not summary["original_gate_requalified"] and not summary["new_changed_recipe_allocation"]
    metrics, updates, curves = [
        table(name) for name in ("metrics.csv.gz", "updates.csv.gz", "curves.csv.gz")
    ]
    assert (len(metrics), len(updates), len(curves)) == (6, 4800, 24)
    tensors, local_metrics = {}, {}
    previous_bound_root = previous_checks.ROOT
    try:
        for case in cases:
            is_original = case["label"] in protocol["original_case_labels"]
            selected_audit = old_audit if is_original else audit
            checked = next(r for r in selected_audit["rows"] if r["label"] == case["label"])
            previous_checks.ROOT = previous_root if is_original else ROOT
            found_tensors, found_metrics = previous_checks.verify_case(
                case, checked, updates, curves
            )
            tensors.update(found_tensors)
            local_metrics.update(found_metrics)
            row = next(r for r in metrics if r["label"] == case["label"])
            assert row["policy"] == case["policy"] and int(row["seed"]) == 101
            assert int(row["repetition"]) == case["fixture"].get("repetition", 0)
            assert row["origin"] == ("H117" if is_original else "H118")
            assert float(row["final_nll"]) == case["final_validation"]["nll"]
            assert int(row["total_parameters"]) == case["parameter_count"] == 9099648
            assert int(row["ffn_parameters"]) == case["ffn_parameters"] == 2801664
            assert float(row["peak_job_mib"]) == case["peak_job_allocated_bytes"] / 2**20
            assert float(row["peak_reserved_mib"]) == case["peak_job_reserved_bytes"] / 2**20
            assert float(row["wall_update_ms"]) == case["timing"]["wall_update_ms"]["median"]
            assert float(row["timing_stability"]) == case["timing_stability_ratio"]
    finally:
        previous_checks.ROOT = previous_bound_root
    for name, digest in protocol["checkpoint_hashes"].items():
        tensors[name] = digest
    assert len(tensors) == 25 and len(local_metrics) == 6
    for pair in audit["paired_initial_errors"]:
        error = pair["error"]
        assert pair["passed"] == (
            pair["loss_relative_error"] <= 1e-6
            and error["global_relative_l2"] <= 0.002
            and error["max_tensor_relative_l2"] <= 0.02
        )
    for row in audit["rows"]:
        replay = row["replay"]
        assert replay["passed"] == (
            replay["error"]["global_relative_l2"] <= 1e-5
            and replay["error"]["max_tensor_relative_l2"] <= 1e-4
            and replay["loss_relative_error"] <= 1e-6
        )
    assert audit["passed"] == (
        audit["initial_native_score"]["passed"]
        and all(r["all_scores_passed"] and r["replay"]["passed"] for r in audit["rows"])
        and all(r["passed"] for r in audit["paired_initial_errors"])
    )
    native_scores = [
        c["final_validation"]["nll"] for c in cases if c["policy"] == "fp32_default_native"
    ]
    chunk_scores = [
        c["final_validation"]["nll"] for c in cases if c["policy"] == "fp32_default_chunks"
    ]
    for policy, values in zip(protocol["policies"], (native_scores, chunk_scores), strict=True):
        assert summary["groups"][policy] == dict(
            n=3,
            mean=st.mean(values),
            median=st.median(values),
            sample_variance=st.variance(values),
            min=min(values),
            max=max(values),
            range=max(values) - min(values),
        )
    overlap = max(min(native_scores), min(chunk_scores)) <= min(
        max(native_scores), max(chunk_scores)
    )
    repeated = min(chunk_scores) > 1.01 * max(native_scores)
    interpretation = (
        "UNQUALIFIED"
        if not audit["passed"]
        else "REPEATED MATERIAL DISADVANTAGE"
        if repeated
        else "OVERLAPPING REPEAT VARIATION"
        if overlap
        else "INCONCLUSIVE"
    )
    assert summary["classification"] == interpretation
    assert (
        summary["intervals_overlap"] == overlap
        and summary["repeated_material_disadvantage"] == repeated
    )
    width = max(max(v) - min(v) for v in (native_scores, chunk_scores))
    gap = st.median(chunk_scores) - st.median(native_scores)
    assert summary["maximum_within_policy_range"] == width and summary["median_gap"] == gap
    assert summary["gap_over_range"] == (None if width == 0 else gap / width)
    for pair in summary["paired"]:
        peers = {
            c["policy"]: c for c in cases if c["fixture"].get("repetition", 0) == pair["repetition"]
        }
        n, c = [peers[p] for p in protocol["policies"]]
        assert (
            pair["native_nll"] == n["final_validation"]["nll"]
            and pair["chunk_nll"] == c["final_validation"]["nll"]
        )
        assert (
            pair["ratio"] == pair["chunk_nll"] / pair["native_nll"]
            and pair["gap"] == pair["chunk_nll"] - pair["native_nll"]
        )
        assert pair["memory_ratio"] == c["peak_job_allocated_bytes"] / n["peak_job_allocated_bytes"]
        assert (
            pair["time_ratio"]
            == c["timing"]["wall_update_ms"]["median"] / n["timing"]["wall_update_ms"]["median"]
        )
    assert summary["within_range_covers_original_gap"] == (
        width >= abs(summary["paired"][0]["gap"])
    )
    assert summary["new_paired_ratios_above_1pct"] == [
        p["ratio"] > 1.01 for p in summary["paired"][1:]
    ]
    for row in summary["divergence"]:
        a, b = [next(c for c in cases if c["label"] == row[k]) for k in ("left", "right")]
        differences = [
            x["step"]
            for x, y in zip(a["records"], b["records"], strict=True)
            if x["loss"] != y["loss"]
        ]
        assert row["first_different_recorded_training_loss"] == (
            differences[0] if differences else None
        )
        assert row["different_training_loss_records"] == len(differences)
    pairs = set(itertools.combinations([c["label"] for c in cases], 2))
    assert len(audit["initial_gradient_pairs"]) == 15 and len(audit["checkpoint_pairs"]) == 45
    assert {(r["left"], r["right"]) for r in audit["initial_gradient_pairs"]} == pairs
    for step in (200, 400, 800):
        selected = [r for r in audit["checkpoint_pairs"] if r["step"] == step]
        assert {(r["left"], r["right"]) for r in selected} == pairs
        for row in selected:
            for field in ("model", "exp_avg", "exp_avg_sq"):
                assert math.isfinite(row[field]) and 0 <= row[field] <= 2 + 1e-12
            assert row["model_bitwise_equal"] == (row["model"] == 0)
    for row in audit["initial_gradient_pairs"]:
        assert math.isfinite(row["distance"]) and 0 <= row["distance"] <= 2 + 1e-12
        assert row["bitwise_equal"] == (row["distance"] == 0)
    assert summary["recorded_memory_phases"] == 6444
    for mode in ("prepare", "study", "audit", "analyze", "plot"):
        assert (ROOT / f"{mode}_exit.txt").read_text().strip() == "0"
    assert not (ROOT / "failure.json").exists() and not (ROOT / "audit_failure.json").exists()
    packed = {}
    for raw in [ROOT / "result.json", ROOT / "audit.json", *ROOT.glob("*.log")]:
        data = raw.read_bytes()
        target = raw.with_name(raw.name + ".gz")
        target.write_bytes(gzip.compress(data, mtime=0))
        assert gzip.decompress(target.read_bytes()) == data
        packed[target.as_posix()] = dict(
            sha256=sha(target), original_sha256=sha(raw), bytes=target.stat().st_size
        )
    paths = [
        Path(__file__),
        *ROOT.glob("source/*.py"),
        ROOT / "source/README.md",
        *[
            ROOT / n
            for n in ("protocol.json", "audit_protocol.json", "environment.json", "summary.json")
        ],
        *ROOT.glob("*.csv.gz"),
        *ROOT.glob("*_exit.txt"),
        Path("research/training_variability_plan.md"),
        Path("research/training_variability_results.md"),
        Path("research/figures/training_variability.png"),
        *[Path(n) for n in protocol["before_documents"]],
    ]
    receipt = dict(
        status="PASS",
        study="H118",
        classification=interpretation,
        verification_scope="prior evidence, unchanged common runner, native scoring, initial gradient replay, complete records and conditional arithmetic",
        frozen_sources=137,
        maintained_files_unchanged=61,
        new_optimizer_updates=3200,
        new_backward_passes=3208,
        new_training_targets=13107200,
        new_trained_checkpoints_verified=12,
        original_cases_retained=2,
        new_gradient_replays=4,
        native_scores=13,
        cpu_checkpoint_pair_diagnostics=45,
        cpu_initial_gradient_pair_diagnostics=15,
        distinct_training_seeds=1,
        selected_failure_diagnosis=True,
        original_gate_requalified=False,
        broad_goal_achieved=False,
        no_default_change=True,
        maintained_tests_rerun=False,
        previous_maintained_tests_passed=116,
        all_audits_passed=audit["passed"],
        tensor_hashes=tensors,
        local_metric_hashes=local_metrics,
        packed=packed,
        files={p.as_posix(): sha(p) for p in paths},
    )
    Path("results/verification/training_variability_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("files", "packed", "tensor_hashes", "local_metric_hashes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
