"""Verify H114's prospective sources, complete ledger, audit and public exports."""

import csv
import gzip
import hashlib
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/fp32_classifier_profile_v1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    return dict(
        n=len(values),
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values) if len(values) > 1 else None,
        min=min(values),
        max=max(values),
    )


def csv_rows(path):
    return list(csv.DictReader(io.StringIO(gzip.decompress(Path(path).read_bytes()).decode())))


def run():
    protocol = read(ROOT / "protocol.json")
    for field in ("sources", "checkpoint_hashes", "extra_source_provenance"):
        for path, digest in protocol[field].items():
            assert sha(path) == digest, path
    prior = Path("results/verification/classifier_precision_final_v1.json")
    assert sha(prior) == protocol["previous_receipt_sha256"]
    for path, digest in read(prior)["files"].items():
        actual = protocol["before_documents"].get(path, {}).get("preserved_path", path)
        assert sha(actual) == digest, path
    for record in protocol["before_documents"].values():
        assert sha(record["preserved_path"]) == record["sha256"]
    for dataset in protocol["datasets"].values():
        directory = Path(dataset["path"])
        assert sha(directory / "manifest.json") == dataset["manifest_sha256"]
        for path, digest in dataset["manifest"]["files"].items():
            assert sha(directory / path) == digest
    for name in (
        "audit_protocol.json",
        "audit_recovery_protocol.json",
        "completion_audit_protocol.json",
    ):
        for path, digest in read(ROOT / name)["files"].items():
            assert sha(path) == digest, path
    result, audit, summary, qualification = [
        read(ROOT / name)
        for name in ("result.json", "completion_audit.json", "summary.json", "qualification.json")
    ]
    assert result["status"] == summary["status"] == "COMPLETE" and audit["passed"]
    assert len(result["cases"]) == 56 and len(qualification["double_model_checks"]) == 16
    assert qualification["passed"] and qualification["optimizer_updates"] == 0
    assert result["optimizer_updates"] == summary["optimizer_updates"] == 2800
    assert result["profile_backward_passes"] == summary["profile_backward_passes"] == 2856
    assert result["training_targets"] == summary["training_targets"] == 10649600
    assert result["all_phase_peaks_recorded"] and summary["all_phase_peaks_recorded"]
    assert audit["saved_gradient_hashes_verified"] == 56 and audit["native_validation_scores"] == 70
    assert not audit["full_gradient_replay_qualified"] and audit["candidate_advancement_held"]
    assert audit["training_batches_verified"] == 2800 and audit["optimizer_updates"] == 0
    assert len(result["boundaries"]) == 57 and len(audit["boundaries"]) == 15
    assert all(
        b == dict(allocated=0, reserved=0) for b in result["boundaries"] + audit["boundaries"]
    )
    tensors = {}
    for case in result["cases"]:
        assert len(case["records"]) == 50 and len(case["memory_phases"]) == 107
        assert case["optimizer_updates"] == 50 and case["profile_backward_passes"] == 51
        assert case["weights_gradients_moments_finite"]
        assert case["parameter_bytes"] == case["gradient_bytes"] == 4 * case["parameter_count"]
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            assert case[key.replace("peak_", "peak_job_")] == max(
                p[key] for p in case["memory_phases"]
            )
        for index, row in enumerate(case["records"], 1):
            assert row["step"] == index and row["warmup"] == (index <= 20)
            assert row["absolute_step"] == case["fixture"]["source_step"] + index
            assert row["phase"] == f"update_{index}"
            phase = next(p for p in case["memory_phases"] if p["phase"] == row["phase"])
            assert all(row[k] == v for k, v in phase.items())
            assert all(
                math.isfinite(row[k]) and row[k] >= 0
                for k in (
                    "wall_update_ms",
                    "event_update_ms",
                    "event_forward_ms",
                    "event_backward_ms",
                    "event_optimizer_ms",
                    "loss",
                    "preclip_norm",
                )
            )
        for key, stored in case["timing"].items():
            recomputed = describe([r[key] for r in case["records"][20:]])
            assert stored == recomputed, (case["label"], key)
        means = [
            st.mean(r["wall_update_ms"] for r in case["records"][start : start + 10])
            for start in (20, 30, 40)
        ]
        assert means == case["timing_block_means"]
        assert max(means) / min(means) == case["timing_stability_ratio"]
        for artifact in (case["checkpoint"], case["initial_probe"]):
            assert sha(artifact["path"]) == artifact["sha256"]
            tensors[artifact["path"]] = artifact["sha256"]
    assert len(tensors) == 112
    rows, updates = csv_rows(ROOT / "metrics.csv.gz"), csv_rows(ROOT / "updates.csv.gz")
    assert len(rows) == 56 and len(updates) == 2800
    for case in result["cases"]:
        selected = [r for r in updates if r["label"] == case["label"]]
        assert len(selected) == 50
        for raw, packed in zip(case["records"], selected, strict=True):
            for key in (
                "wall_update_ms",
                "event_update_ms",
                "loss",
                "preclip_norm",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ):
                assert float(packed[key]) == raw[key]
            assert (
                packed["tokens_hash"] == raw["tokens_hash"]
                and packed["targets_hash"] == raw["targets_hash"]
            )
    for fixture in protocol["fixtures"]:
        peers = [c for c in result["cases"] if c["fixture"]["label"] == fixture["label"]]
        assert len(peers) == 4
        assert [p["policy"] for p in peers] == fixture["policy_order"]
        for key in (
            "initial_model_hash",
            "initial_optimizer_hash",
            "initial_sampler_hash",
            "data_order_hash",
            "final_sampler_hash",
        ):
            assert len({p[key] for p in peers}) == 1, (fixture["label"], key)
    for group in summary["groups"]:
        peers = [r for r in rows if r["scope"] == group["scope"] and r["policy"] == group["policy"]]
        for key, stats in group["metrics"].items():
            assert stats == describe([float(p[key]) for p in peers])
    for decision in summary["decisions"]:
        peers = [
            r for r in rows if r["scope"] == decision["scope"] and r["policy"] == "chunks_fp32"
        ]
        gates = dict(
            all_finite=all(p["all_finite"] == "True" for p in peers),
            initial_loss_fidelity=all(
                float(p["initial_loss_relative_error"]) <= 1e-6 for p in peers
            ),
            initial_global_gradient_fidelity=all(
                float(p["global_gradient_relative_error"]) <= 0.002 for p in peers
            ),
            initial_per_tensor_gradient_fidelity=all(
                float(p["max_tensor_gradient_relative_error"]) <= 0.02 for p in peers
            ),
            all_memory_ratios_at_most_0_85=all(float(p["memory_ratio"]) <= 0.85 for p in peers),
            median_wall_ratio_at_most_1_25=st.median(float(p["wall_ratio"]) for p in peers) <= 1.25,
            all_reference_and_candidate_timing_stable=all(
                max(
                    float(p["timing_stability_ratio"]), float(p["reference_timing_stability_ratio"])
                )
                <= 1.25
                for p in peers
            ),
            every_native_nll_ratio_at_most_1_01=all(
                float(p["final_nll_ratio"]) <= 1.01 for p in peers
            ),
        )
        assert gates == decision["gates"]
        assert all(gates.values()) == decision["passes_original_measured_gates"]
        assert (
            not decision["qualifies_fresh_lm_experiment"] and decision["gradient_replay_audit_hold"]
        )
    assert summary["qualified_scopes"] == []
    for mode in (
        "study",
        "completion_audit",
        "gradient_replay_diagnosis",
        "analyze_recovery",
        "plot_recovery",
    ):
        assert (ROOT / f"{mode}_exit.txt").read_text().strip() == "0"
    assert not (ROOT / "failure.json").exists()
    for mode in ("audit", "audit_recovery"):
        assert (ROOT / f"{mode}_exit.txt").read_text().strip() == "1"
        assert "AssertionError" in read(ROOT / f"{mode}_failure.json")["traceback"]
    assert (ROOT / "analyze_exit.txt").read_text().strip() == "4294967295"
    assert (ROOT / "plot_exit.txt").read_text().strip() == "1"
    assert "FileNotFoundError" in (ROOT / "plot.log").read_text()
    postprocessing = read(ROOT / "postprocessing_record.json")
    assert not postprocessing["scientific_training_repeated"]
    for path, digest in postprocessing["source_hashes"].items():
        assert sha(path) == digest, path
    packed_files = {}
    for path in (ROOT / "result.json", ROOT / "completion_audit.json", *ROOT.glob("*.log")):
        target = path.with_name(path.name + ".gz")
        target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        assert gzip.decompress(target.read_bytes()) == path.read_bytes()
        packed_files[target.as_posix()] = dict(
            sha256=sha(target), original_sha256=sha(path), bytes=target.stat().st_size
        )
    paths = [
        Path(__file__),
        *ROOT.glob("source/*.py"),
        ROOT / "source/README.md",
        ROOT / "protocol.json",
        ROOT / "audit_protocol.json",
        ROOT / "audit_recovery_protocol.json",
        ROOT / "completion_audit_protocol.json",
        ROOT / "audit_failure.json",
        ROOT / "audit_recovery_failure.json",
        ROOT / "gradient_replay_diagnosis.json",
        ROOT / "postprocessing_record.json",
        ROOT / "qualification.json",
        ROOT / "summary.json",
        ROOT / "environment.json",
        ROOT / "metrics.csv.gz",
        ROOT / "updates.csv.gz",
        Path("research/fp32_classifier_profile_plan.md"),
        Path("research/fp32_classifier_profile_audit_recovery.md"),
        Path("research/fp32_classifier_profile_completion_audit.md"),
        Path("research/fp32_classifier_profile_results.md"),
        Path("research/figures/fp32_classifier_profile.png"),
        *[Path(n) for n in protocol["before_documents"]],
    ]
    receipt = dict(
        status="PASS",
        verification_scope="evidence integrity, native scores and stream/state verification; candidate qualification failed",
        candidate_qualification="FAILED",
        study="H114",
        frozen_sources=104,
        maintained_files_unchanged=61,
        source_checkpoints=14,
        saved_tensor_artifacts=112,
        double_model_checks=16,
        continuation_cases=56,
        optimizer_updates=2800,
        training_targets=10649600,
        profile_backward_passes=2856,
        completion_audit_backward_passes=0,
        diagnosis_backward_passes=6,
        failed_recovery_backward_passes=6,
        failed_original_backward_count_range=[1, 4],
        native_validation_scores=70,
        saved_gradient_hashes_verified=56,
        full_gradient_replay_qualified=False,
        candidate_advancement_held=True,
        preserved_failed_audits=2,
        training_batches_verified=2800,
        recorded_memory_phases=56 * 107,
        all_phase_peaks_recorded=True,
        qualified_scopes=summary["qualified_scopes"],
        fresh_training_runs=0,
        previous_maintained_tests_passed=116,
        maintained_tests_rerun=False,
        broad_goal_achieved=False,
        tensor_hashes=tensors,
        packed=packed_files,
        files={p.as_posix(): sha(p) for p in paths},
    )
    Path("results/verification/fp32_classifier_profile_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("files", "packed", "tensor_hashes")},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
