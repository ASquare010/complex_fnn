"""Verify H116 evidence, full-job accounting and prospective gates without GPU work."""

import csv
import gzip
import hashlib
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/fp32_decoder_resource_v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(values):
    for path, digest in values.items():
        assert sha(path) == digest, path


def describe(values):
    return dict(
        n=len(values),
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values) if len(values) > 1 else None,
        min=min(values),
        max=max(values),
    )


def csv_rows(name):
    return list(csv.DictReader(io.StringIO(gzip.decompress((ROOT / name).read_bytes()).decode())))


def run():
    protocol = read(ROOT / "protocol.json")
    for field in (
        "sources",
        "maintained_files",
        "checkpoint_hashes",
        "previous_scientific_records",
    ):
        hashes(protocol[field])
    assert len(protocol["sources"]) == 115 and len(protocol["maintained_files"]) == 61
    assert len(protocol["checkpoint_hashes"]) == len(protocol["fixtures"]) == 6
    prior_path = Path("results/verification/decoder_gradient_transport_final_v1.json")
    assert sha(prior_path) == protocol["previous_receipt_sha256"]
    prior = read(prior_path)
    for path, digest in prior["files"].items():
        actual = protocol["before_documents"].get(path, {}).get("preserved_path", path)
        assert sha(actual) == digest, path
    for field in ("tensor_hashes", "local_metric_hashes"):
        hashes(prior[field])
    for path, info in prior["packed"].items():
        assert sha(path) == info["sha256"]
    for info in protocol["before_documents"].values():
        assert sha(info["preserved_path"]) == info["sha256"]
    for dataset in protocol["datasets"].values():
        folder = Path(dataset["path"])
        assert sha(folder / "manifest.json") == dataset["manifest_sha256"]
        for name, digest in dataset["manifest"]["files"].items():
            assert sha(folder / name) == digest
    for name in ("audit_protocol.json", "analysis_recovery_protocol.json"):
        hashes(read(ROOT / name)["files"])
    result, audit, summary, qualification = [
        read(ROOT / n) for n in ("result.json", "audit.json", "summary.json", "qualification.json")
    ]
    assert result["status"] == summary["status"] == "COMPLETE" and audit["passed"]
    assert len(result["cases"]) == summary["cases"] == protocol["cases"] == 30
    assert (
        result["optimizer_updates"]
        == summary["optimizer_updates"]
        == protocol["optimizer_updates"]
        == 1500
    )
    assert result["training_targets"] == summary["training_targets"] == 5120000
    assert result["profile_backward_passes"] == summary["profile_backward_passes"] == 1530
    assert result["elapsed_seconds"] == summary["elapsed_seconds"] > 0
    assert result["all_phase_peaks_recorded"] and summary["all_phase_peaks_recorded"]
    assert all(not obj["broad_goal_achieved"] for obj in (result, audit, summary, protocol))
    assert all(obj["fresh_training_runs"] == 0 for obj in (result, audit, summary, protocol))
    assert qualification["passed"] and len(qualification["double_model_checks"]) == 20
    assert qualification["reference_backward_passes"] == 4
    assert (
        qualification["qualification_backward_passes"]
        == result["qualification_backward_passes"]
        == 24
    )
    assert qualification["optimizer_updates"] == 0
    assert {
        (r["variant"], tuple(r["shape"]), r["policy"]) for r in qualification["double_model_checks"]
    } == {
        (v, s, p)
        for v in ("gelu", "swiglu")
        for s in ((2, 7), (1, 11))
        for p in protocol["policies"]
    }
    assert all(r["passed"] for r in qualification["double_model_checks"])
    assert len(result["boundaries"]) == 31 and len(audit["boundaries"]) == 37
    assert all(
        b == dict(allocated=0, reserved=0) for b in result["boundaries"] + audit["boundaries"]
    )
    assert len(audit["rows"]) == 30 and len(audit["initial_native_scores"]) == 6
    assert audit["native_validation_scores"] == 36 and audit["training_batches_verified"] == 1500
    assert audit["saved_gradient_hashes_verified"] == 30 and audit["optimizer_updates"] == 0
    replays = [r["replay"] for r in audit["rows"] if r["replay"] is not None]
    assert len(replays) == audit["gradient_replay_backward_passes"] == 24
    assert audit["bitwise_gradient_replays"] == sum(r["bitwise_equal"] for r in replays) == 16
    for row in replays:
        error = row["error"]
        assert error["max_tensor_relative_l2"] == max(error["per_tensor_relative_l2"].values())
        assert row["passed"] == (
            error["global_relative_l2"] <= 1e-5
            and error["max_tensor_relative_l2"] <= 1e-4
            and row["loss_relative_error"] <= 1e-6
        )
    assert all(r["passed"] for r in replays) == audit["replay_gates_passed"]
    assert all(
        r["passed"] and r["maximum_relative_error"] <= 1e-6 for r in audit["initial_native_scores"]
    )
    assert summary["maximum_gradient_replay_global"] == max(
        r["error"]["global_relative_l2"] for r in replays
    )
    assert summary["maximum_native_score_relative_error"] == max(
        [r["final_score_relative_error"] for r in audit["rows"]]
        + [r["maximum_relative_error"] for r in audit["initial_native_scores"]]
    )
    rows, updates = csv_rows("metrics.csv.gz"), csv_rows("updates.csv.gz")
    assert len(rows) == 30 and len(updates) == 1500 and len({r["label"] for r in rows}) == 30
    tensors, local_metrics = {}, {}
    for case in result["cases"]:
        assert case["status"] == "COMPLETE" and len(case["records"]) == 50
        assert case["weights_gradients_moments_finite"] and all(
            r["finite"] for r in case["diagnostics"].values()
        )
        assert case["optimizer_updates"] == 50 and case["profile_backward_passes"] == 51
        assert case["parameter_bytes"] == case["gradient_bytes"] == 4 * case["parameter_count"]
        assert case["optimizer_bytes"] >= 8 * case["parameter_count"]
        assert case["training_config"]["precision"] == case["execution"]["decoder_precision"]
        assert (
            case["training_targets"] == 50 * case["fixture"]["batch"] * case["fixture"]["context"]
        )
        phases = case["memory_phases"]
        expected_names = [
            "construction_data_and_optimizer",
            "initial_full_validation",
            "initial_gradient_probe_and_serialization",
        ]
        expected_names += [s for i in range(1, 51) for s in (f"before_update_{i}", f"update_{i}")]
        expected_names += [
            "after_update_logging",
            "final_full_validation",
            "final_layer_and_finite_diagnostics",
            "final_checkpoint_and_accounting",
        ]
        assert [p["phase"] for p in phases] == expected_names and len(phases) == 107
        for kind in ("allocated", "reserved"):
            assert case[f"peak_job_{kind}_bytes"] == max(p[f"peak_{kind}_bytes"] for p in phases)
        assert case["peak_training_allocated_bytes"] == max(
            r["peak_allocated_bytes"] for r in case["records"]
        )
        exported = [r for r in updates if r["label"] == case["label"]]
        assert len(exported) == 50
        for index, (record, packed) in enumerate(zip(case["records"], exported, strict=True), 1):
            assert record["step"] == int(packed["step"]) == index
            assert (
                record["absolute_step"]
                == int(packed["absolute_step"])
                == case["fixture"]["source_step"] + index
            )
            assert record["warmup"] == (packed["warmup"] == "True") == (index <= 20)
            phase = next(p for p in phases if p["phase"] == record["phase"])
            assert all(record[k] == v for k, v in phase.items())
            for key in ("tokens_hash", "targets_hash"):
                assert packed[key] == record[key]
            for key in (
                "loss",
                "preclip_norm",
                "wall_update_ms",
                "event_update_ms",
                "event_forward_ms",
                "event_backward_ms",
                "event_optimizer_ms",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ):
                assert (
                    float(packed[key]) == record[key]
                    and math.isfinite(record[key])
                    and record[key] >= 0
                )
        for key, stats in case["timing"].items():
            assert stats == describe([r[key] for r in case["records"][20:]])
        block_means = [
            st.mean(r["wall_update_ms"] for r in case["records"][i : i + 10]) for i in (20, 30, 40)
        ]
        assert case["timing_block_means"] == block_means
        assert case["timing_stability_ratio"] == max(block_means) / min(block_means)
        for name in ("checkpoint", "initial_probe"):
            value = case[name]
            assert sha(value["path"]) == value["sha256"]
            tensors[value["path"]] = value["sha256"]
        metric_path = ROOT / "runs" / case["label"] / "metrics.json"
        assert read(metric_path) == case
        local_metrics[metric_path.as_posix()] = sha(metric_path)
        audited = next(a for a in audit["rows"] if a["label"] == case["label"])
        assert audited["gradient_hash_verified"] and audited["source_and_final_states_verified"]
        assert audited["training_batches_verified"] == 50 and audited["final_score_passed"]
        score = audited["final_native_score"]
        assert all(score[k] == case["final_validation"][k] for k in ("targets", "order_sha256"))
        assert (
            audited["final_score_relative_error"]
            == abs(score["nll"] / case["final_validation"]["nll"] - 1)
            <= 1e-6
        )
        assert (audited["replay"] is None) == case["policy"].startswith("bf16")
        peers = {
            c["policy"]: c
            for c in result["cases"]
            if c["fixture"]["label"] == case["fixture"]["label"]
        }
        reference = peers["bf16_default_native"]
        native = peers[case["policy"].replace("chunks", "native")]
        row = next(r for r in rows if r["label"] == case["label"])
        assert row["scope"] == case["fixture"]["scope"] and row["policy"] == case["policy"]
        assert row["matched_reference"] == native["policy"]
        assert float(row["peak_job_mib"]) == case["peak_job_allocated_bytes"] / 2**20
        assert float(row["peak_reserved_mib"]) == case["peak_job_reserved_bytes"] / 2**20
        assert float(row["peak_training_mib"]) == case["peak_training_allocated_bytes"] / 2**20
        for metric in (
            "wall_update_ms",
            "event_update_ms",
            "event_forward_ms",
            "event_backward_ms",
            "event_optimizer_ms",
        ):
            assert float(row[metric]) == case["timing"][metric]["median"]
        for kind, base in (("bf16", reference), ("native", native)):
            assert (
                float(row["memory_ratio_" + kind])
                == case["peak_job_allocated_bytes"] / base["peak_job_allocated_bytes"]
            )
            assert (
                float(row["wall_ratio_" + kind])
                == case["timing"]["wall_update_ms"]["median"]
                / base["timing"]["wall_update_ms"]["median"]
            )
            assert (
                float(row["nll_ratio_" + kind])
                == case["final_validation"]["nll"] / base["final_validation"]["nll"]
            )
        assert float(row["timing_stability"]) == case["timing_stability_ratio"]
        assert float(row["reference_stability"]) == reference["timing_stability_ratio"]
        assert float(row["native_stability"]) == native["timing_stability_ratio"]
        assert float(row["final_nll"]) == case["final_validation"]["nll"]
        assert (
            float(row["clip_fraction"])
            == case["clip_fraction"]
            == sum(r["preclip_norm"] > 1 for r in case["records"]) / 50
        )
        assert row["all_finite"] == str(case["weights_gradients_moments_finite"])
        if case["policy"].endswith("chunks"):
            pair = next(p for p in audit["paired_initial_errors"] if p["label"] == case["label"])
            assert pair["reference"] == native["label"]
            assert float(row["paired_loss_error"]) == pair["loss_relative_error"]
            assert float(row["paired_global_gradient_error"]) == pair["error"]["global_relative_l2"]
            assert float(row["paired_max_tensor_error"]) == pair["error"]["max_tensor_relative_l2"]
        if audited["replay"] is not None:
            assert row["audited_gradient_replay"] == str(audited["replay"]["passed"])
    assert len(tensors) == 60 and len(local_metrics) == 30
    for fixture in protocol["fixtures"]:
        peers = [c for c in result["cases"] if c["fixture"] == fixture]
        assert len(peers) == 5 and [p["policy"] for p in peers] == fixture["policy_order"]
        for key in (
            "initial_model_hash",
            "source_optimizer_hash",
            "initial_optimizer_hash",
            "initial_sampler_hash",
            "data_order_hash",
            "final_sampler_hash",
        ):
            assert len({p[key] for p in peers}) == 1
    for group in summary["groups"]:
        peers = [r for r in rows if r["scope"] == group["scope"] and r["policy"] == group["policy"]]
        assert group["fixtures"] == len(peers)
        for key, stats in group["metrics"].items():
            assert stats == describe([float(r[key]) for r in peers])
    qualified = []
    for decision in summary["decisions"]:
        peers = [
            r for r in rows if r["scope"] == decision["scope"] and r["policy"] == decision["policy"]
        ]
        natives = [
            r
            for r in rows
            if r["scope"] == decision["scope"]
            and r["policy"] == decision["policy"].replace("chunks", "native")
        ]
        gates = dict(
            finite=all(r["all_finite"] == "True" for r in peers),
            paired_loss=all(float(r["paired_loss_error"]) <= 1e-6 for r in peers),
            paired_global_gradient=all(
                float(r["paired_global_gradient_error"]) <= 0.002 for r in peers
            ),
            paired_tensor_gradient=all(float(r["paired_max_tensor_error"]) <= 0.02 for r in peers),
            memory_bf16=all(float(r["memory_ratio_bf16"]) <= 0.85 for r in peers),
            memory_native=all(float(r["memory_ratio_native"]) <= 0.85 for r in peers),
            wall_bf16=st.median(float(r["wall_ratio_bf16"]) for r in peers) <= 1.25,
            wall_native=st.median(float(r["wall_ratio_native"]) for r in peers) <= 1.25,
            timing_stability=all(
                max(
                    float(r[k])
                    for k in ("timing_stability", "reference_stability", "native_stability")
                )
                <= 1.25
                for r in peers
            ),
            short_quality_bf16=all(float(r["nll_ratio_bf16"]) <= 1.01 for r in peers),
            short_quality_native=all(float(r["nll_ratio_native"]) <= 1.01 for r in peers),
            gradient_replay=all(r["audited_gradient_replay"] == "True" for r in peers + natives),
            independent_audit=audit["passed"],
        )
        assert gates == decision["gates"]
        assert all(gates.values()) == decision["qualifies_broader_replication"]
        assert not decision["fresh_training_allocated"]
        if all(gates.values()):
            qualified.append(dict(scope=decision["scope"], policy=decision["policy"]))
    assert (
        qualified
        == summary["qualified_pairs"]
        == [
            dict(scope="narrow_wikitext2", policy="fp32_default_chunks"),
            dict(scope="narrow_tinystories", policy="fp32_default_chunks"),
        ]
    )
    for pareto in summary["pareto_by_fixture"]:
        peers = [r for r in rows if r["fixture"] == pareto["fixture"]]
        keys = ("peak_job_mib", "wall_update_ms", "final_nll")
        computed = [
            a["policy"]
            for a in peers
            if not any(
                all(float(b[k]) <= float(a[k]) for k in keys)
                and any(float(b[k]) < float(a[k]) for k in keys)
                for b in peers
            )
        ]
        assert computed == pareto["raw_nondominated_policies"]
    for stage in ("study", "audit", "analyze_recovery", "plot"):
        assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
    assert (ROOT / "analyze_exit.txt").read_text().strip() == "3221225477"
    assert "access violation" in (ROOT / "analyze.log").read_text()
    assert not (ROOT / "failure.json").exists() and not (ROOT / "audit_failure.json").exists()
    recovery = read(ROOT / "analysis_recovery_protocol.json")
    assert (
        not recovery["scientific_training_repeated"] and recovery["additional_backward_passes"] == 0
    )
    packed = {}
    for path in (ROOT / "result.json", ROOT / "audit.json", *ROOT.glob("*.log")):
        data = path.read_bytes()
        target = path.with_name(path.name + ".gz")
        target.write_bytes(gzip.compress(data, mtime=0))
        assert gzip.decompress(target.read_bytes()) == data
        packed[target.as_posix()] = dict(
            sha256=sha(target), original_sha256=sha(path), bytes=target.stat().st_size
        )
    paths = [
        Path(__file__),
        *ROOT.glob("source/*.py"),
        ROOT / "source/README.md",
        *[
            ROOT / n
            for n in (
                "protocol.json",
                "audit_protocol.json",
                "analysis_recovery_protocol.json",
                "qualification.json",
                "environment.json",
                "summary.json",
            )
        ],
        *ROOT.glob("*.csv.gz"),
        *ROOT.glob("*_exit.txt"),
        Path("research/fp32_decoder_resource_plan.md"),
        Path("research/fp32_decoder_resource_results.md"),
        Path("research/figures/fp32_decoder_resource.png"),
        *[Path(n) for n in protocol["before_documents"]],
    ]
    receipt = dict(
        status="PASS",
        study="H116",
        verification_scope="full-job resource screen, native scores, FP32 gradient replays and stream/state integrity",
        frozen_sources=115,
        maintained_files_unchanged=61,
        source_checkpoints=6,
        saved_tensor_artifacts=60,
        continuation_cases=30,
        optimizer_updates=1500,
        training_targets=5120000,
        profile_backward_passes=1530,
        qualification_backward_passes=24,
        independent_audit_backward_passes=24,
        total_backward_passes=1578,
        double_model_comparisons=20,
        native_double_references=4,
        native_validation_scores=36,
        fp32_gradient_replays=24,
        bitwise_gradient_replays=16,
        bf16_gradient_replay_claim=False,
        training_batches_verified=1500,
        recorded_memory_phases=3210,
        zero_allocator_boundaries=68,
        all_phase_peaks_recorded=True,
        qualified_pairs=qualified,
        fresh_training_runs=0,
        no_default_change=True,
        broad_goal_achieved=False,
        preserved_analysis_failures=1,
        scientific_training_repeated=False,
        previous_maintained_tests_passed=116,
        maintained_tests_rerun=False,
        study_lint_exception="E402 only: intentional native-library preloads before other imports; frozen source preserved",
        tensor_hashes=tensors,
        local_metric_hashes=local_metrics,
        packed=packed,
        files={p.as_posix(): sha(p) for p in paths},
    )
    Path("results/verification/fp32_decoder_resource_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("files", "tensor_hashes", "local_metric_hashes", "packed")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
