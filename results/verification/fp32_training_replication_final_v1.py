"""Verify H117's complete grid and unchanged scientific gates; no GPU work."""

import csv
import gzip
import hashlib
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/fp32_training_replication_recovery_v1")
OLD = Path("results/fp32_training_replication_v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(mapping):
    for path, digest in mapping.items():
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


def table(name):
    return list(csv.DictReader(io.StringIO(gzip.decompress((ROOT / name).read_bytes()).decode())))


def verify_case(case, audit_row, updates, curves):
    assert case["status"] == "COMPLETE" and len(case["records"]) == 800
    assert case["fixture"]["source_step"] == 0
    assert case["optimizer_updates"] == 800 and case["profile_backward_passes"] == 801
    assert case["training_targets"] == 3276800
    assert case["parameter_count"] == 9099648 and case["ffn_parameters"] == 2801664
    assert case["parameter_bytes"] == case["gradient_bytes"] == 4 * case["parameter_count"]
    assert case["optimizer_bytes"] >= 8 * case["parameter_count"]
    assert case["weights_gradients_moments_finite"]
    assert all(d["finite"] for d in case["diagnostics"].values())
    assert (
        case["training_config"]["steps"] == 800
        and case["training_config"]["learning_rate"] == 0.0006
    )
    assert case["execution"]["attention_backend"] == "default"
    assert case["execution"]["decoder_precision"] == case["training_config"]["precision"]
    expected = [
        "construction_data_and_optimizer",
        "initial_full_validation",
        "initial_gradient_probe_and_serialization",
    ]
    for step in range(1, 801):
        expected.extend([f"before_update_{step}", f"update_{step}"])
        if step in (200, 400):
            expected.extend([f"validation_{step}", f"checkpoint_{step}"])
    expected.extend(
        [
            "after_update_logging",
            "final_full_validation",
            "final_layer_and_finite_diagnostics",
            "final_checkpoint_and_accounting",
        ]
    )
    phases = case["memory_phases"]
    assert [p["phase"] for p in phases] == expected and len(phases) == 1611
    for kind in ("allocated", "reserved"):
        assert case[f"peak_job_{kind}_bytes"] == max(p[f"peak_{kind}_bytes"] for p in phases)
        assert all(0 <= p[f"live_{kind}_bytes"] <= p[f"peak_{kind}_bytes"] for p in phases)
    assert case["peak_training_allocated_bytes"] == max(
        r["peak_allocated_bytes"] for r in case["records"]
    )
    selected = [r for r in updates if r["label"] == case["label"]]
    assert len(selected) == 800
    for index, (raw, packed) in enumerate(zip(case["records"], selected, strict=True), 1):
        assert (
            raw["step"]
            == raw["absolute_step"]
            == int(packed["step"])
            == int(packed["absolute_step"])
            == index
        )
        assert raw["warmup"] == (packed["warmup"] == "True") == (index <= 20)
        assert ("gradient_statistics_after_clip" in raw) == (index in (1, 20, 200, 400, 800))
        phase = next(p for p in phases if p["phase"] == raw["phase"])
        assert all(raw[k] == v for k, v in phase.items())
        for k in ("tokens_hash", "targets_hash"):
            assert packed[k] == raw[k]
        for k in (
            "wall_update_ms",
            "event_update_ms",
            "event_forward_ms",
            "event_backward_ms",
            "event_optimizer_ms",
            "loss",
            "preclip_norm",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            assert float(packed[k]) == raw[k] and math.isfinite(raw[k]) and raw[k] >= 0
    for key, stats in case["timing"].items():
        assert stats == describe([r[key] for r in case["records"][20:]])
    blocks = [
        st.mean(r["wall_update_ms"] for r in case["records"][i : i + 260]) for i in (20, 280, 540)
    ]
    assert case["timing_block_means"] == blocks
    assert case["timing_stability_ratio"] == max(blocks) / min(blocks)
    assert case["clip_fraction"] == sum(r["preclip_norm"] > 1 for r in case["records"]) / 800
    assert audit_row["gradient_hash_verified"] and audit_row["source_and_final_states_verified"]
    assert (
        audit_row["trained_checkpoints_verified"] == 3
        and audit_row["training_batches_verified"] == 800
    )
    assert (audit_row["replay"] is None) == case["policy"].startswith("bf16")
    expected_scores = [(0, case["initial_validation"])]
    expected_scores += [(c["step"], c["validation"]) for c in case["intermediate_checkpoints"]]
    expected_scores += [(800, case["final_validation"])]
    assert [s for s, _ in expected_scores] == [0, 200, 400, 800]
    points = [r for r in curves if r["label"] == case["label"]]
    assert len(points) == 4
    for (step, score), point in zip(expected_scores, points, strict=True):
        assert int(point["step"]) == step and float(point["nll"]) == score["nll"]
        assert (
            int(point["targets"]) == score["targets"]
            and point["order_sha256"] == score["order_sha256"]
        )
        assert (
            int(point["seed"]) == case["fixture"]["seed"]
            and point["corpus"] == case["fixture"]["dataset"]
        )
    assert [r["step"] for r in audit_row["checkpoint_scores"]] == [200, 400, 800]
    for (_, expected_score), audited in zip(
        expected_scores[1:], audit_row["checkpoint_scores"], strict=True
    ):
        actual = audited["native_score"]
        assert (
            actual["targets"] == expected_score["targets"]
            and actual["order_sha256"] == expected_score["order_sha256"]
        )
        assert audited["relative_error"] == abs(actual["nll"] / expected_score["nll"] - 1)
        assert audited["passed"] == (audited["relative_error"] <= 1e-6)
        assert audited["state_and_sampler_verified"]
    assert audit_row["final_native_score"] == audit_row["checkpoint_scores"][-1]["native_score"]
    assert (
        audit_row["final_score_relative_error"]
        == audit_row["checkpoint_scores"][-1]["relative_error"]
    )
    artifacts = [case["initial_probe"], case["checkpoint"], *case["intermediate_checkpoints"]]
    hashes({a["path"]: a["sha256"] for a in artifacts})
    metric = ROOT / "runs" / case["label"] / "metrics.json"
    assert read(metric) == case
    history = [
        json.loads(line) for line in metric.with_name("history.jsonl").read_text().splitlines()
    ]
    assert history == case["records"]
    return {a["path"]: a["sha256"] for a in artifacts}, {metric.as_posix(): sha(metric)}


def run():
    protocol = read(ROOT / "protocol.json")
    for field in (
        "sources",
        "maintained_files",
        "previous_scientific_records",
        "original_evidence",
    ):
        hashes(protocol[field])
    assert len(protocol["sources"]) == 129 and len(protocol["maintained_files"]) == 61
    assert protocol["checkpoint_hashes"] == {} and protocol["seeds"] == [101, 113, 127]
    assert (
        not protocol["training_repeated"]
        and protocol["original_failed_attempt"]["completed_updates"] == 0
    )
    assert (OLD / "study_exit.txt").read_text().strip() == "1"
    assert "assert not torch.cuda.is_initialized()" in read(OLD / "failure.json")["traceback"]
    assert read(OLD / "qualification.json")["qualification_backward_passes"] == 24
    assert not list((OLD / "runs").iterdir()) and not (OLD / "initializations.json").exists()
    prior_path = Path("results/verification/fp32_decoder_resource_final_v1.json")
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
    hashes(read(ROOT / "audit_protocol.json")["files"])
    plot_recovery = read(ROOT / "plot_recovery_protocol.json")
    hashes(plot_recovery["files"])
    assert plot_recovery["prior_raw_exit"] == 3221225477
    assert plot_recovery["missing_figure_before_retry"]
    assert not plot_recovery["scientific_training_repeated"]
    assert not plot_recovery["scientific_audit_repeated"]
    assert plot_recovery["additional_backward_passes"] == 0
    assert plot_recovery["additional_optimizer_updates"] == 0
    assert (ROOT / "plot_exit.txt").read_text().strip() == "3221225477"
    assert "Windows fatal exception: access violation" in (ROOT / "plot.log").read_text()
    initials, result, audit, summary, qualification = [
        read(ROOT / n)
        for n in (
            "initializations.json",
            "result.json",
            "audit.json",
            "summary.json",
            "qualification.json",
        )
    ]
    assert (
        initials["initialized_on_cpu"]
        and not initials["cuda_initialized"]
        and initials["optimizer_updates"] == 0
    )
    assert len(initials["initializations"]) == 6
    tensors = {r["path"]: r["sha256"] for r in initials["initializations"]}
    hashes(tensors)
    assert result["status"] == summary["status"] == "COMPLETE"
    assert len(result["cases"]) == summary["cases"] == protocol["cases"] == 18
    for key, expected in (
        ("optimizer_updates", 14400),
        ("training_targets", 58982400),
        ("profile_backward_passes", 14418),
        ("fresh_training_runs", 18),
    ):
        assert result[key] == summary[key] == protocol[key] == expected
    assert all(not p["broad_goal_achieved"] for p in (result, summary, audit, protocol))
    assert qualification["passed"] and len(qualification["double_model_checks"]) == 20
    assert (
        qualification["reference_backward_passes"] == 4
        and qualification["qualification_backward_passes"] == 24
    )
    assert all(r["passed"] for r in qualification["double_model_checks"])
    assert len(result["boundaries"]) == 19 and len(audit["boundaries"]) == 25
    assert all(
        b == dict(allocated=0, reserved=0) for b in result["boundaries"] + audit["boundaries"]
    )
    assert audit["regenerated_initializations"] == 6 and audit["trained_checkpoints_verified"] == 54
    assert audit["native_validation_scores"] == 60 and audit["training_batches_verified"] == 14400
    assert audit["saved_gradient_hashes_verified"] == 18 and audit["optimizer_updates"] == 0
    replays = [r["replay"] for r in audit["rows"] if r["replay"] is not None]
    assert len(replays) == audit["gradient_replay_backward_passes"] == 12
    for row in replays:
        error = row["error"]
        assert error["max_tensor_relative_l2"] == max(error["per_tensor_relative_l2"].values())
        assert row["passed"] == (
            error["global_relative_l2"] <= 1e-5
            and error["max_tensor_relative_l2"] <= 1e-4
            and row["loss_relative_error"] <= 1e-6
        )
    assert audit["bitwise_gradient_replays"] == sum(r["bitwise_equal"] for r in replays)
    assert audit["replay_gates_passed"] == all(r["passed"] for r in replays)
    assert all(r["initialization_regenerated_exactly"] for r in audit["initial_native_scores"])
    assert audit["passed"] == (
        all(r["passed"] for r in replays + audit["initial_native_scores"])
        and all(r["all_scores_passed"] for r in audit["rows"])
    )
    rows, updates, curves = table("metrics.csv.gz"), table("updates.csv.gz"), table("curves.csv.gz")
    assert (len(rows), len(updates), len(curves)) == (18, 14400, 72)
    local_metrics = {}
    for case in result["cases"]:
        inspected = next(a for a in audit["rows"] if a["label"] == case["label"])
        artifacts, metrics = verify_case(case, inspected, updates, curves)
        tensors.update(artifacts)
        local_metrics.update(metrics)
    assert len(tensors) == 78 and len(local_metrics) == 18
    # Endpoint mapping, decision checks and publication are defined below.
    finish(protocol, result, audit, summary, rows, curves, tensors, local_metrics)


def finish(protocol, result, audit, summary, rows, curves, tensors, local_metrics):
    assert len({r["label"] for r in rows}) == 18 and len(summary["groups"]) == 6
    initials = read(ROOT / "initializations.json")["initializations"]
    for fixture in protocol["fixtures"]:
        peers = [c for c in result["cases"] if c["fixture"] == fixture]
        assert len(peers) == 3 and [c["policy"] for c in peers] == fixture["policy_order"]
        source = next(r for r in initials if r["fixture"] == fixture["label"])
        for c in peers:
            assert c["source_checkpoint_sha256"] == source["sha256"]
            assert c["initial_model_hash"] == source["model_hash"]
            assert c["source_optimizer_hash"] == source["optimizer_hash"]
        for key in (
            "initial_model_hash",
            "initial_optimizer_hash",
            "initial_sampler_hash",
            "data_order_hash",
            "final_sampler_hash",
        ):
            assert len({c[key] for c in peers}) == 1
        initial_audit = next(
            r for r in audit["initial_native_scores"] if r["fixture"] == fixture["label"]
        )
        errors = []
        for c in peers:
            actual, expected = initial_audit["score"], c["initial_validation"]
            assert all(actual[k] == expected[k] for k in ("targets", "order_sha256"))
            errors.append(abs(actual["nll"] / expected["nll"] - 1))
        assert initial_audit["maximum_relative_error"] == max(errors)
        assert initial_audit["passed"] == (max(errors) <= 1e-6)
    for case in result["cases"]:
        row = next(r for r in rows if r["label"] == case["label"])
        peers = {
            c["policy"]: c
            for c in result["cases"]
            if c["fixture"]["label"] == case["fixture"]["label"]
        }
        base, native = (
            peers["bf16_default_native"],
            peers[case["policy"].replace("chunks", "native")],
        )
        assert row["scope"] == case["fixture"]["scope"] and row["policy"] == case["policy"]
        assert int(row["seed"]) == case["fixture"]["seed"]
        assert row["matched_reference"] == native["policy"]
        for field, key in (
            ("peak_job_mib", "peak_job_allocated_bytes"),
            ("peak_reserved_mib", "peak_job_reserved_bytes"),
            ("peak_training_mib", "peak_training_allocated_bytes"),
        ):
            assert float(row[field]) == case[key] / 2**20
        for key in (
            "wall_update_ms",
            "event_update_ms",
            "event_forward_ms",
            "event_backward_ms",
            "event_optimizer_ms",
        ):
            assert float(row[key]) == case["timing"][key]["median"]
        for kind, reference in (("bf16", base), ("native", native)):
            assert (
                float(row["memory_ratio_" + kind])
                == case["peak_job_allocated_bytes"] / reference["peak_job_allocated_bytes"]
            )
            assert (
                float(row["wall_ratio_" + kind])
                == case["timing"]["wall_update_ms"]["median"]
                / reference["timing"]["wall_update_ms"]["median"]
            )
            assert (
                float(row["nll_ratio_" + kind])
                == case["final_validation"]["nll"] / reference["final_validation"]["nll"]
            )
        for field, reference in (
            ("timing_stability", case),
            ("reference_stability", base),
            ("native_stability", native),
        ):
            assert float(row[field]) == reference["timing_stability_ratio"]
        assert float(row["final_nll"]) == case["final_validation"]["nll"]
        assert float(row["clip_fraction"]) == case["clip_fraction"]
        assert row["all_finite"] == str(case["weights_gradients_moments_finite"])
        verified = next(r for r in audit["rows"] if r["label"] == case["label"])
        if verified["replay"] is not None:
            assert row["audited_gradient_replay"] == str(verified["replay"]["passed"])
            assert (
                float(row["gradient_replay_global"])
                == verified["replay"]["error"]["global_relative_l2"]
            )
        if case["policy"].endswith("chunks"):
            pair = next(p for p in audit["paired_initial_errors"] if p["label"] == case["label"])
            assert pair["reference"] == native["label"]
            assert float(row["paired_loss_error"]) == pair["loss_relative_error"]
            assert pair["loss_relative_error"] == abs(
                case["initial_probe"]["loss"] / native["initial_probe"]["loss"] - 1
            )
            assert float(row["paired_global_gradient_error"]) == pair["error"]["global_relative_l2"]
            assert float(row["paired_max_tensor_error"]) == pair["error"]["max_tensor_relative_l2"]
    for group in summary["groups"]:
        peers = [r for r in rows if r["scope"] == group["scope"] and r["policy"] == group["policy"]]
        assert group["fixtures"] == len(peers) == 3
        assert sorted(int(r["seed"]) for r in peers) == [101, 113, 127]
        for key, stats in group["metrics"].items():
            assert stats == describe([float(r[key]) for r in peers])
    qualified = []
    assert len(summary["decisions"]) == 2
    for decision in summary["decisions"]:
        assert decision["policy"] == "fp32_default_chunks"
        peers = [
            r for r in rows if r["scope"] == decision["scope"] and r["policy"] == decision["policy"]
        ]
        natives = [
            r
            for r in rows
            if r["scope"] == decision["scope"] and r["policy"] == "fp32_default_native"
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
            wall_bf16=all(float(r["wall_ratio_bf16"]) <= 1.25 for r in peers),
            wall_native=all(float(r["wall_ratio_native"]) <= 1.25 for r in peers),
            timing_stability=all(
                max(
                    float(r[k])
                    for k in ("timing_stability", "reference_stability", "native_stability")
                )
                <= 1.25
                for r in peers
            ),
            final_quality_bf16=all(float(r["nll_ratio_bf16"]) <= 1.01 for r in peers),
            final_quality_native=all(float(r["nll_ratio_native"]) <= 1.01 for r in peers),
            gradient_replay=all(r["audited_gradient_replay"] == "True" for r in peers + natives),
            independent_audit=audit["passed"],
        )
        assert gates == decision["gates"]
        assert all(gates.values()) == decision["qualifies_scoped_800_update_component"]
        assert not decision["new_training_allocation"]
        if all(gates.values()):
            qualified.append(dict(scope=decision["scope"], policy=decision["policy"]))
    assert qualified == summary["qualified_pairs"]
    assert summary["two_corpus_component_qualified"] == (len(qualified) == 2)
    assert len(summary["convergence"]) == 24
    for entry in summary["convergence"]:
        selected = [
            float(r["nll"])
            for r in curves
            if r["corpus"] == entry["corpus"]
            and r["policy"] == entry["policy"]
            and int(r["step"]) == entry["step"]
        ]
        assert len(selected) == 3 and describe(selected) == entry["statistics"]
    for entry in summary["pareto_by_fixture"]:
        peers = [r for r in rows if r["fixture"] == entry["fixture"]]
        keys = ("peak_job_mib", "wall_update_ms", "final_nll")
        calculated = [
            a["policy"]
            for a in peers
            if not any(
                all(float(b[k]) <= float(a[k]) for k in keys)
                and any(float(b[k]) < float(a[k]) for k in keys)
                for b in peers
            )
        ]
        assert calculated == entry["raw_nondominated_policies"]
    assert result["all_phase_peaks_recorded"] and summary["recorded_memory_phases"] == 28998
    assert summary["total_backward_passes_across_attempts"] == 14478
    for stage in ("study", "audit", "analyze", "plot_recovery"):
        assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
    assert not (ROOT / "failure.json").exists() and not (ROOT / "audit_failure.json").exists()
    packed = {}
    raw = [ROOT / "result.json", ROOT / "audit.json", *ROOT.glob("*.log"), *OLD.glob("*.log")]
    for path in raw:
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
        *OLD.glob("source/*.py"),
        ROOT / "source/README.md",
        *[
            ROOT / n
            for n in (
                "protocol.json",
                "audit_protocol.json",
                "plot_recovery_protocol.json",
                "qualification.json",
                "environment.json",
                "initializations.json",
                "summary.json",
            )
        ],
        *[
            OLD / n
            for n in (
                "protocol.json",
                "failure.json",
                "qualification.json",
                "environment.json",
                "study_exit.txt",
            )
        ],
        *ROOT.glob("*.csv.gz"),
        *ROOT.glob("*_exit.txt"),
        Path("research/fp32_training_replication_plan.md"),
        Path("research/fp32_training_replication_recovery_plan.md"),
        Path("research/fp32_training_replication_results.md"),
        Path("research/figures/fp32_training_replication.png"),
        *[Path(n) for n in protocol["before_documents"]],
    ]
    receipt = dict(
        status="PASS",
        study="H117",
        verification_scope="frozen evidence, fresh-state regeneration, native scoring, FP32 gradient replay and all-seed fixed-gate arithmetic",
        frozen_sources=129,
        maintained_files_unchanged=61,
        fresh_trials=18,
        optimizer_updates=14400,
        training_targets=58982400,
        profile_backward_passes=14418,
        successful_qualification_backward_passes=24,
        failed_startup_qualification_backward_passes=24,
        audit_backward_passes=12,
        total_backward_passes=14478,
        regenerated_initial_states=6,
        trained_checkpoints_verified=54,
        saved_tensor_artifacts=78,
        saved_gradient_hashes_verified=18,
        native_validation_scores=60,
        fp32_gradient_replays=12,
        bitwise_gradient_replays=audit["bitwise_gradient_replays"],
        all_audits_passed=audit["passed"],
        bf16_gradient_replay_claim=False,
        training_batches_verified=14400,
        recorded_memory_phases=28998,
        zero_allocator_boundaries=44,
        qualified_pairs=qualified,
        two_corpus_component_qualified=summary["two_corpus_component_qualified"],
        original_failed_training_updates=0,
        preserved_startup_failures=1,
        preserved_plot_import_access_violations=1,
        successful_unchanged_plot_recoveries=1,
        scientific_audit_repeated=False,
        training_repeated=False,
        no_default_change=True,
        broad_goal_achieved=False,
        previous_maintained_tests_passed=116,
        maintained_tests_rerun=False,
        tensor_hashes=tensors,
        local_metric_hashes=local_metrics,
        packed=packed,
        files={p.as_posix(): sha(p) for p in paths},
    )
    Path("results/verification/fp32_training_replication_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
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
