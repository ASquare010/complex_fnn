"""Verify H115 evidence and unchanged gates; stdlib only, no new GPU execution."""

import csv
import gzip
import hashlib
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/decoder_gradient_transport_v1")
MODES = (
    "bf16_default_block",
    "bf16_default_none",
    "bf16_math_block",
    "fp32_default_block",
    "fp32_math_block",
)
POLICIES = ("native_fp32", "chunks_fp32")
BOUNDARIES = {"embedding", "normalized_hidden", *(f"block_{i}" for i in range(8))}


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


def csv_rows(name):
    return list(csv.DictReader(io.StringIO(gzip.decompress((ROOT / name).read_bytes()).decode())))


def compare(row, values):
    for key, value in values.items():
        if isinstance(value, (bool, str)):
            assert row[key] == str(value), (key, row[key], value)
        else:
            assert float(row[key]) == value, (key, row[key], value)


def error_valid(error):
    assert math.isfinite(error["global_relative_l2"]) and error["global_relative_l2"] >= 0
    per = error["per_tensor_relative_l2"]
    assert all(math.isfinite(v) and v >= 0 for v in per.values())
    assert max(per.values()) == error["max_tensor_relative_l2"]


def audit_errors(actual, expected):
    for key in ("global_relative_l2", "max_tensor_relative_l2"):
        assert math.isclose(actual[key], expected[key], abs_tol=1e-12, rel_tol=1e-10)
    assert set(actual["per_tensor_relative_l2"]) == set(expected["per_tensor_relative_l2"])
    for key, value in actual["per_tensor_relative_l2"].items():
        assert math.isclose(
            value, expected["per_tensor_relative_l2"][key], abs_tol=1e-12, rel_tol=1e-10
        )


def run():
    protocol = read(ROOT / "protocol.json")
    for field in ("sources", "maintained_files", "input_hashes", "previous_scientific_records"):
        hashes(protocol[field])
    assert len(protocol["sources"]) == 109 and len(protocol["maintained_files"]) == 61
    assert len(protocol["input_hashes"]) == 18
    prior_path = Path("results/verification/fp32_classifier_profile_final_v1.json")
    assert sha(prior_path) == protocol["previous_receipt_sha256"]
    prior = read(prior_path)
    for path, digest in prior["files"].items():
        actual = protocol["before_documents"].get(path, {}).get("preserved_path", path)
        assert sha(actual) == digest, path
    hashes(prior["tensor_hashes"])
    for path, info in prior["packed"].items():
        assert sha(path) == info["sha256"], path
    for info in protocol["before_documents"].values():
        assert sha(info["preserved_path"]) == info["sha256"]
    for dataset in protocol["datasets"].values():
        folder = Path(dataset["path"])
        assert sha(folder / "manifest.json") == dataset["manifest_sha256"]
        for name, digest in dataset["manifest"]["files"].items():
            assert sha(folder / name) == digest
    audit_protocol = read(ROOT / "audit_protocol.json")
    hashes(audit_protocol["files"])
    assert len(audit_protocol["files"]) == 64
    assert audit_protocol["backward_passes"] == audit_protocol["optimizer_updates"] == 0
    result, audit, summary, qualification = [
        read(ROOT / name)
        for name in ("result.json", "audit.json", "summary.json", "qualification.json")
    ]
    assert result["status"] == summary["status"] == "COMPLETE"
    assert len(result["conditions"]) == summary["conditions"] == protocol["conditions"] == 30
    for name, count in (
        ("classifier_backward_passes", 60),
        ("decoder_backward_passes", 240),
        ("qualification_backward_passes", 20),
        ("optimizer_updates", 0),
    ):
        assert result[name] == summary[name] == protocol[name] == count
    assert result["elapsed_seconds"] == summary["elapsed_seconds"] > 0
    for name in (
        "whole_job_memory_measured",
        "fresh_language_training_earned",
        "broad_goal_achieved",
    ):
        assert not result[name] and not summary[name]
    assert qualification["passed"] and qualification["qualification_backward_passes"] == 20
    assert qualification["optimizer_updates"] == 0
    recon = qualification["reconstruction_cases"]
    assert len(recon) == 4
    assert {(r["variant"], r["policy"]) for r in recon} == {
        (variant, policy) for variant in ("gelu", "swiglu") for policy in POLICIES
    }
    for row in recon:
        assert row["passed"]
        error_valid(row["error"])
        assert row["error"]["global_relative_l2"] == row["error"]["max_tensor_relative_l2"] == 0
    witness = qualification["rounding_witness"]
    assert len(witness) == 8
    low, high = 1 + 2**-8 - 2**-23, 1 + 2**-8 + 2**-23
    assert 2**-7 / (high - low) == 32768
    for device in ("cpu", "cuda"):
        for precision in ("bf16", "fp32"):
            selected = [r for r in witness if (r["device"], r["precision"]) == (device, precision)]
            assert [r["incoming"] for r in selected] == [low, high]
            values = [1.0, 1 + 2**-7] if precision == "bf16" else [low, high]
            for row, value in zip(selected, values, strict=True):
                assert row["expected"] == row["input_gradient"] == row["weight_gradient"] == value
    assert audit["passed"] and audit["no_claim_of_new_gradient_replay"]
    for name in (
        "native_forward_recaptures",
        "scalar_tensor_arithmetic_conditions",
        "source_model_checks",
    ):
        assert audit[name] == 30
    assert audit["saved_first_gradient_sets"] == audit["tensor_artifacts"] == 60
    assert (
        audit["backward_passes"]
        == summary["audit_backward_passes"]
        == audit["optimizer_updates"]
        == 0
    )
    assert len(audit["rows"]) == 30
    assert summary["independent_audit"] == {
        k: v for k, v in audit.items() if k not in ("rows", "boundaries")
    }
    for owner in (result, audit):
        assert len(owner["boundaries"]) == 31
        assert all(b == dict(allocated=0, reserved=0) for b in owner["boundaries"])
    rows, pairs, replays, traces = [
        csv_rows(name + ".csv.gz") for name in ("metrics", "pairs", "replays", "traces")
    ]
    assert [len(v) for v in (rows, pairs, replays, traces)] == [30, 120, 180, 1200]
    assert len({r["label"] for r in rows}) == 30
    tensors, local_metrics = {}, {}
    for case in result["conditions"]:
        label = case["label"]
        assert case["status"] == "COMPLETE" and case["all_finite"] and case["model_unchanged"]
        assert (
            not case["whole_job_memory_measured"]
            and case["instrumented_hooks_can_affect_scheduling"]
        )
        assert case["classifier_backward_passes"] == 2 and case["decoder_backward_passes"] == 8
        assert case["optimizer_updates"] == 0 and set(case["forward_hashes"]) == BOUNDARIES
        expected_nodes = (
            ["ScaledDotProductEfficientAttentionBackward0"] if "default" in case["mode"] else []
        )
        assert case["attention_backward_nodes"] == expected_nodes
        assert len(case["artifacts"]) == 2
        for artifact in case["artifacts"]:
            assert sha(artifact["path"]) == artifact["sha256"]
            tensors[artifact["path"]] = artifact["sha256"]
        metric_path = ROOT / "conditions" / label / "metrics.json"
        assert read(metric_path) == case
        local_metrics[metric_path.as_posix()] = sha(metric_path)
        expected_phases = ["construction_and_data", "forward_capture"]
        expected_phases += ["classifier_" + policy for policy in POLICIES]
        expected_phases += [f"decoder_{i}_{policy}" for i in range(4) for policy in POLICIES]
        expected_phases += ["serialization_and_final_integrity"]
        assert [p["phase"] for p in case["memory_phases"]] == expected_phases
        for kind in ("allocated", "reserved"):
            assert case[f"peak_diagnostic_{kind}_bytes"] == max(
                p[f"peak_{kind}_bytes"] for p in case["memory_phases"]
            )
        for phase in case["memory_phases"]:
            assert 0 <= phase["live_allocated_bytes"] <= phase["peak_allocated_bytes"]
            assert 0 <= phase["live_reserved_bytes"] <= phase["peak_reserved_bytes"]
        assert [(r["repetition"], r["policy"]) for r in case["records"]] == [
            (i, policy) for i in range(4) for policy in POLICIES
        ]
        for record in case["records"]:
            assert record["forward_equal"] and record["forward_hashes"] == case["forward_hashes"]
            assert (
                math.isfinite(record["instrumented_seconds"]) and record["instrumented_seconds"] > 0
            )
            assert set(record["replay_trace"]) == BOUNDARIES
            for kind in ("decoder", "total"):
                error_valid(record["replay_" + kind])
            first = next(r for r in case["records"] if r["policy"] == record["policy"])
            assert record["bitwise_replay_total"] == (record["total_hash"] == first["total_hash"])
        nontrivial = [r for r in case["records"] if r["repetition"] > 0]
        row = next(r for r in rows if r["label"] == label)
        incoming = case["classifier_difference"]["hidden_gradient"]["relative_l2"]
        values = dict(
            label=label,
            fixture=case["fixture"]["label"],
            scope=case["fixture"]["scope"],
            mode=case["mode"],
            backend_nodes="|".join(expected_nodes),
            incoming_hidden_gradient_error=incoming,
            classifier_weight_gradient_error=case["classifier_difference"]["weight_gradient"][
                "relative_l2"
            ],
            paired_total_median=st.median(
                p["total"]["global_relative_l2"] for p in case["paired_repetitions"]
            ),
            paired_total_max=max(
                p["total"]["global_relative_l2"] for p in case["paired_repetitions"]
            ),
            paired_tensor_max=max(
                p["total"]["max_tensor_relative_l2"] for p in case["paired_repetitions"]
            ),
            maximum_replay_global=max(r["replay_total"]["global_relative_l2"] for r in nontrivial),
            maximum_replay_tensor=max(
                r["replay_total"]["max_tensor_relative_l2"] for r in nontrivial
            ),
            bitwise_repeat_matches=sum(r["bitwise_replay_total"] for r in nontrivial),
            repeated_comparisons=6,
            all_forward_boundaries_equal=True,
            peak_diagnostic_mib=case["peak_diagnostic_allocated_bytes"] / 2**20,
            peak_reserved_mib=case["peak_diagnostic_reserved_bytes"] / 2**20,
            instrumented_pass_median_seconds=st.median(
                r["instrumented_seconds"] for r in case["records"]
            ),
        )
        values["paired_relative_amplification_median"] = values["paired_total_median"] / incoming
        values["transport_gate_passed"] = (
            values["paired_total_max"] <= 0.002 and values["paired_tensor_max"] <= 0.02
        )
        values["repeat_gate_passed"] = (
            values["maximum_replay_global"] <= 1e-5 and values["maximum_replay_tensor"] <= 1e-4
        )
        compare(row, values)
        assert [p["repetition"] for p in case["paired_repetitions"]] == list(range(4))
        for pair in case["paired_repetitions"]:
            for kind in ("decoder", "total"):
                error_valid(pair[kind])
            packed = [
                p
                for p in pairs
                if p["label"] == label and int(p["repetition"]) == pair["repetition"]
            ]
            assert len(packed) == 1 and set(pair["trace"]) == BOUNDARIES
            compare(
                packed[0],
                dict(
                    mode=case["mode"],
                    decoder_global_error=pair["decoder"]["global_relative_l2"],
                    total_global_error=pair["total"]["global_relative_l2"],
                    total_max_tensor_error=pair["total"]["max_tensor_relative_l2"],
                ),
            )
            for name, error in pair["trace"].items():
                selected = [
                    t
                    for t in traces
                    if t["label"] == label
                    and int(t["repetition"]) == pair["repetition"]
                    and t["boundary"] == name
                ]
                assert len(selected) == 1
                compare(
                    selected[0],
                    dict(
                        mode=case["mode"],
                        relative_l2=error["relative_l2"],
                        absolute_error=error["max_absolute_error"],
                        relative_amplification=error["relative_l2"] / incoming,
                    ),
                )
        for record in nontrivial:
            selected = [
                r
                for r in replays
                if r["label"] == label
                and int(r["repetition"]) == record["repetition"]
                and r["policy"] == record["policy"]
            ]
            assert len(selected) == 1
            compare(
                selected[0],
                dict(
                    mode=case["mode"],
                    global_error=record["replay_total"]["global_relative_l2"],
                    max_tensor_error=record["replay_total"]["max_tensor_relative_l2"],
                    bitwise_equal=record["bitwise_replay_total"],
                ),
            )
        audited = next(a for a in audit["rows"] if a["label"] == label)
        assert audited["native_forward_equal"] and audited["forward_relative_drift"] == 0
        assert audited["arithmetic_verified"] and audited["initial_gradient_hashes_verified"]
        for kind in ("decoder", "total"):
            audit_errors(audited["paired_errors"][kind], case["paired_repetitions"][0][kind])
    assert len(tensors) == 60 and len(local_metrics) == 30
    for fixture in protocol["fixtures"]:
        selected = [c for c in result["conditions"] if c["fixture"] == fixture]
        assert len(selected) == 5 and {c["mode"] for c in selected} == set(MODES)
        assert len({c["initial_model_hash"] for c in selected}) == 1
    for group in summary["groups"]:
        selected = [r for r in rows if r["mode"] == group["mode"]]
        assert group["fixtures"] == len(selected) == 6 and group["repeated_comparisons"] == 36
        assert group["bitwise_repeat_matches"] == sum(
            int(r["bitwise_repeat_matches"]) for r in selected
        )
        for key, stats in group["metrics"].items():
            assert describe([float(r[key]) for r in selected]) == stats
    assert {g["mode"] for g in summary["groups"]} == set(MODES)
    candidates = []
    for decision in summary["decisions"]:
        selected = [r for r in rows if r["mode"] == decision["mode"]]
        gates = dict(
            all_forward_boundaries_equal=all(
                r["all_forward_boundaries_equal"] == "True" for r in selected
            ),
            all_transport_gates=all(r["transport_gate_passed"] == "True" for r in selected),
            all_repeat_gates=all(r["repeat_gate_passed"] == "True" for r in selected),
        )
        assert decision["gates"] == gates
        assert decision["qualifies_uninstrumented_resource_screen"] == all(gates.values())
        assert (
            not decision["fresh_language_training_earned"]
            and not decision["global_determinism_proved"]
        )
        if all(gates.values()):
            candidates.append(decision["mode"])
    assert (
        candidates
        == summary["resource_screen_candidates"]
        == ["fp32_default_block", "fp32_math_block"]
    )
    assert len(summary["cross_mode"]) == 6
    for row in summary["cross_mode"]:
        forward = row["forward_comparisons_to_bf16_default_block"]
        assert (
            forward["bf16_default_none"]["bitwise_equal"]
            and forward["bf16_default_none"]["relative_l2"] == 0
        )
    env = read(ROOT / "environment.json")
    assert env["threads"] == 4 and not env["tf32"] and not env["deterministic_algorithms"]
    assert env["cublas_workspace_config"] is None and env["cuda_available"]
    for stage in ("study", "audit", "analyze", "plot"):
        assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
    assert not list(ROOT.glob("*failure*.json"))
    packed_files = {}
    for path in (ROOT / "result.json", ROOT / "audit.json", *ROOT.glob("*.log")):
        data = path.read_bytes()
        target = path.with_name(path.name + ".gz")
        target.write_bytes(gzip.compress(data, mtime=0))
        assert gzip.decompress(target.read_bytes()) == data
        packed_files[target.as_posix()] = dict(
            sha256=sha(target), original_sha256=sha(path), bytes=target.stat().st_size
        )
    paths = [
        Path(__file__),
        *ROOT.glob("source/*.py"),
        ROOT / "source/README.md",
        *[
            ROOT / name
            for name in (
                "protocol.json",
                "audit_protocol.json",
                "qualification.json",
                "summary.json",
                "environment.json",
            )
        ],
        *ROOT.glob("*.csv.gz"),
        *ROOT.glob("*_exit.txt"),
        Path("research/decoder_gradient_transport_plan.md"),
        Path("research/decoder_gradient_transport_results.md"),
        Path("research/figures/decoder_gradient_transport.png"),
        *[Path(name) for name in protocol["before_documents"]],
    ]
    receipt = dict(
        status="PASS",
        study="H115",
        verification_scope="frozen evidence, native forward recapture, stored gradient arithmetic and numerical diagnostic gates",
        frozen_sources=109,
        maintained_files_unchanged=61,
        frozen_input_files=18,
        source_checkpoints=6,
        conditions=30,
        saved_tensor_artifacts=60,
        saved_first_gradient_sets=60,
        classifier_backward_passes=60,
        decoder_backward_passes=240,
        qualification_backward_passes=20,
        qualification_cpu_backward_passes=16,
        qualification_cuda_backward_passes=4,
        total_scientific_backward_passes=320,
        optimizer_updates=0,
        native_forward_recaptures=30,
        independent_audit_backward_passes=0,
        independent_full_gradient_replay_claim=False,
        chain_rule_reconstruction_cases=4,
        scalar_rounding_cases=8,
        scalar_absolute_amplification=32768,
        recorded_memory_phases=390,
        zero_allocator_boundaries=62,
        condition_rows=30,
        paired_rows=120,
        nontrivial_replay_rows=180,
        trace_rows=1200,
        resource_screen_candidates=candidates,
        whole_job_memory_measured=False,
        fresh_language_training_earned=False,
        fresh_training_runs=0,
        previous_maintained_tests_passed=116,
        maintained_tests_rerun=False,
        broad_goal_achieved=False,
        tensor_hashes=tensors,
        local_metric_hashes=local_metrics,
        packed=packed_files,
        files={p.as_posix(): sha(p) for p in paths},
    )
    Path("results/verification/decoder_gradient_transport_final_v1.json").write_text(
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
