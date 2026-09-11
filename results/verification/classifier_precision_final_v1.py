"""Verify H113's frozen evidence, every gate/statistic and compact exports."""

import csv
import gzip
import hashlib
import io
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("results/classifier_precision_v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    return dict(
        mean=st.mean(values),
        median=st.median(values),
        sample_variance=st.variance(values),
        min=min(values),
        max=max(values),
        n=len(values),
    )


def run():
    protocol = read(ROOT / "protocol.json")
    for name, digest in protocol["sources"].items():
        assert sha(name) == digest, name
    for name, digest in protocol["checkpoint_hashes"].items():
        assert sha(name) == digest, name
    prior = Path("results/verification/whole_job_memory_final_v1.json")
    assert sha(prior) == protocol["previous_receipt_sha256"]
    for name, digest in protocol["previous_results"].items():
        assert sha(name) == digest, name
    for name, digest in read(prior)["files"].items():
        path = (
            protocol["before_documents"][name]["preserved_path"]
            if name in protocol["before_documents"]
            else ROOT / "before_documents/gitignore"
            if name == ".gitignore"
            else name
        )
        assert sha(path) == digest, name
    for dataset in protocol["datasets"].values():
        directory = Path(dataset["path"])
        assert sha(directory / "manifest.json") == dataset["manifest_sha256"]
        for name, digest in dataset["manifest"]["files"].items():
            assert sha(directory / name) == digest
    result, summary, audit, qualification = [
        read(ROOT / name)
        for name in ("result.json", "summary.json", "audit.json", "qualification.json")
    ]
    assert result["status"] == summary["status"] == "COMPLETE"
    assert result["fixtures"] == summary["fixtures"] == audit["fixtures"] == 24
    assert result["classifier_backward_passes"] == summary["classifier_backward_passes"] == 504
    assert (
        result["optimizer_updates"]
        == summary["optimizer_updates"]
        == audit["optimizer_updates"]
        == 0
    )
    assert audit["passed"] and audit["exact_gradient_reruns"] == 144
    assert audit["native_recaptures"] == audit["closed_form_reference_checks"] == 24
    assert audit["scalar_error_records"] == 240
    assert qualification["passed"] and len(qualification["double_checks"]) == 10
    assert len(qualification["finite_difference_policies"]) == 3
    assert len(qualification["linear_witness"]) == 8
    for witness in qualification["linear_witness"]:
        expected = 256 if witness["cached"] and witness["terms"][-1] == 256 else 260
        assert witness["gradient"] == expected
        assert witness["weight_cast_nodes"] == (1 if witness["cached"] else 5)
    hashes = {}
    for row in result["rows"]:
        for path, digest in (
            (row["path"], row["sha256"]),
            (row["gradients_path"], row["gradients_sha256"]),
        ):
            assert sha(path) == digest
            hashes[path] = digest
        assert len(row["records"]) == 5
        for record in row["records"]:
            assert len(record["repetitions"]) == 3
            assert all(
                math.isfinite(r[k]) and r[k] >= 0
                for r in record["repetitions"]
                for k in ("forward_ms", "backward_ms", "total_ms")
            )
            assert record["median_total_ms"] == st.median(
                r["total_ms"] for r in record["repetitions"]
            )
            assert record["peak_allocated_bytes"] == max(
                r["peak_allocated_bytes"] for r in record["repetitions"]
            )
    assert len(hashes) == 48
    assert summary["cache_mechanism_supported"]
    for topology in summary["topology"]:
        assert topology["cached_cast_nodes"] == 1 and topology["uncached_cast_nodes"] == 8
        assert topology["loss_equal"] and topology["hidden_gradient_equal"]
        assert not topology["weight_gradient_equal"]
        assert topology["cached_cast_gradient_dtypes"] == ["torch.bfloat16"]
        assert topology["uncached_cast_gradient_dtypes"] == ["torch.bfloat16"] * 8
        source = next(r for r in result["rows"] if r["label"] == topology["label"])
        assert all(topology[k] == v for k, v in source["comparison"].items())
    entries = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode())
        )
    )
    assert len(entries) == 120
    for group in summary["groups"] + summary["by_step"]:
        peers = [
            r
            for r in entries
            if r["dataset"] == group["dataset"]
            and r["policy"] == group["policy"]
            and ("step" not in group or int(r["step"]) == group["step"])
        ]
        for key, stats in group["metrics"].items():
            assert stats == describe([float(r[key]) for r in peers]), (
                group["dataset"],
                group["policy"],
                key,
            )
    for decision in summary["decisions"]:
        peers = [
            r
            for r in entries
            if r["dataset"] == decision["dataset"] and r["policy"] == decision["policy"]
        ]
        gates = dict(
            median_weight_error_ratio_at_most_0_75=st.median(
                float(r["weight_error_ratio_to_cached"]) for r in peers
            )
            <= 0.75,
            all_hidden_error_ratios_at_most_1_01=all(
                float(r["hidden_error_ratio_to_cached"]) <= 1.01 for r in peers
            ),
            all_memory_ratios_at_most_0_85=all(
                float(r["memory_ratio_to_native_bf16"]) <= 0.85 for r in peers
            ),
            median_time_ratio_at_most_1_5=st.median(
                float(r["time_ratio_to_native_bf16"]) for r in peers
            )
            <= 1.5,
        )
        assert decision["gates"] == gates
        assert decision["earns_whole_model_resource_screen"] == all(gates.values())
    assert summary["whole_model_screen_candidates"] == ["chunks_fp32"]
    assert not summary["whole_diagnostic_process_peak_measured"]
    assert not result["full_training_job_measured"] and not summary["language_quality_proved"]
    for mode in ("study", "analyze", "audit", "plot"):
        assert (ROOT / f"{mode}_exit.txt").read_text().strip() == "0"
    assert not (ROOT / "failure.json").exists() and not (ROOT / "audit_failure.json").exists()
    packed = {}
    for path in [ROOT / "result.json", *ROOT.glob("*.log")]:
        target = path.with_name(path.name + ".gz")
        target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        assert gzip.decompress(target.read_bytes()) == path.read_bytes()
        packed[target.as_posix()] = dict(
            sha256=sha(target), original_sha256=sha(path), bytes=target.stat().st_size
        )
    files = [
        Path(__file__),
        ROOT / "protocol.json",
        ROOT / "summary.json",
        ROOT / "audit.json",
        ROOT / "qualification.json",
        ROOT / "captures.json",
        ROOT / "metrics.csv.gz",
        ROOT / "source/README.md",
        *ROOT.glob("source/*.py"),
        Path("research/classifier_precision_plan.md"),
        Path("research/classifier_precision_results.md"),
        Path("research/figures/classifier_precision.png"),
        *[Path(n) for n in protocol["before_documents"]],
        Path(".gitignore"),
    ]
    receipt = dict(
        status="PASS",
        frozen_sources=len(protocol["sources"]),
        maintained_files_unchanged=len(protocol["maintained_files"]),
        source_checkpoints=24,
        tensor_artifacts=48,
        mathematical_double_cases=10,
        finite_difference_policies=3,
        rounding_witnesses=8,
        policy_comparisons=120,
        classifier_backward_passes=504,
        optimizer_updates=0,
        exact_native_recaptures=24,
        exact_gradient_reruns=144,
        closed_form_reference_checks=24,
        cache_mechanism_supported=True,
        changed_weight_gradients=24,
        whole_model_screen_candidates=summary["whole_model_screen_candidates"],
        whole_diagnostic_peak_incomplete=True,
        local_timing_variability_requires_requalification=True,
        previous_maintained_tests_passed=116,
        maintained_tests_rerun=False,
        broad_goal_achieved=False,
        tensor_hashes=hashes,
        packed=packed,
        files={p.as_posix(): sha(p) for p in files},
    )
    Path("results/verification/classifier_precision_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("tensor_hashes", "packed", "files")},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
