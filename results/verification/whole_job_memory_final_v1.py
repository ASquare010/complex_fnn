"""Independent CPU audit of H112 accounting, fixed decisions and preserved evidence."""

import csv
import gzip
import hashlib
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path("results/whole_job_memory_workspace_v1")
OLD = Path("results/whole_job_memory_v1")
CONT = Path("results/whole_job_memory_continuation_v1")
DIAG = Path("results/cublas_boundary_diagnosis_v1")


def read(path):
    return json.loads(Path(path).read_text())


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
    for path, digest in protocol["sources"].items():
        assert sha(path) == digest, path
    preserved = {}
    for name in ("runtime_recovery", "workspace_recovery"):
        recovery = protocol[name]
        assert recovery["previous_optimizer_updates"] == 0
        preserved.update(recovery["before"])
    for path, digest in preserved.items():
        assert sha(path) == digest, path
    audit_recovery = read(ROOT / "audit_recovery_before.json")
    assert not audit_recovery["training_repeated"] and not audit_recovery["root_cause_proved"]
    for path, digest in audit_recovery["files"].items():
        assert sha(path) == digest, path
    assert (ROOT / "audit_exit.txt").read_text().strip() == "3221225477"
    plot_recovery = read(ROOT / "plot_recovery_before.json")
    assert plot_recovery["optimizer_updates"] == 0 and not plot_recovery["cuda_requested"]
    for path, digest in plot_recovery["files"].items():
        assert sha(path) == digest, path
    assert (ROOT / "plot_exit.txt").read_text().strip() == "3221225477"
    for root in (OLD, CONT):
        assert not any(root.glob("*/runs/*/history.jsonl"))
        assert (root / "study_exit.txt").read_text().strip() == "1"
        assert (root / "failure.json").exists()
    assert (OLD / "qualification_exit.txt").read_text().strip() == "3221225477"
    assert read(CONT / "qualification.json")["passed"]
    diagnosis = read(DIAG / "result.json")
    assert diagnosis["optimizer_updates"] == 0 and len(diagnosis["records"]) == 2
    assert not diagnosis["native_import_failure_root_cause_proved"]
    for record in diagnosis["records"]:
        assert record["after_gc_empty_cache_bytes"] == 17039360
        assert record["after_cublas_clear_bytes"] == 0
        assert [b["size"] for b in record["active_blocks_before_clear"]] == [8519680] * 2
    h111 = read("results/verification/streamed_evaluation_final_v1.json")
    assert h111["status"] == "PASS" and h111["classifier_chunks_qualifies_both_shapes"]
    for path, digest in h111["files"].items():
        assert sha(path) == digest, path
    before_documents = read(ROOT / "before_documents.json")
    for name, digest in before_documents.items():
        assert sha(ROOT / "before_documents" / name.replace("/", "__")) == digest
    for dataset in protocol["datasets"].values():
        directory = Path(dataset["path"])
        assert sha(directory / "manifest.json") == dataset["manifest_sha256"]
        for name, digest in dataset["manifest"]["files"].items():
            assert sha(directory / name) == digest, directory / name
    result, summary, audit, qualification = [
        read(ROOT / name)
        for name in ("result.json", "summary.json", "audit.json", "qualification.json")
    ]
    assert result["status"] == summary["status"] == "COMPLETE"
    assert result["updates"] == summary["updates"] == 9600
    assert result["training_targets"] == summary["training_targets"] == 39321600
    assert result["repeated_scientific_updates"] == 0
    assert result["same_process_fresh_models"]
    assert len(result["boundaries"]) == 25
    assert all(
        b["allocated_bytes"] == b["reserved_bytes"] == 0
        and b["workspace_bytes_subtracted_from_trial_measurements"] == 0
        for b in result["boundaries"]
    )
    assert qualification["passed"] and len(qualification["checks"]) == 8
    assert audit["passed"] and audit["audit_updates"] == 0
    assert len(audit["scores"]) == 72 and audit["checkpoints"] == 48
    assert audit["initializations"] == 12
    assert audit["max_relative_score_difference"] <= 0.0001
    primary = {
        (r["dataset"], r["seed"], r["policy"]): r["native_full_nll"] for r in audit["trials"]
    }
    rows = {(r["dataset"], r["seed"], r["policy"]): r for r in result["cases"]}
    assert len(rows) == len(primary) == 12
    states = {}
    for key, row in rows.items():
        assert row["steps"] == 800 and row["training_targets"] == 3276800
        assert row["parameter_count"] == 9099648 and row["ffn_parameter_count"] == 2801664
        assert row["evaluation_policy"] == "classifier_chunks"
        assert row["weights_and_moments_finite"]
        assert len(row["records"]) == 800
        assert row["peak_training_allocated_bytes"] == max(
            h["peak_allocated_bytes"] for h in row["records"]
        )
        assert row["peak_job_allocated_bytes"] >= max(
            row["peak_training_allocated_bytes"], row["peak_validation_allocated_bytes"]
        )
        checkpoints = row["checkpoints"] + row["extra_checkpoints"]
        assert sorted(c["step"] for c in checkpoints) == [100, 200, 400, 800]
        for checkpoint in checkpoints:
            assert sha(checkpoint["path"]) == checkpoint["sha256"]
            states[checkpoint["path"]] = checkpoint["sha256"]
        native = [s for s in audit["scores"] if (s["dataset"], s["seed"], s["policy"]) == key]
        assert len(native) == 6
        assert [(s["step"], s["scope"]) for s in native] == [
            (0, "subset"),
            (100, "subset"),
            (200, "subset"),
            (400, "subset"),
            (800, "subset"),
            (800, "full"),
        ]
        assert native[-1]["native_nll"] == primary[key]
        for score in native:
            assert score["relative_difference_abs"] == abs(
                score["streamed_nll"] / score["native_nll"] - 1
            )
            assert score["relative_difference_abs"] <= 0.0001
    assert len(states) == 48
    for pair in summary["pairs"]:
        dataset, seed = pair["dataset"], pair["seed"]
        ref, candidate = [rows[dataset, seed, policy] for policy in ("block", "loss_chunks")]
        for key in (
            "initial_state_hashes",
            "initial_sampler_sha256",
            "final_sampler_sha256",
            "data_order_sha256",
        ):
            assert ref[key] == candidate[key]
        expected = {
            "native_nll_ratio": primary[dataset, seed, "loss_chunks"]
            / primary[dataset, seed, "block"],
            "job_memory_ratio": candidate["peak_job_allocated_bytes"]
            / ref["peak_job_allocated_bytes"],
            "training_memory_ratio": candidate["peak_training_allocated_bytes"]
            / ref["peak_training_allocated_bytes"],
            "update_time_ratio": candidate["timing"]["update_ms"]["median"]
            / ref["timing"]["update_ms"]["median"],
        }
        assert all(pair[k] == v for k, v in expected.items())
        gates = dict(
            native_nll_at_most_one_percent_worse=expected["native_nll_ratio"] <= 1.01,
            whole_job_saving_at_least_15_percent=expected["job_memory_ratio"] <= 0.85,
            median_update_cost_at_most_25_percent=expected["update_time_ratio"] <= 1.25,
            finite=True,
        )
        assert pair["gates"] == gates and pair["passes"] == all(gates.values())
    for group in summary["groups"]:
        dataset = group["dataset"]
        peers = [p for p in summary["pairs"] if p["dataset"] == dataset]
        assert len(peers) == 3
        assert group["all_seeds_pass"] == all(p["passes"] for p in peers)
        for key in (
            "native_nll_ratio",
            "job_memory_ratio",
            "training_memory_ratio",
            "update_time_ratio",
        ):
            assert group[key] == describe([p[key] for p in peers])
        for policy, group_stats in group["policies"].items():
            subset = [r for r in rows.values() if r["dataset"] == dataset and r["policy"] == policy]
            for key, values in group_stats.items():
                data = [
                    primary[dataset, r["seed"], policy]
                    if key == "native_full_nll"
                    else r["timing"]["update_ms"]["median"]
                    if key == "median_update_ms"
                    else r[key]
                    for r in subset
                ]
                assert values == describe(data)
    assert summary["two_corpus_component_qualifies"] == all(
        g["all_seeds_pass"] for g in summary["groups"]
    )
    assert not any(
        summary[k] for k in ("parameters_changed", "new_architecture", "broad_goal_achieved")
    )
    csv_rows = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((ROOT / "metrics.csv.gz").read_bytes()).decode())
        )
    )
    assert len(csv_rows) == 12
    for row in csv_rows:
        assert (
            float(row["native_full_nll"])
            == primary[row["dataset"], int(row["seed"]), row["policy"]]
        )
    assert len(json.loads(gzip.decompress((ROOT / "diagnostics.json.gz").read_bytes()))) == 12
    for mode in ("study", "audit_recovery", "analyze", "plot_recovery"):
        assert (ROOT / f"{mode}_exit.txt").read_text().strip() == "0"
    packed = {}
    for path in [
        ROOT / "result.json",
        *[p for root in (ROOT, OLD, CONT, DIAG) for p in root.glob("*.log")],
    ]:
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
        ROOT / "audit_recovery_before.json",
        ROOT / "plot_recovery_before.json",
        ROOT / "before_documents.json",
        ROOT / "source/README.md",
        ROOT / "metrics.csv.gz",
        ROOT / "diagnostics.json.gz",
        Path("research/whole_job_memory_results.md"),
        Path("research/figures/whole_job_memory.png"),
        *[Path(name) for name in before_documents],
        Path(".gitignore"),
        *ROOT.glob("source/*.py"),
    ]
    receipt = dict(
        status="PASS",
        frozen_sources=len(protocol["sources"]),
        maintained_files_unchanged=len(protocol["maintained_files"]),
        preserved_failed_attempt_and_diagnosis_files=len(preserved),
        preserved_audit_recovery_files=len(audit_recovery["files"]),
        preserved_plot_recovery_files=len(plot_recovery["files"]),
        preserved_before_documents=len(before_documents),
        fresh_trials=12,
        optimizer_updates=9600,
        training_targets=39321600,
        native_rescores=72,
        checkpoints=48,
        exact_initializations=12,
        max_relative_score_difference=audit["max_relative_score_difference"],
        zero_allocation_boundaries=25,
        workspace_bytes_subtracted=0,
        two_corpus_component_qualifies=summary["two_corpus_component_qualifies"],
        corpus_verdicts={g["dataset"]: g["verdict"] for g in summary["groups"]},
        previous_maintained_tests_passed=116,
        maintained_tests_rerun=False,
        broad_goal_achieved=False,
        checkpoint_hashes=states,
        packed=packed,
        files={p.as_posix(): sha(p) for p in files},
    )
    Path("results/verification/whole_job_memory_final_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("checkpoint_hashes", "packed", "files")},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
