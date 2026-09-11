"""Independently check H110 arithmetic, frozen evidence and compact publication."""

import csv
import gzip
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path("results/token_memory_duration_fresh_v1")
OLD = Path("results/token_memory_duration_v1")
RECOVERY = Path("results/token_memory_duration_recovery_v1")
DIAGNOSIS = Path("results/runtime_import_diagnosis_v1")
DESTINATION = Path("results/verification/token_memory_duration_final_v1.json")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def describe(values):
    return {
        "mean": st.mean(values),
        "median": st.median(values),
        "sample_variance": st.variance(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def run():
    protocol = read(OLD / "protocol.json")
    assert (ROOT / "protocol.json").read_bytes() == (OLD / "protocol.json").read_bytes()
    for path, digest in protocol["sources"].items():
        assert sha(path) == digest, path
    preserved = read(ROOT / "before.json")["before"]
    for path, digest in preserved.items():
        assert sha(path) == digest, path
    previous = read("results/verification/sobolev_learning_final_v1.json")
    assert previous["status"] == "PASS" and previous["previous_maintained_tests_passed"] == 116
    for path, digest in previous["files"].items():
        assert sha(path) == digest, path
    for root in (OLD, RECOVERY):
        assert read(root / "coordinator_status.json")["status"] == "WORKER_FAILED"
        assert (root / "b16_t128_s17_loss_chunks_exit.txt").read_text().strip() == "3221225477"
        assert (root / "study_exit.txt").read_text().strip() == "1"
    assert read(OLD / "audit.json")["passed"]
    assert read(OLD / "partial_result.json")["candidate_updates"] == 0
    runtime = read(DIAGNOSIS / "result.json")
    assert runtime["status"] == "COMPLETE" and runtime["neural_updates"] == 0
    assert not runtime["root_cause_proved"]
    assert runtime["integrity"] == {"checked_hashed_files": 1572, "missing": [], "mismatched": []}
    for path, digest in read(DIAGNOSIS / "protocol.json")["sources"].items():
        assert sha(path) == digest, path
    runtime_counts = {}
    for version, fresh in (
        ("3.12.9", False),
        ("3.12.9", True),
        ("3.12.12", False),
        ("3.12.12", True),
    ):
        probes = [
            r for r in runtime["rows"] if r["python"] == version and r["fresh_bytecode"] == fresh
        ]
        assert len(probes) == 3
        runtime_counts[f"{version}_fresh={fresh}"] = sum(r["exit_code"] == 0 for r in probes)
    assert list(runtime_counts.values()) == [1, 3, 2, 2]
    for root in (OLD, ROOT):
        q = read(root / "qualification.json")
        assert q["passed"] and len(q["checks"]) == 8
    result, summary, audit = [read(ROOT / n) for n in ("result.json", "summary.json", "audit.json")]
    assert result["status"] in ("COMPLETE", "STOPPED_FIDELITY_GATE", "RUNTIME_INTERRUPTED")
    assert summary["experiment_status"] == result["status"]
    rows = result["cases"]
    lookup = {(r["context"], r["seed"], r["policy"]): r for r in rows}
    assert len(lookup) == len(rows) == summary["completed_trials"]
    assert result["updates"] == summary["updates"] == 800 * len(rows)
    assert summary["training_targets"] == sum(800 * r["batch"] * r["context"] for r in rows)
    assert len(audit["trials"]) == len(rows) and audit["passed"]
    assert len(audit["scores"]) == 5 * len(rows) and audit["checkpoints"] == 3 * len(rows)
    assert audit["max_nll_difference"] == 0 and audit["audit_updates"] == 0
    for row in rows:
        source_root = Path(row.get("source_root", OLD))
        assert (source_root / f"{row['label']}_exit.txt").read_text().strip() == "0"
        original = read(source_root / "runs" / row["label"] / "metrics.json")
        assert {k: v for k, v in row.items() if k != "source_root"} == original
        assert row["parameter_count"] == 9099648 and row["ffn_parameter_count"] == 2801664
        assert len(row["records"]) == 800 and row["steps"] == 800
        assert row["weights_and_moments_finite"]
        timed = row["records"][20:]
        for key in ("update_ms", "forward_ms", "backward_ms", "optimizer_ms"):
            assert row["timing"][key] == {
                "mean": st.mean(r[key] for r in timed),
                "median": st.median(r[key] for r in timed),
                "sample_variance": st.variance(r[key] for r in timed),
            }
        assert row["peak_training_allocated_bytes"] == max(
            r["peak_allocated_bytes"] for r in row["records"]
        )
        assert row["peak_training_reserved_bytes"] == max(
            r["peak_reserved_bytes"] for r in row["records"]
        )
        close(row["clip_fraction"], sum(r["preclip_norm"] > 1 for r in row["records"]) / 800)
    assert len(summary["comparisons"]) == len(result["pairs"])
    for pair, recorded in zip(summary["comparisons"], result["pairs"], strict=True):
        ref, candidate = [
            lookup[pair["context"], pair["seed"], p] for p in ("block", "loss_chunks")
        ]
        for key in (
            "initial_state_hashes",
            "initial_sampler_sha256",
            "final_sampler_sha256",
            "data_order_sha256",
        ):
            assert ref[key] == candidate[key], key
        memory = candidate["peak_training_allocated_bytes"] / ref["peak_training_allocated_bytes"]
        job = candidate["peak_job_allocated_bytes"] / ref["peak_job_allocated_bytes"]
        timing = candidate["timing"]["update_ms"]["median"] / ref["timing"]["update_ms"]["median"]
        nll = candidate["full_validation"]["nll"] / ref["full_validation"]["nll"] - 1
        trajectory = {
            str(a["step"]): b["nll"] / a["nll"] - 1
            for a, b in zip(ref["evaluations"], candidate["evaluations"], strict=True)
        }
        for key, value in (
            ("training_memory_ratio", memory),
            ("job_memory_ratio", job),
            ("update_time_ratio", timing),
            ("full_relative_nll_change", nll),
        ):
            close(pair[key], value)
        assert pair["trajectory_relative_nll_changes"] == trajectory
        gates = {
            "training_memory_saving_at_least_15_percent": memory <= 0.85,
            "median_update_cost_at_most_25_percent": timing <= 1.25,
            "full_nll_within_one_percent": abs(nll) <= 0.01,
            "trajectory_nll_within_one_percent": all(
                abs(v) <= 0.01 for k, v in trajectory.items() if k != "0"
            ),
            "finite": True,
        }
        assert gates == pair["gates"] and pair["passes"] == all(gates.values())
        assert recorded["fidelity_passes"] == (
            abs(nll) <= 0.01 and all(abs(v) <= 0.01 for v in trajectory.values())
        )
    for group in summary["groups"]:
        pairs = [p for p in summary["comparisons"] if p["context"] == group["context"]]
        assert group["paired_seeds"] == [p["seed"] for p in pairs]
        for key in (
            "training_memory_ratio",
            "job_memory_ratio",
            "update_time_ratio",
            "full_relative_nll_change",
        ):
            assert group[key] == describe([p[key] for p in pairs])
        for policy in ("block", "loss_chunks"):
            peers = [r for r in rows if r["context"] == group["context"] and r["policy"] == policy]
            for key, getter in (
                ("full_nll", lambda r: r["full_validation"]["nll"]),
                ("training_peak_bytes", lambda r: r["peak_training_allocated_bytes"]),
                ("validation_peak_bytes", lambda r: r["peak_validation_allocated_bytes"]),
                ("job_peak_bytes", lambda r: r["peak_job_allocated_bytes"]),
                ("median_update_ms", lambda r: r["timing"]["update_ms"]["median"]),
            ):
                assert group["policies"][policy][key] == describe([getter(r) for r in peers])
        passes = len(pairs) == 3 and all(p["passes"] for p in pairs)
        assert group["verdict"] == (
            "QUALIFIED_800_STEP_EXECUTION_RESULT"
            if passes
            else "FIXED_DURATION_RECIPE_NOT_QUALIFIED"
        )
        assert group["whole_job_saving_at_least_15_percent_each_seed"] == (
            len(pairs) == 3 and all(p["job_memory_ratio"] <= 0.85 for p in pairs)
        )
    if result["status"] == "COMPLETE":
        assert len(rows) == 12 and len(result["pairs"]) == 6
    elif result["status"] == "STOPPED_FIDELITY_GATE":
        assert not result["pairs"][-1]["fidelity_passes"] and all(
            p["fidelity_passes"] for p in result["pairs"][:-1]
        )
    assert not summary["breakthrough"] and not summary["parameters_changed"]
    with gzip.open(ROOT / "metrics.csv.gz", "rt", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == len(rows)
    for item, row in zip(table, rows, strict=True):
        assert item["label"] == row["label"]
        close(float(item["full_validation_nll"]), row["full_validation"]["nll"])
        close(float(item["median_update_ms"]), row["timing"]["update_ms"]["median"])
    for stage in ("audit", "analyze", "plot"):
        assert (ROOT / f"{stage}_exit.txt").read_text().strip() == "0"
    packed = {}
    for root in (OLD, RECOVERY, ROOT, DIAGNOSIS):
        for path in sorted(root.glob("*.log")) + [
            p for p in (root / "result.json", root / "partial_result.json") if p.exists()
        ]:
            target = path.with_name(path.name + ".gz")
            target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            assert gzip.decompress(target.read_bytes()) == path.read_bytes()
            packed[target.as_posix()] = {
                "sha256": sha(target),
                "original_sha256": sha(path),
                "bytes": target.stat().st_size,
            }
    documents = [
        Path("research/token_memory_duration_results.md"),
        Path("research/runtime_import_diagnosis.md"),
        ROOT / "source/README.md",
        OLD / "source/README.md",
    ]
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if not link.startswith("https:"):
                target = (document.parent / link).resolve()
                assert target == DESTINATION.resolve() or target.exists(), (document, link)
    public = [
        *documents,
        Path("research/token_memory_duration_plan.md"),
        Path("research/token_memory_duration_recovery_plan.md"),
        Path("research/figures/token_memory_duration.png"),
        Path(__file__),
    ]
    for root in (OLD, RECOVERY, ROOT, DIAGNOSIS):
        public += sorted(root.glob("source/*.py"))
        public += sorted(root.glob("*_exit.txt"))
        public += [
            p
            for p in root.glob("*.json")
            if p.name not in ("result.json", "partial_result.json", "progress.json")
        ]
    public += [ROOT / "metrics.csv.gz", ROOT / "diagnostics.json.gz", DIAGNOSIS / "result.json"]
    assert all(p.stat().st_size < 5 * 2**20 for p in public + [Path(p) for p in packed])
    receipt = {
        "status": "PASS",
        "experiment_status": result["status"],
        "frozen_scientific_sources_verified": len(protocol["sources"]),
        "preserved_recovery_files_verified": len(preserved),
        "maintained_files_unchanged": len(protocol["provenance"]["source_files"]),
        "completed_trials": len(rows),
        "fresh_updates": 800 * len(rows),
        "reused_control_updates_in_continuation": 800,
        "new_updates_in_continuation": 800 * (len(rows) - 1),
        "repeated_scientific_updates": 0,
        "audited_checkpoints": 3 * len(rows),
        "independent_nll_checks": 5 * len(rows),
        "max_nll_rescore_difference": 0,
        "audit_updates": 0,
        "qualification_cases_original": 8,
        "qualification_cases_continuation": 8,
        "runtime_probe_successes": runtime_counts,
        "native_root_cause_proved": False,
        "two_pretraining_failures_preserved": True,
        "previous_maintained_tests_passed": 116,
        "maintained_tests_rerun_in_h110": False,
        "broad_goal_achieved": False,
        "packed": packed,
        "files": {p.as_posix(): sha(p) for p in public},
    }
    DESTINATION.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("packed", "files")}, indent=2))


if __name__ == "__main__":
    run()
