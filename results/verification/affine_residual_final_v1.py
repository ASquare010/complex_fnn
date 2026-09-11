"""Verify decisions, preserve frozen sources and package H106/H107 evidence."""

import csv
import gzip
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

CAPACITY = Path("results/affine_residual_capacity_v1")
FIT = Path("results/affine_residual_fit_v1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    frozen = {}
    for folder in (CAPACITY, FIT):
        sources = read(folder / "protocol.json")["sources"]
        for path, digest in sources.items():
            assert sha(path) == digest, path
            frozen[path] = digest
        assert read(folder / "qualification.json")["passed"]
        assert read(folder / "audit.json")["passed"]
        assert (folder / "study_exit.txt").read_text().strip() == "0"
        audit_exit = "audit_exit.txt" if folder == CAPACITY else "audit_recovery_exit.txt"
        assert (folder / audit_exit).read_text().strip() == "0"
    capacity, result, summary, audit = (
        read(p)
        for p in (
            CAPACITY / "result.json",
            FIT / "result.json",
            FIT / "summary.json",
            FIT / "audit.json",
        )
    )
    assert len(capacity["rows"]) == 126 and read(CAPACITY / "audit.json")["count"] == 126
    assert {k for k, v in capacity["decisions"].items() if v["verdict"] == "EARNS_FITTING"} == {
        "gelu256",
        "swiglu170",
    }
    assert len(result["rows"]) == 144 and result["neural_updates"] == 172800
    assert result["resource_profile_updates"] == 540 and len(result["diagnostics"]) == 18
    assert len(audit["scores"]) == 864 and len(audit["datasets"]) == 18 and audit["selected"] == 144
    assert len(audit["exports"]) == len(audit["initializations"]) == 144
    lookup = {(r["teacher"], r["layer"], r["seed"], r["form"]): r for r in result["rows"]}
    for item in summary["statistics"]:
        peers = [
            r
            for r in result["rows"]
            if (r["teacher"], r["form"]) == (item["teacher"], item["form"])
        ]
        values = [r["reporting_mse"] for r in peers]
        assert item["reporting_mse"]["mean"] == st.mean(values)
        assert item["reporting_mse"]["median"] == st.median(values)
        assert item["reporting_mse"]["sample_variance"] == st.variance(values)
    for candidate, decision in summary["decisions"].items():
        assert decision["verdict"] == "CLOSED_AT_THIS_RECIPE"
        for teacher, data in decision["teachers"].items():
            peers = [r for r in result["rows"] if (r["teacher"], r["form"]) == (teacher, candidate)]
            for control, ratio in data["paired_geomean_ratios"].items():
                independently = math.prod(
                    r["reporting_mse"]
                    / lookup[teacher, r["layer"], r["seed"], control]["reporting_mse"]
                    for r in peers
                ) ** (1 / 9)
                assert abs(independently - ratio) < 1e-12
            assert not data["gates"]["mean_error_at_most_five_percent"]
            assert not data["gates"]["five_percent_better_than_every_control"]
            assert data["passes"] == all(data["gates"].values())
    with gzip.open(FIT / "metrics.csv.gz", "rt", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 432 and sum(r["selected"] == "True" for r in table) == 144
    for record in table:
        row = lookup[record["teacher"], int(record["layer"]), int(record["seed"]), record["form"]]
        endpoint = next(e for e in row["endpoints"] if e["rate"] == float(record["rate"]))
        assert float(record["reporting_mse"]) == endpoint["reporting_mse"]
        assert record["checkpoint_sha256"] == endpoint["checkpoint"]["sha256"]
    assert read(FIT / "diagnostic_summary.json")["all_finite"]
    assert (FIT / "tests_recovery_exit.txt").read_text().strip() == "0"
    assert "116 passed" in (FIT / "tests_recovery.log").read_text()
    assert (FIT / "plot_recovery_exit.txt").read_text().strip() == "0"
    recovery = {}
    for mode in ("audit", "plot", "diagnostics", "tests"):
        before = read(FIT / f"{mode}_recovery_before.json")
        for path, digest in before["before"].items():
            assert sha(path) == digest, path
        assert (FIT / f"{mode}_recovery_exit.txt").read_text().strip() == "0"
        recovery[mode] = {
            "before_sha256": sha(FIT / f"{mode}_recovery_before.json"),
            "completed_exit": 0,
        }
        if mode in ("audit", "plot"):
            original_exit = int((FIT / f"{mode}_exit.txt").read_text())
            assert original_exit != 0
            recovery[mode]["original_native_failure_exit"] = original_exit
    deployment = read(FIT / "inference_memory.json")
    assert len(deployment["students"]) == 144 and len(deployment["full_teachers"]) == 18
    assert deployment["posthoc"] and not deployment["gradients"]
    (FIT / "postprocess_recovery.json").write_text(
        json.dumps(
            {"completed": True, "stages": recovery, "retrained": False, "root_cause_proved": False},
            indent=2,
        )
        + "\n"
    )
    packed = {}
    for folder, names in (
        (CAPACITY, ("result.json", "study.log", "audit.log")),
        (
            FIT,
            (
                "result.json",
                "audit.json",
                "study.log",
                "audit.log",
                "plot.log",
                "audit_recovery.log",
                "plot_recovery.log",
                "tests_recovery.log",
            ),
        ),
    ):
        for name in names:
            original = folder / name
            destination = original.with_name(original.name + ".gz")
            destination.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
            assert gzip.decompress(destination.read_bytes()) == original.read_bytes()
            packed[str(destination)] = {
                "sha256": sha(destination),
                "original_sha256": sha(original),
                "bytes": destination.stat().st_size,
            }
    reports = [
        Path("research/affine_residual_capacity_results.md"),
        Path("research/affine_residual_fit_results.md"),
    ]
    for report in reports:
        for link in re.findall(r"\]\(([^)]+)\)", report.read_text()):
            if not link.startswith("https:"):
                target = (report.parent / link).resolve()
                pending_receipt = Path(
                    "results/verification/affine_residual_final_v1.json"
                ).resolve()
                assert target == pending_receipt or target.exists(), (report, link)
    public = [
        *reports,
        Path("research/figures/affine_residual.png"),
        FIT / "summary.json",
        FIT / "diagnostics.json.gz",
        FIT / "diagnostic_summary.json",
        FIT / "inference_memory.json",
        FIT / "postprocess_recovery.json",
        FIT / "metrics.csv.gz",
        CAPACITY / "audit.json",
        *sorted((FIT / "source").glob("*.py")),
        *sorted((CAPACITY / "source").glob("*.py")),
    ]
    for path in public:
        assert path.stat().st_size < 5 * 2**20, path
    receipt = {
        "status": "PASS",
        "frozen_sources_verified": len(frozen),
        "capacity_checks": 126,
        "qualification_groups": {"H106": 8, "H107": 10},
        "exact_recaptured_datasets": 18,
        "independent_scores": 864,
        "audited_exports_and_selections": 144,
        "ridge_initializations_checked": 144,
        "neural_updates": 172800,
        "resource_profile_updates": 540,
        "max_score_difference": audit["max_score_difference"],
        "pytest_passed": 116,
        "both_fixed_recipes_rejected": True,
        "broad_goal_achieved": False,
        "active_model_structure_unchanged": True,
        "original_postprocessing_failures_preserved": True,
        "postprocessing_recovered_without_training_repeat": True,
        "packed": packed,
        "files": {str(p): sha(p) for p in public},
    }
    destination = Path("results/verification/affine_residual_final_v1.json")
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("files", "packed")}, indent=2))


if __name__ == "__main__":
    run()
