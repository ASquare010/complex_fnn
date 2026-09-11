"""Verify public H103/H104 accounting against the completed independent audit.

This checks report consistency and immutable sources; it does not replace the
separate Torch reconstruction of data, moments, initializations and checkpoints.
"""

import csv
import gzip
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path("results/spectral_fitting_v1")


def read(path):
    raw = Path(path).read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def js(path):
    return json.loads(read(path))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    frozen = {}
    for root in (Path("results/spectral_discovery_v1"), ROOT):
        for path, digest in js(root / "protocol.json")["sources"].items():
            assert sha(path) == digest, path
            frozen[path] = digest
        assert js(root / "qualification.json")["passed"]
    discovery = js("results/spectral_discovery_v1/audit.json")
    assert discovery["passed"] and len(discovery["cases"]) == 36
    result, audit, summary = (js(ROOT / name) for name in ("result.json", "audit.json", "summary.json"))
    assert result["status"] == "COMPLETE" and result["runs"] == 408
    assert audit["passed"] and audit["checkpoint_count"] == 408 and audit["score_count"] == 816
    assert audit["rate_selections"] == 204
    checks = {r["label"]: r for r in audit["checks"]}
    rows = result["rows"]
    assert len(rows) == len(checks) == 408
    for row in rows:
        check = checks[row["label"]]
        assert check["initialization_exact"] and check["data_exact"]
        assert check["scores"] == [row["selection_mse"], row["reporting_mse"]]
        assert row["all_finite"] and row["training_presentations"] == 76800
        assert sha(Path(row["checkpoint"])) == row["checkpoint_sha256"]
    selected = [r for r in rows if r["selected"]]
    assert len(selected) == 204
    with gzip.open(ROOT / "metrics.csv.gz", "rt", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 408
    for exported, actual in zip(table, rows, strict=True):
        assert exported["label"] == actual["label"]
        assert exported["selected"] == str(actual["selected"])
        for key in ("reporting_mse", "selection_mse", "parameters", "median_update_ms"):
            assert float(exported[key]) == actual[key]
    for form, tasks in summary["per_task"].items():
        for task, record in tasks.items():
            values = [r["reporting_mse"] for r in selected if r["form"] == form and r["task"] == task]
            assert values == record["values"]
            assert statistics.mean(values) == record["mean"]
            assert statistics.median(values) == record["median"]
            assert statistics.variance(values) == record["variance"]
        peaks = [max(r["peak_allocated_bytes"], r["preprocessing"]["peak_cuda_bytes"])
                 for r in selected if r["form"] == form]
        assert summary["resources"][form]["pipeline_peak_allocated_bytes"]["mean"] == statistics.mean(peaks)
    candidate = summary["decisions"]["factor_cubic_bounded"]
    assert [k for k, v in candidate["gates"].items() if not v] == ["per_task_cap", "inference_time"]
    assert all(d["verdict"] == "REJECTED_AT_THIS_BUDGET" for d in summary["decisions"].values())
    assert summary["breakthrough"] is False
    packed_results = {}
    for folder in (Path("results/spectral_discovery_v1"), ROOT):
        original = folder / "result.json"
        packed = original.with_suffix(".json.gz")
        packed.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
        assert gzip.decompress(packed.read_bytes()) == original.read_bytes()
        packed_results[str(packed)] = {"sha256": sha(packed), "original_sha256": sha(original)}
    for reference in ("full_gelu_bounded", "full_swiglu_bounded", "tiny_cubic_bounded"):
        pairs = []
        for r in selected:
            if r["form"] == "factor_cubic_bounded":
                control = next(c for c in selected if (c["form"], c["task"], c["seed"])
                               == (reference, r["task"], r["seed"]))
                pairs.append(r["reporting_mse"] / control["reporting_mse"])
        measured = math.prod(pairs) ** (1 / len(pairs))
        assert abs(measured - candidate["ratios"][reference]) < 1e-12
    expected = {
        "tests_final": "113 passed", "triton_path_recovery": "8 passed",
        "plot_isolated": "Audited fitting figure written",
        "analysis_final": "REJECTED_AT_THIS_BUDGET",
    }
    for name, needle in expected.items():
        assert read(ROOT / f"{name}_exit.txt").strip() == "0"
        assert needle in read(ROOT / f"{name}.log")
    assert "6 failed, 107 passed" in read(ROOT / "tests_recovery.log")
    assert "access violation" in read(ROOT / "tests_original_failure.log")
    assert "access violation" in read(ROOT / "analysis_recovery.log")
    trace_names = ["tests_original_failure", "tests_recovery", "triton_path_recovery",
                   "tests_final", "analysis_recovery", "analysis_final", "plot_isolated"]
    traces = {}
    for name in trace_names:
        path = ROOT / f"{name}.log"
        packed = path.with_suffix(".log.gz")
        packed.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        assert gzip.decompress(packed.read_bytes()) == path.read_bytes()
        traces[name] = {"path": str(packed), "uncompressed_sha256": sha(path)}
    recovery = {
        "status": "PASS", "numerical_recipe_changes": False,
        "neural_training_retries": 0, "checkpoint_audit_retries": 0,
        "numeric_test_python": "UV-managed CPython 3.12.9; existing project site-packages",
        "initial_test_recovery": "107 passed; six existing Triton tests could not locate the bundled C compiler",
        "compiler_fix": "Set CC to .venv/Lib/site-packages/triton/runtime/tcc/tcc.exe",
        "triton_recheck": "8 passed", "complete_suite": "113 passed",
        "plot_recovery": "Separate Matplotlib process reads audited JSON without Torch; manual layout",
        "additional_shell_failure": "PowerShell 7 CLR startup access violation; later shell commands use Windows PowerShell 5.1",
        "frozen_sources_unchanged": frozen, "traces": traces,
    }
    (ROOT / "postprocess_recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
    report = Path("research/spectral_fitting_results.md")
    for target in re.findall(r"\]\(([^)]+)\)", read(report)):
        if not target.startswith("https:"):
            assert (report.parent / target).exists(), target
    assert Path("research/figures/spectral_fitting.png").read_bytes().startswith(b"\x89PNG")
    files = [report, Path("research/figures/spectral_fitting.png"), ROOT / "result.json",
             ROOT / "summary.json", ROOT / "audit.json", ROOT / "postprocess_recovery.json",
             Path("src/core/function_fitting.py"), Path("tests/test_function_fitting.py"),
             ROOT / "source/analyze.py", ROOT / "source/plot.py", ROOT / "source/audit.py"]
    receipt = {
        "status": "PASS", "frozen_source_files": len(frozen), "all_frozen_hashes_match": True,
        "moment_audits": 36, "checkpoint_audits": 408, "score_audits": 816,
        "selections": 204, "pytest_passed": 113, "report_links_exist": True,
        "mathematical_qualification_checks": {
            "H103": len(js("results/spectral_discovery_v1/qualification.json")["checks"]),
            "H104": len(js(ROOT / "qualification.json")["checks"]),
        },
        "pipeline_memory_accounted": True, "broad_goal_achieved": False,
        "lossless_results": packed_results,
        "files": {str(p): sha(p) for p in files},
    }
    Path("results/verification/spectral_final_v1.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    # Inspect prospective tracked size without staging or changing the index.
    listed = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    paths = {Path(p.decode()) for p in listed.split(b"\0") if p}
    print("Prospective public files MiB:", round(sum(p.stat().st_size for p in paths if p.is_file()) / 2**20, 3))


if __name__ == "__main__":
    run()
