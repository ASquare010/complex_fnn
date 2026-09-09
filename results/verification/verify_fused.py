"""Check retained fused execution evidence; no GPU execution or training."""

import ast
import hashlib
import json
import re
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.core.reproducibility import provenance, sha256, write_json


class RenameOutput(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "O":
            node.id = "OUTPUT_WIDTH"
        return node

    def visit_arg(self, node):
        if node.arg == "O":
            node.arg = "OUTPUT_WIDTH"
        return self.generic_visit(node)


def main():
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance(),
        "verification_script_sha256": sha256(Path(__file__)),
        "archives": {},
    }
    for name in ("fused_kernel_v1", "fused_probe_v1", "fused_serving_scale384_v1"):
        folder = Path("results") / name
        protocol = json.loads((folder / "protocol.json").read_text())
        with zipfile.ZipFile(folder / "source.zip") as archive:
            assert all(
                hashlib.sha256(archive.read(k)).hexdigest() == v
                for k, v in protocol["provenance"]["source_files"].items()
            )
            plan_hash = hashlib.sha256(archive.read("research/fused_execution_plan.md")).hexdigest()
            if "plan_sha256" in protocol:
                assert plan_hash == protocol["plan_sha256"]
        result["archives"][name] = {
            "all_source_hashes_match": True,
            "source_zip_sha256": sha256(folder / "source.zip"),
            "archived_plan_sha256": plan_hash,
        }
    path = Path("results/fused_serving_scale384_v1/result.json")
    serving = json.loads(path.read_text())
    assert all(serving["retention_gates"].values())
    for label, worker in serving["workers"].items():
        assert (
            worker["validation_tokens"] == 32768
            and worker["numerical_checks_pass"]
            and worker["parameter_objects_unchanged"]
        )
        assert worker["persistent_extra_cache_bytes"] == 0
        assert abs(worker["compiled_nll"] - worker["native_eager_nll"]) <= 0.001
        run = Path("results/runs") / worker["run"]
        assert sha256(run / "checkpoint.pt") == worker["checkpoint_sha256"]
    for reference in ("full_reference", "native_candidate"):
        samples = [
            r["seconds_per_forward"][reference] / r["seconds_per_forward"]["fused_candidate"]
            for r in serving["paired_rounds"]
        ]
        assert samples == serving["fused_throughput_ratio"][reference]["samples"]
        assert statistics.median(samples) == serving["fused_throughput_ratio"][reference]["median"]
    result["serving_gates"] = serving["retention_gates"]
    result["serving_sha256"] = sha256(path)
    with zipfile.ZipFile("results/fused_serving_scale384_v1/source.zip") as archive:
        old = ast.parse(archive.read("src/blockshuffle_ffn/triton_inference.py").decode())
    current = ast.parse(Path("src/blockshuffle_ffn/triton_inference.py").read_text())
    assert ast.dump(RenameOutput().visit(old), include_attributes=False) == ast.dump(
        current, include_attributes=False
    )
    result["current_kernel_ast_identical_after_argument_rename"] = True
    profile = json.loads(
        Path("results/profiles/scale384_blockshuffle_fused_compiled.json").read_text()
    )
    counts = {
        key: sum(r["calls"] for r in profile["rows"] if r["operator"] == key)
        / profile["profiled_forwards"]
        for key in ("_paired_first", "_paired_second", "_down_factor")
    }
    assert counts == {"_paired_first": 8, "_paired_second": 8, "_down_factor": 16}
    assert not any(r["operator"] == "aten::bmm" for r in profile["rows"])
    result["observed_kernel_counts"] = counts
    tests = json.loads(Path("results/verification/fused_test_run_v1.json").read_text())
    assert tests["returncode"] == 0 and "123 passed" in tests["stdout"]
    result["test_run"] = tests
    result["checks"] = {}
    for command in (
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        ["git", "diff", "--check"],
    ):
        r = subprocess.run(command, capture_output=True, text=True)
        assert r.returncode == 0, (command, r.stdout, r.stderr)
        result["checks"][" ".join(command)] = {
            "returncode": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    docs = [
        Path("README.md"),
        Path("research/CURRENT_STATE.md"),
        Path("research/fused_execution_plan.md"),
        Path("research/fused_execution_results.md"),
        Path("src/blockshuffle_ffn/README.md"),
    ]
    links = 0
    for p in docs:
        for ref in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if "://" in ref or ref.startswith("#"):
                continue
            target = (p.parent / ref.split("#")[0]).resolve()
            assert (
                target.exists()
                or target == Path("results/verification/fused_checks_v1.json").resolve()
            ), (p, ref)
            links += 1
    result["checked_local_links"] = links
    result["visually_inspected_figure"] = "results/plots/fused_serving_scale384.png"
    result["interpretation"] = (
        "Verified single-checkpoint serving result. Larger-model replication is a separate active cohort; this record does not claim its completion."
    )
    write_json(Path("results/verification/fused_checks_v1.json"), result)
    print(
        json.dumps(
            {
                "archives": len(result["archives"]),
                "tests": "123 passed",
                "kernel_counts": counts,
                "serving_gates": result["serving_gates"],
                "local_links": links,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
