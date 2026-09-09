"""Verify the completed larger-model cohort and disjoint tail evidence."""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.core.config import ModelConfig
from src.core.reproducibility import provenance, sha256, write_json
from src.core.scale_replication_report import load
from src.core.validation_tail_report import report as tail_report


def main():
    torch.set_num_threads(4)
    records = load(Path("results"))
    progress = json.loads(Path("results/scale_replication_v1/progress.json").read_text())
    assert progress["status"] == "complete" and len(progress["completed"]) == 8
    protocol = json.loads(Path("results/scale_replication_v1/protocol.json").read_text())
    critical = [
        k for k in protocol["preflight"]["training_source_checks"] if k != "src/core/benchmark.py"
    ]
    with zipfile.ZipFile("results/runs/scale384_blockshuffle_s17_800/source.zip") as z:
        old_sources = {name: z.read(name) for name in critical}
        old_tree = ast.parse(z.read("src/core/benchmark.py").decode())
    functions = {
        n.name: ast.dump(n, include_attributes=False)
        for n in old_tree.body
        if isinstance(n, ast.FunctionDef)
    }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance(),
        "verification_script_sha256": sha256(Path(__file__)),
        "training_runs": {},
    }
    for group in records.values():
        for key, m in group.items():
            folder = Path("results/runs") / m["run"]
            cp = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
            cfg = ModelConfig(**cp["model_config"])
            assert (
                sum(t.numel() for t in cp["model"].values())
                == cfg.total_parameters
                == m["total_parameters"]
            )
            assert cfg.ffn_parameters * cfg.layers == m["ffn_parameters"]
            assert all(torch.isfinite(t).all() for t in cp["model"].values())
            with zipfile.ZipFile(folder / "source.zip") as z:
                assert all(
                    hashlib.sha256(z.read(name)).hexdigest() == digest
                    for name, digest in m["provenance"]["source_files"].items()
                )
                assert all(z.read(name) == data for name, data in old_sources.items())
                tree = ast.parse(z.read("src/core/benchmark.py").decode())
                actual = {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef)
                }
                assert all(actual[name] == body for name, body in functions.items())
            result["training_runs"][m["run"]] = {
                "checkpoint_sha256": sha256(folder / "checkpoint.pt"),
                "actual_parameter_counts_match": True,
                "all_weights_finite": True,
                "all_archived_source_hashes_match": True,
                "critical_training_sources_unchanged": True,
            }
    assert len(result["training_runs"]) == 12
    result["cohort_archives"] = {}
    for name, plan in (
        ("scale_replication_v1", "scale_replication_plan.md"),
        ("validation_tail_scale384_v1", "validation_tail_plan.md"),
    ):
        folder = Path("results") / name
        p = json.loads((folder / "protocol.json").read_text())
        with zipfile.ZipFile(folder / "source.zip") as z:
            assert all(
                hashlib.sha256(z.read(k)).hexdigest() == v
                for k, v in p["provenance"]["source_files"].items()
            )
            assert hashlib.sha256(z.read("research/" + plan)).hexdigest() == p["plan_sha256"]
        result["cohort_archives"][name] = {
            "source_and_plan_hashes_match": True,
            "archive_sha256": sha256(folder / "source.zip"),
        }
    tail = tail_report()
    assert tail["all_target_accounting_checks_pass"] and tail["all_prefix_scores_reproduce_exactly"]
    result["tail_checks"] = {
        k: tail[k]
        for k in (
            "all_target_accounting_checks_pass",
            "all_prefix_scores_reproduce_exactly",
            "all_tail_quality_gates_pass",
        )
    }
    tests = json.loads(Path("results/verification/replication_test_run_v1.json").read_text())
    assert tests["returncode"] == 0 and "125 passed" in tests["stdout"]
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
    links = 0
    for p in (
        Path("README.md"),
        *[
            Path("research") / n
            for n in (
                "CURRENT_STATE.md",
                "scale_replication_results.md",
                "validation_tail_results.md",
                "current_comparator_audit.md",
            )
        ],
    ):
        for ref in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if "://" in ref or ref.startswith("#"):
                continue
            target = (p.parent / ref.split("#")[0]).resolve()
            assert (
                target.exists()
                or target == Path("results/verification/replication_checks_v1.json").resolve()
            ), (p, ref)
            links += 1
    result["checked_local_doc_links"] = links
    result["visually_inspected_figures"] = [
        "results/plots/scale384_replication.png",
        "results/plots/validation_tail_scale384.png",
    ]
    result["interpretation"] = (
        "Completed locked three-seed and additional-tail evidence. This verification does not establish convergence, second-corpus generalization, equal tuning or novelty. Full-suite source provenance precedes presentation/report-only additions."
    )
    write_json(Path("results/verification/replication_checks_v1.json"), result)
    print(
        json.dumps(
            {
                "training_runs": 12,
                "new_training_runs": 8,
                "tail_checkpoints": 12,
                "tests": "125 passed",
                "local_links": links,
                "tail_checks": result["tail_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
