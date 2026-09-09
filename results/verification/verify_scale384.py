"""Recheck retained scale, compiler and conditioning artifacts without training."""

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.core.conditioning import floor_square_blocks
from src.core.config import ModelConfig
from src.core.reproducibility import environment, provenance, sha256, write_json


def archive_checks(folder, prov):
    with zipfile.ZipFile(folder / "source.zip") as archive:
        checked = all(
            hashlib.sha256(archive.read(name)).hexdigest() == digest
            for name, digest in prov["source_files"].items()
        )
    assert checked
    return {
        "source_files": len(prov["source_files"]),
        "all_archived_source_hashes_match": checked,
        "archive_sha256": sha256(folder / "source.zip"),
    }


def main():
    torch.set_num_threads(4)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance(),
        "environment": environment(),
        "optional_triton_windows": importlib.metadata.version("triton-windows"),
        "training_runs": {},
        "conditioning": {},
    }
    for folder in sorted(Path("results/runs").glob("scale384_*_s17_*")):
        m = json.loads((folder / "metrics.json").read_text())
        cp = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        cfg = ModelConfig(**cp["model_config"])
        checks = {
            "model_config_matches": cp["model_config"] == m["model"],
            "checkpoint_tensor_count_matches": sum(v.numel() for v in cp["model"].values())
            == m["total_parameters"]
            == cfg.total_parameters,
            "ffn_count_matches": cfg.ffn_parameters * cfg.layers == m["ffn_parameters"],
            "all_model_tensors_finite": all(
                bool(torch.isfinite(v).all()) for v in cp["model"].values()
            ),
            "correct_validation_targets": m["validation_tokens"] == 32768,
            "correct_training_tokens": m["training_tokens"] == m["training"]["steps"] * 16 * 128,
        }
        for phase in ("initial", "final"):
            d = json.loads((folder / f"{phase}_diagnostics.json").read_text())
            checks[f"{phase}_diagnostics_finite"] = all(v["finite"] for v in d.values())
        assert all(checks.values()), (folder, checks)
        result["training_runs"][folder.name] = {
            "checks": checks,
            "checkpoint_sha256": sha256(folder / "checkpoint.pt"),
            "metrics_sha256": sha256(folder / "metrics.json"),
            "source_archive": archive_checks(folder, m["provenance"]),
        }
    assert len(result["training_runs"]) == 8
    for folder in sorted(Path("results").glob("conditioning_*_v1")):
        data = json.loads((folder / "result.json").read_text())
        assert all(data["integrity_checks"].values())
        transformed = torch.load(folder / "transforms.pt", map_location="cpu", weights_only=True)
        original = Path("results/runs") / data["protocol"]["run"] / "checkpoint.pt"
        assert sha256(original) == data["protocol"]["checkpoint_sha256"]
        state = torch.load(original, map_location="cpu", weights_only=True)["model"]
        for name, case in data["cases"].items():
            assert (
                case["finite"]
                and case["other_weights_bitwise_unchanged"]
                and case["validation_tokens"] == 32768
            )
            for key, value in transformed[name].items():
                assert (
                    hashlib.sha256(value.numpy().tobytes()).hexdigest()
                    == case["transformed_weights_sha256"][key]
                )
                if name.startswith("floor_"):
                    assert torch.equal(value, floor_square_blocks(state[key], case["alpha"]))
            if name.startswith("floor_"):
                noise = data["cases"][name.replace("floor_", "random_")]
                for key, v in case["changes"].items():
                    for a, b in zip(
                        v["per_block_change_norm"],
                        noise["changes"][key]["per_block_change_norm"],
                        strict=True,
                    ):
                        assert abs(a - b) <= 1e-6 + 1e-5 * abs(a)
        ac = archive_checks(folder, data["protocol"]["provenance"])
        with zipfile.ZipFile(folder / "source.zip") as archive:
            assert (
                hashlib.sha256(archive.read("research/conditioning_plan.md")).hexdigest()
                == data["protocol"]["plan_sha256"]
            )
            if data["protocol"].get("replication"):
                assert (
                    hashlib.sha256(
                        archive.read("research/conditioning_replication_plan.md")
                    ).hexdigest()
                    == data["protocol"]["replication_plan_sha256"]
                )
        result["conditioning"][folder.name] = {
            "case_count": len(data["cases"]),
            "integrity_checks": data["integrity_checks"],
            "transformed_tensor_hashes_match": True,
            "floor_tensors_reproduce_bitwise": True,
            "noise_perturbation_norms_match": True,
            "original_checkpoint_unchanged": True,
            "plan_archive_hashes_match": True,
            "source_archive": ac,
            "result_sha256": sha256(folder / "result.json"),
            "transforms_sha256": sha256(folder / "transforms.pt"),
        }
    assert sum(v["case_count"] for v in result["conditioning"].values()) == 20
    compiled = json.loads(Path("results/compile_scale384_default_v2/result.json").read_text())
    result["compiler"] = {
        "retention_gates": compiled["retention_gates"],
        "native_first_failure_retained": Path(
            "results/compile_scale384_default_v1/failure.json"
        ).exists(),
        "source_archive": archive_checks(
            Path("results/compile_scale384_default_v2"), compiled["protocol"]["provenance"]
        ),
    }
    assert result["compiler"]["retention_gates"] == {
        "numerical_checks": True,
        "within_1_percent_compiled_full_nll": True,
        "median_relative_speed_at_least_point_eight": False,
    }
    result["serving"] = {}
    for p in Path("results").glob("serving_scale384_*.json"):
        d = json.loads(p.read_text())
        entry = {"sha256": sha256(p), "rows": len(d["rows"])}
        if "graph_v2" in p.name:
            assert d["memory_protocol"].startswith("shared_warmup_stream_v2")
            assert all(max(row["fresh_input_eager_max_abs_errors"]) == 0 for row in d["rows"])
            entry["graph_v2_memory_protocol_and_fresh_inputs_pass"] = True
        result["serving"][p.name] = entry
    assert len(result["serving"]) == 4
    result["test_run"] = json.loads(
        Path("results/verification/scale_conditioning_test_run.json").read_text()
    )
    assert result["test_run"]["returncode"] == 0 and "116 passed" in result["test_run"]["stdout"]
    result["checks"] = {}
    for command in (
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        ["git", "diff", "--check"],
    ):
        completed = subprocess.run(command, text=True, capture_output=True)
        result["checks"][" ".join(command)] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        assert completed.returncode == 0
    docs = [
        Path("README.md"),
        *[
            Path("research") / name
            for name in (
                "CURRENT_STATE.md",
                "trained_scale_results.md",
                "compiler_serving_results.md",
                "conditioning_results.md",
                "conditioning_plan.md",
                "conditioning_replication_plan.md",
            )
        ],
    ]
    checked_links = 0
    for p in docs:
        for ref in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if "://" in ref or ref.startswith("#"):
                continue
            target = (p.parent / ref.split("#")[0]).resolve()
            assert (
                target.exists()
                or target == Path("results/verification/scale384_checks.json").resolve()
            ), (p, ref)
            checked_links += 1
    result["verification_script_sha256"] = sha256(Path(__file__))
    result["checked_local_doc_links"] = checked_links
    result["retained_run_budgets"] = dict(
        Counter(
            json.loads(p.read_text())["training"]["steps"]
            for p in Path("results/runs").glob("*/metrics.json")
        )
    )
    result["standard_checkpoint_audit_count"] = len(list(Path("results/audits").glob("*.json")))
    result["visually_inspected_figures"] = [
        "results/plots/scale384_200.png",
        "results/plots/scale384_800.png",
        "results/plots/conditioning_transfer.png",
    ]
    result["interpretation"] = (
        "All listed integrity checks passed. The compiled speed gate intentionally fails; the overall research goal remains unmet. Test provenance precedes final report wording only. Original source archives and checkpoints are preserved."
    )
    write_json(Path("results/verification/scale384_checks.json"), result)
    print(
        json.dumps(
            {
                "scale_runs": len(result["training_runs"]),
                "conditioning_cases": 20,
                "tests": "116 passed",
                "local_doc_links": checked_links,
                "run_budgets": result["retained_run_budgets"],
                "checkpoint_audits": result["standard_checkpoint_audit_count"],
                "compiler_speed_gate": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
