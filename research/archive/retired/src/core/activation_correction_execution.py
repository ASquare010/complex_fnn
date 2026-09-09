"""Bounded correction-only execution audit with archived native memory controls."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from src.core.reproducibility import environment, provenance, sha256, write_json

PLAN = Path("research/activation_correction_execution_plan.md")
CASE = "rational_correction_compiled"


def profile_loop(source):
    tree = ast.parse(source)
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step"
    ]
    assert len(loops) == 1
    return ast.dump(loops[0], include_attributes=False)


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    old = Path("results/activation_execution_v1")
    old_result = json.loads((old / "result.json").read_text())
    reused = {}
    reporting_differences = {}
    with zipfile.ZipFile(old / "source.zip") as archive:
        old_protocol = json.loads((old / "protocol.json").read_text())
        assert all(
            hashlib.sha256(archive.read(n)).hexdigest() == h
            for n, h in old_protocol["provenance"]["source_files"].items()
        )
        previous = archive.read("src/core/activation_execution.py")
        assert profile_loop(previous) == profile_loop(
            Path("src/core/activation_execution.py").read_bytes()
        )
        for name in (
            "src/core/optimization.py",
            "src/core/data.py",
            "src/core/benchmark.py",
            "src/blockshuffle_ffn/__init__.py",
            "src/dense_ffn/__init__.py",
        ):
            assert archive.read(name) == Path(name).read_bytes()
        for case in ("full_gelu", "full_swiglu", "rational_native", "rational_compiled"):
            path = old / case / "result.json"
            assert sha256(path) == old_result["result_hashes"][case]
            row = json.loads(path.read_text())
            differences = [
                n
                for n, h in row["provenance"]["source_files"].items()
                if hashlib.sha256(archive.read(n)).hexdigest() != h
            ]
            # This report-only edit was made during H036; computation sources match.
            assert set(differences) <= {"src/core/activation_report.py"}
            reporting_differences[case] = differences
            assert row["environment"] == environment()
            assert row["checkpoint_sha256"] == sha256(
                Path("results/runs") / row["run"] / "checkpoint.pt"
            )
            reused[case] = row
    protocol = {
        "plan_sha256": sha256(PLAN),
        "provenance": provenance(),
        "environment": environment(),
        "profile_loop_ast_unchanged": True,
        "archived_worker_reporting_only_differences": reporting_differences,
        "reused_result_hashes": old_result["result_hashes"],
        "worker_timeout_seconds": 900,
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    with (output / "worker.log").open("x", encoding="utf-8") as log:
        run = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "src.core.activation_execution",
                "worker",
                "--case",
                CASE,
                "--output",
                str(output / CASE),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    worker = output / CASE
    if run.returncode:
        failure = (
            json.loads((worker / "failure.json").read_text())
            if (worker / "failure.json").exists()
            else {"error": "worker terminated without failure artifact"}
        )
        result = {
            "status": "failed",
            "returncode": run.returncode,
            "failure": failure,
            "earns_training_repeat": False,
            "protocol": protocol,
        }
    else:
        candidate = json.loads((worker / "result.json").read_text())
        assert all(candidate["minibatch_hashes"] == r["minibatch_hashes"] for r in reused.values())
        assert candidate["data_hashes"] == reused["rational_native"]["data_hashes"]
        assert candidate["checkpoint_sha256"] == reused["rational_native"]["checkpoint_sha256"]
        gates = {
            f"memory_within_ten_percent_{key}": candidate["peak_allocated_vram_bytes"]
            <= 1.1 * reused[key]["peak_allocated_vram_bytes"]
            for key in ("full_gelu", "full_swiglu")
        }
        gates["strict_gradient_fidelity"] = (
            candidate["fidelity"]["all_gradients_relative_l2"] <= 0.0001
        )
        result = {
            "status": "complete",
            "candidate": candidate,
            "gates": gates,
            "earns_training_repeat": all(gates.values()),
            "protocol": protocol,
            "candidate_sha256": sha256(worker / "result.json"),
        }
    write_json(output / "result.json", result)
    write_json(
        output / "progress.json",
        {"status": result["status"], "earns_training_repeat": result["earns_training_repeat"]},
    )
    report(output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("protocol", "candidate", "failure")}
        )
    )


def report(output):
    result = json.loads((output / "result.json").read_text())
    row = result.get("candidate", result.get("failure", {}))
    fidelity = row.get("fidelity", {})
    lines = [
        "# Rational correction-only compiler isolation",
        "",
        f"**Status: {result['status']}; earns a training repeat: {result['earns_training_repeat']}.** This diagnostic retains native SiLU and gate multiplication, compiling only the rational residual. It does not modify the previously failed full-product backend or its training result.",
        "",
        "The [frozen protocol](activation_correction_execution_plan.md) requires all-parameter gradient relative L2<=0.0001, stricter than the original local audit. Native checkpoints, full-validation targets, fidelity batch, optimizer state and transient profiling-loop AST are held fixed. Source archives, checkpoint hashes, environment and reused memory controls are verified before execution.",
        "",
        "| Local quantity | Correction-only | Previous full product |",
        "|---|---:|---:|",
    ]
    prior = json.loads(
        Path("results/activation_execution_v1/rational_compiled/result.json").read_text()
    )
    for key, value in fidelity.items():
        lines.append(f"| {key} | {value} | {prior['fidelity'].get(key, 'n/a')} |")
    if result["status"] == "failed":
        lines += [
            "",
            f"Worker stopped at {row.get('error', 'unknown failure')}. No later unmeasured memory or training-quality claim is made. [Retained worker log](../results/activation_correction_execution_v1/worker.log).",
        ]
    else:
        lines += [
            "",
            f"Allocated training peak: {row['peak_allocated_vram_bytes'] / 2**20:.3f} MiB. Full-validation NLL: {row['compiled_nll']:.9f}, native {row['native_nll']:.9f}. Frozen gates: {result['gates']}.",
        ]
    lines += [
        "",
        "This isolates a different compilation boundary. A smaller derivative error suggests numerical improvement only on the measured batch; it does not prove the cause of the earlier 200-step drift or guarantee a matching training trajectory. Transient optimizer updates, if reached, are discarded and are not a trained model result.",
        "",
        "[Raw audit](../results/activation_correction_execution_v1/result.json). All execution artifacts and the original checkpoint remain retained.",
    ]
    Path("research/activation_correction_execution_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
