"""Freeze once, run 25 checks once, and independently inspect saved tensor pairs."""

import hashlib
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from src.core.reproducibility import environment, provenance

ROOT = Path("results/input_basis_lift_v1")


def audit():
    before = read(ROOT / "before.json")
    assert all(sha(p) == h for p, h in before["source_hashes"].items())
    assert all(sha(p) == h for p, h in before["anchors"].items())
    observations = list((ROOT / "qualification").glob("*.json"))
    assert len(observations) == 25
    pairs, exact, tensors = 0, 0, 0

    def compare(a, b, tol):
        nonlocal pairs, exact
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)
            pairs += 1
            exact += int(tol == 0)
        elif isinstance(a, list):
            assert len(a) == len(b)
            for aa, bb in zip(a, b):
                compare(aa, bb, tol)

    for path in observations:
        row = read(path)
        assert row["status"] == "PASS" and row["optimizer_updates"] == 0
        if "tensor_file" not in row:
            continue
        tensor_path = path.with_name(row["tensor_file"])
        assert sha(tensor_path) == row["tensor_sha256"]
        with zipfile.ZipFile(tensor_path) as archive:
            assert archive.testzip() is None
        data = torch.load(tensor_path, map_location="cpu", weights_only=True)
        tensors += 1
        if path.stem.startswith("formula"):
            compare(data["actual"], data["reference"], 1e-11)
        elif path.stem.startswith(("collapse", "bounds")):
            for item in data.values():
                compare(item["actual"], item["reference"], 1e-11)
                if "bounds" in item:
                    for grad, bound in zip(item["actual"], item["bounds"]):
                        assert grad.abs().max() <= bound + 1e-11
        elif path.stem.startswith("initial"):
            for item in data["actual"].values():
                compare(item, data["reference"], 1e-11)
        elif path.stem.startswith("cuda"):
            for precision, tol in (("fp32", 1e-5), ("bf16", 0.02)):
                item = data[precision]
                compare(item["native"], item["reference"], tol)
                compare(item["native"], item["checkpoint"], 0)
        else:
            assert path.stem == "kernel_witness"
            compare(data["u"] @ data["v"], torch.zeros(304, dtype=torch.float64), 1e-10)
            compare(data["feature_a"], data["feature_b"], 1e-10)
            compare(data["direct_delta"][:384], data["v"], 1e-10)
    assert tensors == 17
    write_json(
        ROOT / "result.json",
        {
            "status": "PASS",
            "checks": 25,
            "tensor_files": tensors,
            "independently_compared_tensor_pairs": pairs,
            "checkpoint_exact_pairs": exact,
            "optimizer_updates": 0,
            "training_allocated": False,
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "observations": {str(p): sha(p) for p in observations},
        },
    )


def main():
    assert not (ROOT / "before.json").exists()
    sources = dict(read("research/evidence/release_v1.json")["sources"])
    for path in [
        *sorted((ROOT / "source").glob("*.py")),
        Path("research/input_basis_lift_plan.md"),
        Path("results/blast_operator_recovery_v1/source/storage.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("doc/RESEARCH_GOAL.md"),
        Path("doc/RESEARCH_RULES.md"),
    ]:
        sources[path.as_posix()] = sha(path)
    assert all(sha(p) == h for p, h in sources.items())
    anchors = {
        p: sha(p)
        for p in (
            "research/evidence/neuron_geometry_release.json",
            "results/neuron_geometry_recovery_v1/result.json",
            "results/neuron_geometry_audit_v1/result.json",
        )
    }
    write_json(
        ROOT / "before.json",
        {
            "source_hashes": sources,
            "anchors": anchors,
            "checks": 25,
            "optimizer_updates": 0,
            "repetitions": 1,
        },
    )
    write_json(
        ROOT / "protocol.json",
        {
            **read(ROOT / "before.json"),
            "environment": environment(),
            "provenance": provenance(),
            "before_sha256": sha(ROOT / "before.json"),
        },
    )
    with (ROOT / "source.zip").open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in {**sources, **anchors}:
                archive.write(p, p)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        assert archive.testzip() is None
        assert all(
            hashlib.sha256(archive.read(p)).hexdigest() == h
            for p, h in {**sources, **anchors}.items()
        )
    (ROOT / "qualification").mkdir()
    status = {
        "status": "RUNNING",
        "coordinator_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "coordinator_status.json", status)
    started = time.perf_counter()
    try:
        tests = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:anyio",
            "-q",
            "-x",
            "--tb=short",
            str(ROOT / "source/test_qualification.py"),
        ]
        for phase, command in (
            ("collection", [*tests, "--collect-only"]),
            ("qualification", tests),
        ):
            log = ROOT / f"{phase}.log"
            with log.open("xb") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                process = {
                    "status": "RUNNING",
                    "pid": child.pid,
                    "command": command,
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                write_json(ROOT / f"{phase}_process.json", process)
                code = child.wait()
                stream.flush()
                os.fsync(stream.fileno())
            process.update(
                status="PASS" if code == 0 else "FAIL",
                returncode=code,
                finished_utc=datetime.now(timezone.utc).isoformat(),
                log_sha256=sha(log),
            )
            write_json(ROOT / f"{phase}_process.json", process, exclusive=False)
            assert code == 0, log.read_text()[-10000:]
            assert (
                "25 tests collected" if phase == "collection" else "25 passed"
            ) in log.read_text()
        audit()
        status["status"] = "PASS"
    except BaseException as exc:
        status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        status.update(
            elapsed_seconds=time.perf_counter() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(ROOT / "coordinator_status.json", status, exclusive=False)


if __name__ == "__main__":
    main()
