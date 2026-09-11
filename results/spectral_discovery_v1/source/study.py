"""H103: independent datasets and falsifying controls, with zero training updates."""

import gc
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from results.nonlinear_residual_v1.source.model import TASKS, target_features
from results.spectral_discovery_v1.source.moments import (
    centered_moment,
    leading_subspace,
    weights_from_labels,
)
from src.core.reproducibility import environment, provenance, sha256, write_json

ROOT = Path("results/spectral_discovery_v1")
PLAN = Path("research/spectral_discovery_plan.md")


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(
        str((value.dtype, tuple(value.shape))).encode() + value.numpy().tobytes()
    ).hexdigest()


def generator(seed):
    return torch.Generator().manual_seed(seed)


def qualify():
    checks = []
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    z = np.sqrt(2) * nodes
    for degree, values in ((2, (z * z - 1) / math.sqrt(2)), (3, (z**3 - 3 * z) / math.sqrt(6))):
        moment = float(np.dot(weights, values**2 * (z * z - 1)) / math.sqrt(math.pi))
        assert abs(moment - 2 * degree) < 1e-10
        checks.append({"name": f"Hermite_{degree}", "value": moment, "expected": 2 * degree})
    x = torch.randn(37, 8, dtype=torch.float64, generator=generator(5))
    y = torch.randn(37, 3, dtype=torch.float64, generator=generator(6))
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=generator(7))).Q
    for mode in ("raw", "bounded"):
        w = weights_from_labels(y, mode)
        expected = x.T @ (w[:, None] * x) / len(x)
        for device in ("cpu", "cuda"):
            actual = centered_moment(x, w, 11, device)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            checks.append(
                {
                    "name": f"stream_{mode}_{device}",
                    "max_error": (actual - expected).abs().max().item(),
                }
            )
        torch.testing.assert_close(
            weights_from_labels(3 * y @ rotation, mode), w, rtol=1e-10, atol=1e-10
        )
        checks.append({"name": f"rotation_scale_{mode}", "passed": True})
        constant = torch.eye(37, dtype=torch.float64)
        assert torch.count_nonzero(weights_from_labels(constant, mode)) == 0
        checks.append({"name": f"one_hot_zero_{mode}", "passed": True})
    basis, _ = leading_subspace(expected, 3)
    torch.testing.assert_close(
        basis.T @ basis, torch.eye(3, dtype=torch.float64), rtol=1e-10, atol=1e-10
    )
    checks.append({"name": "orthonormal_basis", "passed": True})
    for operation in (
        lambda: weights_from_labels(torch.zeros(4, 2), "raw"),
        lambda: leading_subspace(torch.eye(3), 4),
        lambda: centered_moment(x, w, 0),
    ):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Missing validation")
    checks.append({"name": "invalid_inputs", "passed": True})
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})


def alignment(estimated, teacher):
    singular = torch.linalg.svdvals(teacher.T @ estimated)
    return {
        "recovery": singular.square().mean().item(),
        "smallest_squared_cosine": singular.square().min().item(),
        "squared_cosines": singular.square().tolist(),
    }


def run():
    if (ROOT / "protocol.json").exists():
        raise FileExistsError("Frozen experiment exists; use a new directory")
    sources = {p.as_posix(): sha256(p) for p in (ROOT / "source").glob("*.py")}
    sources[PLAN.as_posix()] = sha256(PLAN)
    sources["results/nonlinear_residual_v1/source/model.py"] = sha256(
        Path("results/nonlinear_residual_v1/source/model.py")
    )
    write_json(
        ROOT / "protocol.json",
        {
            "sources": sources,
            "environment": environment(),
            "provenance": provenance(),
            "seeds": [17, 29, 43],
            "n": 65536,
            "width": 384,
            "rank": 32,
            "teacher_rank": 16,
            "neural_updates": 0,
            "plan_sha256": sha256(PLAN),
        },
    )
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(p, p)
    qualify()
    records = []
    start_all = time.perf_counter()
    for seed in (17, 29, 43):
        x = torch.randn(65536, 384, generator=generator(10300 + seed))
        teacher = torch.linalg.qr(
            torch.randn(384, 384, dtype=torch.float64, generator=generator(10400 + seed))
        ).Q[:, :16]
        rotation = torch.linalg.qr(
            torch.randn(384, 384, dtype=torch.float64, generator=generator(10500 + seed))
        ).Q[:16]
        q = x.double() @ teacher
        random_basis = torch.linalg.qr(
            torch.randn(384, 32, dtype=torch.float64, generator=generator(10600 + seed))
        ).Q
        pca, _ = leading_subspace(x.double().T @ x.double() / len(x), 32)
        for task in TASKS:
            raw = target_features(q, task) @ rotation
            y = (raw / raw.std(0, correction=0)).float()
            del raw
            weights = {m: weights_from_labels(y, m) for m in ("raw", "bounded")}
            weights["permuted"] = weights["bounded"][
                torch.randperm(len(x), generator=generator(10700 + seed))
            ]
            for mode in ("raw", "bounded", "permuted"):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                matrix = centered_moment(x, weights[mode])
                basis, eigenvalues = leading_subspace(matrix, 32)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                row = {
                    "seed": seed,
                    "task": task,
                    "mode": mode,
                    **alignment(basis, teacher),
                    "seconds_moment_and_eigensolve": seconds,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "matrix_sha256": tensor_sha(matrix),
                    "basis_sha256": tensor_sha(basis),
                    "eigenvalues": eigenvalues.tolist(),
                    "x_sha256": tensor_sha(x),
                    "y_sha256": tensor_sha(y),
                    "random": alignment(random_basis, teacher),
                    "pca": alignment(pca, teacher),
                }
                folder = ROOT / "cases" / f"{task}_{mode}_s{seed}"
                folder.mkdir(parents=True)
                torch.save(
                    {"matrix": matrix, "basis": basis, "eigenvalues": eigenvalues},
                    folder / "moments.pt",
                )
                write_json(folder / "metrics.json", row)
                records.append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in ("seed", "task", "mode", "recovery", "smallest_squared_cosine")
                        }
                    ),
                    flush=True,
                )
            del y, weights
        del x, q, teacher, rotation
    decisions = {}
    for mode in ("raw", "bounded"):
        gates = []
        for seed in (17, 29, 43):
            row = next(
                r for r in records if (r["seed"], r["task"], r["mode"]) == (seed, "cubic", mode)
            )
            null = next(
                r
                for r in records
                if (r["seed"], r["task"], r["mode"]) == (seed, "cubic", "permuted")
            )
            gates.append(
                {
                    "seed": seed,
                    "recovery": row["recovery"] >= 0.30,
                    "null_gap": row["recovery"] - null["recovery"] >= 0.15,
                    "minimum_direction": row["smallest_squared_cosine"] >= 0.01,
                    "null_control": null["recovery"] <= 0.15,
                }
            )
        decisions[mode] = {
            "earns_fitting": all(all(v for k, v in g.items() if k != "seed") for g in gates),
            "gates": gates,
        }
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "records": records,
            "decisions": decisions,
            "elapsed_seconds": time.perf_counter() - start_all,
            "neural_updates": 0,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
        raise
