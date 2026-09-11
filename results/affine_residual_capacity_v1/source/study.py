"""H106: capacity only, using rank-revealing least squares and residual spectra."""

import json
import math
import time
import zipfile
from pathlib import Path

import torch

from src.core.reproducibility import environment, provenance, sha256, write_json

ROOT = Path("results/affine_residual_capacity_v1")
PRIOR = Path("results/real_subspace_v1")
RANKS = (32, 64, 85, 128, 170, 192, 256)


def read(path):
    return json.loads(Path(path).read_text())


def augment(x):
    return torch.cat((x.double(), torch.ones(len(x), 1, dtype=torch.float64)), dim=1)


def fit(x, y):
    solution = torch.linalg.lstsq(x, y.double(), rcond=1e-12, driver="gelsd")
    residual = y.double() - x @ solution.solution
    return solution.solution, residual, int(solution.rank)


def tails(residual):
    covariance = residual.T @ residual / len(residual)
    values, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    return values.clamp_min(0), vectors


def qualify():
    torch.manual_seed(106)
    checks = []
    for deficient in (False, True):
        x = augment(torch.randn(73, 9, dtype=torch.float64))
        if deficient:
            x[:, 2] = x[:, 1]
        y = x @ torch.randn(10, 11, dtype=torch.float64)
        _, residual, rank = fit(x, y)
        assert residual.abs().max() < 1e-10
        assert rank == (9 if deficient else 10)
        u, s, _ = torch.linalg.svd(x, full_matrices=False)
        q = u[:, :rank]
        raw = torch.randn(73, 4, dtype=torch.float64)
        z = raw - q @ (q.T @ raw)
        output = torch.randn(4, 11, dtype=torch.float64)
        target = y + z @ output
        w, r, actual_rank = fit(x, target)
        spectrum, _ = tails(r)
        assert spectrum[:-4].sum() < 1e-10
        torch.testing.assert_close(r, z @ output, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(
            x.T @ r, torch.zeros(10, 11, dtype=torch.float64), atol=1e-9, rtol=1e-9
        )
        # For an arbitrary rank-h correction, truncated residual SVD attains the bound.
        ur, sr, vh = torch.linalg.svd(r, full_matrices=False)
        for h in (1, 2, 4):
            prediction = x @ w + (ur[:, :h] * sr[:h]) @ vh[:h]
            measured = (target - prediction).square().sum()
            expected = sr[h:].square().sum()
            torch.testing.assert_close(measured, expected, atol=1e-9, rtol=1e-9)
            checks.append(
                {
                    "rank_deficient": deficient,
                    "correction_rank": h,
                    "error": abs((measured - expected).item()),
                }
            )
        checks.append(
            {
                "rank_deficient": deficient,
                "design_rank": actual_rank,
                "affine_and_planted_residual": True,
            }
        )
    return checks


def run():
    assert not (ROOT / "protocol.json").exists(), "Preserve completed/partial experiments"
    prior = read(PRIOR / "result.json")
    assert read(PRIOR / "audit.json")["passed"]
    sources = [*(ROOT / "source").glob("*.py"), Path("research/affine_residual_capacity_plan.md")]
    hashes = {p.as_posix(): sha256(p) for p in sources}
    write_json(
        ROOT / "protocol.json",
        {
            "sources": hashes,
            "prior_result_sha256": sha256(PRIOR / "result.json"),
            "environment": environment(),
            "provenance": provenance(),
            "ranks": RANKS,
            "neural_updates": 0,
        },
    )
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in hashes:
            archive.write(path, path)
    checks = qualify()
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    print(f"Qualification passed: {len(checks)}", flush=True)
    rows = []
    start = time.perf_counter()
    for metadata in prior["diagnostics"]:
        activation, layer, seed = (metadata[k] for k in ("teacher", "layer", "seed"))
        label = f"{activation}_l{layer}_s{seed}"
        source = PRIOR / "pairs" / label / "data.pt"
        assert sha256(source) == metadata["data_sha256"]
        data = torch.load(source, weights_only=True)
        x = augment(data["x"])
        y = data["y"].double()
        before = time.perf_counter()
        affine, train_residual, train_rank = fit(x[:32768], y[:32768])
        _, output_basis = tails(train_residual)
        oracle_affine, report_residual, report_rank = fit(x[32768:], y[32768:])
        spectrum, _ = tails(report_residual)
        deployed_residual = y[32768:] - x[32768:] @ affine
        variance, drift = metadata["variance"], metadata["bf16_target_normalized_drift"]
        per_case = []
        for rank in RANKS:
            tail = spectrum[:-rank].sum().item() / 384 / variance
            bound = max(0, math.sqrt(tail) - math.sqrt(drift)) ** 2
            c = output_basis[:, -rank:]
            unrecovered = deployed_residual - (deployed_residual @ c) @ c.T
            fixed_mse = unrecovered.square().mean().item() / variance
            row = {
                "teacher": activation,
                "layer": layer,
                "seed": seed,
                "rank": rank,
                "oracle_bf16_normalized_tail": tail,
                "fp32_lower_bound": bound,
                "train_basis_oracle_bf16_error": fixed_mse,
                # A conservative upper bound rather than undercharging capture drift.
                "train_basis_oracle_fp32_upper": (math.sqrt(fixed_mse) + math.sqrt(drift)) ** 2,
                "bf16_fp32_drift": drift,
                "train_design_rank": train_rank,
                "report_design_rank": report_rank,
                "affine_train_fit_reporting_error": deployed_residual.square().mean().item()
                / variance,
            }
            rows.append(row)
            per_case.append(row)
        folder = ROOT / "cases" / label
        folder.mkdir(parents=True)
        torch.save(
            {
                "affine": affine,
                "oracle_affine": oracle_affine,
                "output_basis": output_basis,
                "residual_spectrum": spectrum,
            },
            folder / "statistics.pt",
        )
        write_json(
            folder / "metrics.json",
            {
                "source_sha256": sha256(source),
                "seconds": time.perf_counter() - before,
                "statistics_sha256": sha256(folder / "statistics.pt"),
                "rows": per_case,
            },
        )
        print(
            json.dumps({"completed": label, "h128": per_case[3], "h256": per_case[-1]}), flush=True
        )
    decisions = {}
    for kind, rank in (("gelu128", 128), ("gelu256", 256), ("swiglu85", 85), ("swiglu170", 170)):
        peers = [r for r in rows if r["rank"] == rank]
        means = {
            a: sum(r["train_basis_oracle_fp32_upper"] for r in peers if r["teacher"] == a) / 9
            for a in ("gelu", "swiglu")
        }
        gates = {
            "all_oracle_floors_below_five_percent": all(
                r["fp32_lower_bound"] <= 0.05 for r in peers
            ),
            "both_train_basis_means_below_five_percent": all(v <= 0.05 for v in means.values()),
        }
        decisions[kind] = {
            "rank": rank,
            "gates": gates,
            "train_basis_means": means,
            "verdict": "EARNS_FITTING" if all(gates.values()) else "CLOSED_AT_THIS_WIDTH",
        }
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": rows,
            "decisions": decisions,
            "elapsed_seconds": time.perf_counter() - start,
            "neural_updates": 0,
            "breakthrough": False,
        },
    )
    print(json.dumps(decisions, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
        raise
