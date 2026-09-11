"""NumPy verification of saved witnesses; no candidate forward implementation."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import numpy as np

ROOT = Path("results/augmentation_barrier_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
assert (ROOT / "check_exit.txt").read_text().strip() == "0"
assert not (ROOT / "audit.json").exists()
hashes(p["sources"])
hashes(p["maintained_files"])
assert sha(ROOT / "witnesses.pt") == r["witnesses_sha256"]
saved = torch.load(ROOT / "witnesses.pt", weights_only=True)
assert len(saved["examples"]) == len(r["rows"]) == 27 and len(saved["recovery"]) == 6
matrix_checks = []
for ex, row in zip(saved["examples"], r["rows"], strict=True):
    q, k, seed = [ex[n] for n in ("q", "k", "seed")]
    assert (q, k, seed) == (row["q"], row["k"], row["seed"])
    Q, v = ex["Q"].numpy(), ex["v"].numpy()
    n = 32 + k
    gen = torch.Generator().manual_seed(seed + q * 100 + k * 10000)
    regenerated = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64)).Q
    assert torch.equal(regenerated, ex["Q"])
    assert Q.shape == (n, n) and v.shape == (32,) and k < q
    assert np.max(np.abs(Q.T @ Q - np.eye(n))) <= 1e-10
    residual = np.linalg.norm(Q[q:, :32] @ v)
    assert residual <= 1e-10 and abs(np.linalg.norm(v) - 1) <= 1e-10
    assert len(row["cases"]) == 6
    ratios = []
    for c in row["cases"]:
        x = v * c["radius"]
        tx = (x * np.roll(x, -1))[:q]
        fx = c["alpha"] * (Q[:q, :32] @ x)
        errors = np.array([np.linalg.norm(fx - tx), np.linalg.norm(-fx - tx)])
        np.testing.assert_allclose(errors, c["errors"], rtol=1e-12, atol=1e-12)
        bound = c["alpha"] * c["radius"]
        assert bound == c["bound"] and errors.max() + 1e-10 >= bound
        np.testing.assert_allclose(2 * np.linalg.norm(fx), 2 * bound, rtol=1e-10, atol=1e-10)
        ratios.append(float(errors.max() / bound))
    matrix_checks.append(
        dict(
            seed=seed,
            q=q,
            k=k,
            kernel_residual=float(residual),
            minimum_error_to_bound_ratio=min(ratios),
        )
    )
x = saved["witness"].numpy()
y = x * np.roll(x, -1, axis=1)
assert np.array_equal(y, np.eye(32, dtype=np.int64)) and np.array_equal(
    y, saved["products"].numpy()
)
assert np.array_equal(y, (-x) * np.roll(-x, -1, axis=1))
recovery_checks = []
for ex, row in zip(saved["recovery"], r["recovery"], strict=True):
    s, out, rest = [ex[n].numpy() for n in ("input", "output", "restored")]
    x, z = s[:, :32], s[:, 32:]
    f = x * np.roll(x, -1, axis=1)
    expected_out = np.concatenate((x, z + f), axis=1)
    expected_rest = np.concatenate((out[:, :32], out[:, 32:] - f), axis=1)
    assert np.array_equal(out, expected_out) and np.array_equal(rest, expected_rest)
    errors = [
        float(
            np.linalg.norm(u.astype(np.float64) - v.astype(np.float64))
            / np.linalg.norm(v.astype(np.float64))
        )
        for u, v in zip(np.split(rest, 2, axis=1), np.split(s, 2, axis=1), strict=True)
    ]
    np.testing.assert_allclose(errors, row["block_relative_errors"], rtol=1e-12, atol=1e-15)
    recovery_checks.append(
        dict(dtype=row["dtype"], scale=row["scale"], block_relative_errors=errors)
    )
assert r["passed"] and r["gradcheck"] and not torch.cuda.is_initialized()
write_json(
    ROOT / "audit.json",
    dict(
        status="EVIDENCE_VERIFIED",
        matrix_checks=matrix_checks,
        scaled_checks=162,
        recovery_checks=recovery_checks,
        integer_span_verified=True,
        training_updates=0,
        gpu_used=False,
        inputs={
            f.as_posix(): sha(f)
            for f in ROOT.iterdir()
            if f.is_file() and f.suffix in (".json", ".pt", ".py")
        },
    ),
)
print("NumPy audit verified every matrix, scaled bound and inverse block.")
