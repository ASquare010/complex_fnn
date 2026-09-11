"""CPU exact witnesses, independent spectral checks and Gaussian risk diagnostics."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.modulated_capacity_v1.model import ModulatedFFN
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/modulated_capacity_v1")
CASES = ((6, 0), (6, 1), (6, 2), (6, 3), (384, 64), (384, 128), (384, 192), (384, 256))


def witness(d, h):
    k = min(h, d // 2)
    J = np.kron(np.eye(d // 2, dtype=np.int64), np.array([[0, -1], [1, 0]], dtype=np.int64))
    D = np.zeros((d, d), dtype=np.int64)
    B = np.zeros((d, k), dtype=np.int64)
    U = np.zeros((k, d), dtype=np.int64)
    for i in range(k):
        D[2 * i, 2 * i] = 1
        D[2 * i + 1, 2 * i + 1] = -1
        B[2 * i, i] = -1
        B[2 * i + 1, i] = 1
        U[i, 2 * i : 2 * i + 2] = 1
    L = B @ U
    residual = J - D - L
    assert np.array_equal(J.T @ J, np.eye(d, dtype=np.int64))
    assert np.array_equal(J, -J.T)
    assert np.array_equal(B.T @ B, 2 * np.eye(k, dtype=np.int64))
    assert np.array_equal(U @ U.T, 2 * np.eye(k, dtype=np.int64))
    squared = int(np.square(residual).sum())
    assert squared == max(d - 2 * h, 0)
    singular = np.linalg.svd(L.astype(float), compute_uv=False)
    assert np.count_nonzero(singular > 1e-10) == k
    assert np.allclose(singular[:k], 2, rtol=0, atol=1e-12)
    skew = (L - L.T) / 2
    assert np.linalg.matrix_rank(skew, tol=1e-10) == 2 * k
    lower = float(
        np.square(np.linalg.svd(J.astype(float), compute_uv=False)[min(2 * h, d) :]).sum()
    )
    assert abs(lower - squared) < 1e-10
    rng = np.random.default_rng(90000 + d + h)
    x = rng.standard_normal((8192, d))
    errors = np.square(x @ residual.T).sum(axis=1)
    mean = float(errors.mean())
    se = float(errors.std(ddof=1) / math.sqrt(len(errors)))
    assert abs(mean - squared) <= 6 * se + 1e-10
    path = ROOT / f"witness_d{d}_h{h}.npz"
    np.savez(path, J=J, D=D, B=B, U=U)
    return dict(
        d=d,
        h=h,
        rank=k,
        squared_error=squared,
        normalized_error=squared / d,
        svd_lower=lower,
        monte_carlo=mean,
        standard_error=se,
        path=path.as_posix(),
        sha256=sha(path),
    )


def conditional(seed, coordinate=False):
    rng = np.random.default_rng(seed)
    d, h = 12, 4
    U = np.eye(d)[:h] if coordinate else np.linalg.qr(rng.standard_normal((d, h)))[0].T
    Q = U.T @ U
    P = np.eye(d) - Q
    M = rng.standard_normal((d, d)) / math.sqrt(d)
    diagonal = np.diag(P)
    diagMP = np.diag(M @ P)
    optimum = np.divide(diagMP, diagonal, out=np.zeros(d), where=diagonal > 1e-12)
    D = np.diag(optimum)
    risk = float(np.square((M - D) @ P).sum())
    affine = D + (M - D) @ Q
    assert np.linalg.matrix_rank(affine - D, tol=1e-10) <= h
    assert np.allclose(np.square(M - affine).sum(), risk, rtol=1e-12, atol=1e-12)
    max_identity_error = 0.0
    for _ in range(200):
        delta = rng.standard_normal(d)
        direct = float(np.square((M - D - np.diag(delta)) @ P).sum())
        decomposed = risk + float(np.sum(diagonal * delta**2))
        max_identity_error = max(max_identity_error, abs(direct - decomposed))
        assert math.isclose(direct, decomposed, rel_tol=1e-12, abs_tol=1e-12)
    x = rng.standard_normal((32768, d))
    z = x @ U.T
    delta = 0.2 * np.tanh(z @ rng.standard_normal((h, d)) / math.sqrt(h))
    gate = optimum + delta
    prediction = x @ Q @ M.T + (x @ P) * gate
    losses = np.square(x @ M.T - prediction).sum(axis=1)
    # Conditional risk for each latent sample; averaging it isolates sampling of n.
    conditional_risk = risk + (delta**2 * diagonal).sum(axis=1)
    difference = losses - conditional_risk
    discrepancy = float(difference.mean())
    se = float(difference.std(ddof=1) / math.sqrt(len(difference)))
    assert abs(discrepancy) <= 6 * se + 1e-10
    return dict(
        seed=seed,
        coordinate=coordinate,
        rank=h,
        optimal_affine_risk=risk,
        nonlinear_mean_risk=float(losses.mean()),
        conditional_mean_risk=float(conditional_risk.mean()),
        discrepancy=discrepancy,
        standard_error=se,
        max_identity_error=max_identity_error,
    )


def run():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    assert not (ROOT / "protocol.json").exists()
    hashes(read("results/conjugated_ffn_v1/receipt.json")["files"])
    base = read("results/conjugated_ffn_v1/protocol.json")
    hashes(base["maintained_files"])
    for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
        (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
    p = dict(
        study="H144",
        training_updates=0,
        gpu_backwards=0,
        cases=CASES,
        seeds=[17, 29, 43],
        maintained_files=base["maintained_files"],
        prior_receipt=sha("results/conjugated_ffn_v1/receipt.json"),
        sources={
            f.as_posix(): sha(f)
            for f in [
                *ROOT.glob("*.py"),
                Path("research/modulated_capacity_plan.md"),
                Path("research/modulated_capacity_theory.md"),
            ]
        },
    )
    write_json(ROOT / "protocol.json", p)
    torch.manual_seed(17)
    model = ModulatedFFN(4, 2).double()
    parameters = dict(model.named_parameters())
    names = list(parameters)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, *ps: functional_call(model, dict(zip(names, ps, strict=True)), (a,)),
        (x, *parameters.values()),
        fast_mode=True,
    )
    identity = ModulatedFFN(6, 1).double()
    with torch.no_grad():
        identity.base.weight.zero_()
        identity.base.bias.zero_()
        identity.gate.weight.zero_()
        identity.gate.bias.fill_(1)
    probe = torch.cat((torch.zeros(1, 6, dtype=torch.float64), torch.eye(6, dtype=torch.float64)))
    result = identity(probe)
    assert torch.equal(result, probe)
    assert int(torch.linalg.matrix_rank(result - result.mean(0))) == 6
    counts = {h: sum(p.numel() for p in ModulatedFFN(384, h).parameters()) for h in (128, 192)}
    assert all(n == 3 * 384 * h + h + 2 * 384 for h, n in counts.items())
    witnesses = [witness(*case) for case in CASES]
    conditional_checks = [conditional(seed) for seed in p["seeds"]] + [conditional(17, True)]
    assert not torch.cuda.is_initialized()
    hashes(p["sources"])
    hashes(p["maintained_files"])
    write_json(
        ROOT / "result.json",
        dict(
            passed=True,
            witnesses=witnesses,
            conditional_checks=conditional_checks,
            gradcheck_passed=True,
            full_rank_identity=True,
            counts=counts,
            training_updates=0,
            gpu_backwards=0,
            cuda_initialized=False,
        ),
    )
    print("Passed:8 exact/SVD witnesses,4 conditional-risk checks,FP64 gradcheck; zero GPU work.")


if __name__ == "__main__":
    run()
