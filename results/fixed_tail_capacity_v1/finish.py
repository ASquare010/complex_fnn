"""Independently verify finite-sample inequalities and publish the capacity result."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import read, hashes, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
from fractions import Fraction
import numpy as np
import math

ROOT = Path("results/fixed_tail_capacity_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "check_exit.txt").read_text().strip() == "0"
hashes(p["sources"])
hashes(p["maintained_files"])
assert sha("results/fused_timing_confirmation_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/fused_timing_confirmation_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
checks = []
for c in r["cases"]:
    assert sha(c["path"]) == c["sha256"]
    s = {k: v.numpy() for k, v in torch.load(c["path"], weights_only=True).items()}
    src = p["models"][c["index"]]
    original = {
        k: v.double().numpy()
        for k, v in torch.load(src["state_path"], weights_only=True)["model"].items()
    }
    assert np.array_equal(s["q"], original["q"])
    A = np.eye(384)
    beta = np.zeros(384)
    tail = np.eye(384)
    for j in reversed(range(8)):
        beta += tail @ original["bias"][j]
        tail = tail @ s["q"][j]
    A = tail
    assert np.allclose(A, s["A"], rtol=1e-11, atol=1e-12)
    assert np.allclose(beta, s["beta"], rtol=1e-11, atol=1e-12)
    for v in c["results"]:
        x = s["x0"] * v["scale"]
        y = s[f"output_{v['scale']}"]
        M = s["target"] if v["target"] == "independent" else -s["A"]
        center = x - x.mean(axis=0)
        covariance = center.T @ center / len(x)
        delta = s["A"] - M
        mismatch = math.sqrt(max(float(np.trace(delta @ covariance @ delta.T)), 0))
        lower = max(mismatch - c["universal_bound"], 0) ** 2
        mse = float(np.sum((y - x @ M.T) ** 2) / len(x))
        assert math.isclose(lower, v["lower"], rel_tol=1e-10, abs_tol=1e-7)
        assert math.isclose(mse, v["mse"], rel_tol=1e-10, abs_tol=1e-7) and mse + 1e-7 >= lower
        assert (
            np.max(np.linalg.norm(y - x @ s["A"].T - s["beta"], axis=1))
            <= c["actual_shape_bound"] + 1e-9
        )
        checks.append(
            dict(index=c["index"], scale=v["scale"], target=v["target"], lower=lower, mse=mse)
        )
# Independent exact finite-distribution witness, all integer arithmetic.
points = 16 * np.concatenate((np.eye(16, dtype=np.int64), -np.eye(16, dtype=np.int64)))
assert np.array_equal(points.T @ points, 512 * np.eye(16, dtype=np.int64))
assert not points.sum(axis=0).any()
energy = Fraction(sum(int(v) ** 2 for v in points.flat), 32)
mismatch_squared = 4 * energy
assert mismatch_squared == 1024
mismatch = math.isqrt(mismatch_squared.numerator)
assert mismatch * mismatch == mismatch_squared
lower = (mismatch - 8) ** 2
assert Fraction(lower, energy) == Fraction(9, 4)
assert r["witness"]["mse_lower"] == lower == 576 and r["witness"]["target_energy"] == energy == 256
assert not torch.cuda.is_initialized()
write_json(
    ROOT / "audit.json",
    dict(
        status="VERIFIED",
        checks=checks,
        exact_witness_normalized="9/4",
        training_updates=0,
        backwards=0,
        cuda_initialized=False,
    ),
)
rows = []
for c in r["cases"]:
    for v in c["results"]:
        rows.append(
            f"| {c['source_batch']} | {c['seed']} | {v['scale']} | {v['target']} | {v['normalized_lower']:.4f} | {v['normalized_mse']:.4f} |"
        )
independent = [
    v["normalized_lower"]
    for c in r["cases"]
    for v in c["results"]
    if v["scale"] == 8 and v["target"] == "independent"
]
report = f"""# H151: fixed mixing locks the asymptotic linear map

**A family-wide capacity obstruction is established.** The H147–H150 reversible
stack cannot learn an arbitrary linear tail by changing its scalar shapes and
biases. An exact finite-distribution witness forces normalized MSE>=9/4 for every
allowed setting. Six saved candidate models corroborate the decomposition and
finite-sample bounds. No GPU work or training was needed.

This is a limitation of the current fixed-mixing predictor, not a rejection of
reversible representations, compact-domain learning, or architectures with a
learnable outer map. Resource gains alone do not resolve this capacity issue.

## Derivation

Write each layer as h_j=Q_j*h_(j-1)+b_j+r_j, where z_j=Q_j*h_(j-1)+b_j and
r_j=a_j*z_j/(1+abs(z_j)) coordinatewise. Since abs(r_ji)<=abs(a_ji), unrolling gives

    F(x)=A*x+beta+R(x),  A=Q_L...Q_1,
    beta=sum_j (Q_L...Q_(j+1))*b_j,
    ||R(x)|| <= B=sum_j ||Q_L...Q_(j+1)||_op*||a_j||.

For orthogonal Q and abs(a_ji)<=c/L, B<=c*sqrt(d), regardless of biases.
Thus for every direction v and every finite learned parameter setting,

    lim_(t->infinity) F(t*v)/t = A*v.

A is fixed before training. If target M differs from A, choose v with (A-M)v!=0:
error along t*v diverges as t grows, since beta+R stays bounded. Uniformly bounded
error on all of R^d is impossible for such a linear target. Bias magnitude may
move where this behavior appears but cannot change the limiting map.

For finite-second-moment X with Cov(X)=sigma^2*I, center F(X)-MX. Reverse triangle
in the Hilbert space of square-integrable random vectors gives

    E||F(X)-MX||^2
      >= Var(F(X)-MX)
      >= max(sigma*||A-M||_F - sqrt(Var(R(X))), 0)^2
      >= max(sigma*||A-M||_F - B, 0)^2.

Var denotes total centered second moment. The translation beta disappears.
The inequality is uniform over all allowed theta and all finite biases.
No Gaussian assumption is needed. For a finite sample, use its empirical
distribution: replace sigma*||A-M||_F by RMS((X-meanX)*(A-M)^T).
These are lower bounds, not claims that the architecture attains them.

## Exact counterexample

Take d16, L8, c2, Q_j=I, M=-I. Let X be uniform on the32 vectors +/-16*e_i.
Then mean(X)=0, Cov(X)=16I, target energy is256, and RMS((A-M)X)=32.
The maximal nonlinear correction bound is B=2*sqrt(16)=8. Consequently

    MSE >= (32-8)^2 = 576,
    MSE / E||MX||^2 >= 576/256 = 9/4.

The zero predictor has ratio1, so every model in this particular constrained
family is provably worse on this distribution. Integer covariance/energy checks
and an independent rational-arithmetic audit verify the witness exactly.
The argument also applies to any exactly orthogonal fixed A with target -A;
Q_j=I makes the numerical certificate entirely integer-based. This single
counterexample does not establish a universal performance ranking.

## Saved-model checks

All six H150 learned:fused repeat0 checkpoints are included. Fresh256x384 Gaussian
samples, scales1/4/8, two targets: the original independent orthogonal target and
-A. Eighteen CPU FP64 forward passes produce36 comparisons. These are diagnostics
of saved models, not fresh fits or unbiased generalization estimates. Normalized
bounds and observed MSE divide by each sample's zero-predictor target energy.

Stored FP32 Q matrices are approximately orthogonal. Their numerical bound uses
sqrt(1+||Q^TQ-I||_F), products of these layer bounds, and1e-12 padding. The general
nonorthogonal theorem above applies; computed bounds are FP64 estimates, not
formally rounded interval certificates. The exact counterexample does not depend
on those estimates. All pointwise residual/decomposition checks pass. Independent
NumPy auditing uses covariance traces rather than the worker's matrix-output RMS.

| Source batch | Seed | Input scale | Target | Normalized lower bound | Observed normalized MSE |
|---:|---:|---:|---|---:|---:|
{chr(10).join(rows)}

At scale8, independent-target estimated lower bounds range from
{min(independent):.4f} to{max(independent):.4f}, above the zero predictor's1.
These bounds hold for every allowed shape/bias setting with these fixed matrices
in the exact-real model; the displayed numerical values use the stated FP64
estimates. Scale1 bounds can be zero and do not rule out good compact-domain
performance. This scale stress does not prove failure on normalized language data.

## Consequence for the next architecture

Do not interpret6,144 trainable scalars plus fixed Q as a general substitute for
learnable FFN weights. Stop refining this exact predictor solely for timing.
The H149/H150 resource evidence remains useful, but the fixed tail map needs a
separate capacity mechanism before making broad replacement claims.

One concrete escape is a learned final linear readout. At theta=0 the stack is
A*x+beta. For invertible A, choose readout W=M*A^(-1) and bias -W*beta; any linear
target Mx is then exactly representable in real arithmetic. At d384, this adds
147,840 trainable scalars including bias, bringing the stack total to153,984.
It preserves internal activation reconstruction but changes endpoint gradients,
optimizer memory and stability. This proves linear representability only; it
requires new gradient/resource/quality tests. H107's rejected additive full-affine
plus low-rank branch is a different architecture and remains rejected.

Learnable mixing or a learnable tail scale are other changes, but a scalar scale
alone cannot generally correct an arbitrary fixed rotation. An unrestricted
readout can evade H147's contraction bound, so that bound must not be applied
unchanged to the extended predictor.

Prior work on [neural extrapolation](https://arxiv.org/abs/2009.11848) analyzes
linear behavior along rays in ReLU networks. Here the restrictive fact is that
the tail matrix is fixed rather than learned. The decomposition and L2 argument
are elementary; independent derivation does not establish novelty.

All source/previous receipt hashes verify. There were zero optimizer updates,
zero backwards and no CUDA initialization. Maintained modules/defaults are
unchanged. The broader research goal remains open.

[Plan](fixed_tail_capacity_plan.md), [saved checks](../results/fixed_tail_capacity_v1/result.json),
[independent audit](../results/fixed_tail_capacity_v1/audit.json),
[receipt](../results/fixed_tail_capacity_v1/receipt.json).
"""
reportpath = Path("research/fixed_tail_capacity_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = """## Latest: fixed-tail capacity obstruction

[H151](fixed_tail_capacity_results.md): bounded scalar corrections cannot change
the fixed mixing product in the linear tail. Exact witness normalized MSE>=9/4;
six saved models/36 risk comparisons audited on CPU. Zero GPU/training work.
A learned outer map can evade the obstruction but needs fresh verification.
Stop timing-only refinement of the unmodified predictor. Research goal remains open."""
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = "Latest: [H151 fixed-tail capacity proof](research/fixed_tail_capacity_results.md)\nidentifies a representational limit and a concrete readout extension to test."
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    reportpath,
    Path("research/fixed_tail_capacity_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H151",
    status="EVIDENCE_VERIFIED",
    gate="FIXED-TAIL CAPACITY OBSTRUCTION VERIFIED",
    training_updates=0,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
