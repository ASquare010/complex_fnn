"""Verify saved integer certificates and publish the bounded capacity result."""

from pathlib import Path

import numpy as np

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/modulated_capacity_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "check_exit.txt").read_text().strip() == "0"
hashes(p["sources"])
hashes(p["maintained_files"])
assert sha("results/conjugated_ffn_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/conjugated_ffn_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
assert (
    r["passed"] and r["training_updates"] == r["gpu_backwards"] == 0 and not r["cuda_initialized"]
)
for w in r["witnesses"]:
    assert sha(w["path"]) == w["sha256"]
    with np.load(w["path"], allow_pickle=False) as data:
        J, D, B, U = [data[k] for k in ("J", "D", "B", "U")]
    k = B.shape[1]
    assert np.array_equal(B.T @ B, 2 * np.eye(k, dtype=np.int64))
    assert np.array_equal(U @ U.T, 2 * np.eye(k, dtype=np.int64))
    residual = J - D - B @ U
    # Exact integer Frobenius certificate; no floating rank threshold is needed.
    assert (
        sum(int(v) * int(v) for v in residual.flat)
        == max(w["d"] - 2 * w["h"], 0)
        == w["squared_error"]
    )
    assert np.array_equal(D, np.diag(np.diag(D)))
    assert np.array_equal(J + J.T, np.zeros_like(J))
rows = [
    f"| {w['d']} | {w['h']} | {w['rank']} | {w['squared_error']} | {w['normalized_error']:.6f} | {w['monte_carlo']:.6f} | {w['standard_error']:.6f} |"
    for w in r["witnesses"]
]
conditional = [
    f"| {c['seed']} | {c['coordinate']} | {c['optimal_affine_risk']:.6f} | {c['nonlinear_mean_risk']:.6f} | {c['conditional_mean_risk']:.6f} | {c['max_identity_error']:.3e} |"
    for c in r["conditional_checks"]
]
counts = [
    f"| {h} | {n:,} | {100 * (1 - n / 295680):.2f}% | {max(1 - 2 * int(h) / 384, 0):.6f} |"
    for h, n in r["counts"].items()
]
report = f"""# H144: output modulation retains a shared-latent capacity limit

**Capacity obstruction verified; no training was allocated.** The proof and
its eight saved integer certificates, four conditional-risk checks and FP64
input/parameter finite-difference test all pass. CUDA was never initialized.
This is a scoped mathematical result, not a measured VRAM improvement or a
novelty claim.

For standard Gaussian input and linear target Mx, allowing arbitrary measurable
base and gate functions of one shared h-dimensional linear observation gives
exactly the same optimal population error as diagonal-plus-rank-h approximation.
For an explicit pairwise rotation J, that normalized error is **max(1-2h/d,0)**.
At d=384 and h=128, even this generous relaxation leaves **at least one third**
of the target energy as MSE. The finite GELU candidate cannot do better than
its relaxation; this does not assert it attains the relaxed optimum.

[Complete derivation and assumptions](modulated_capacity_theory.md) establish the
population statement. Numerical tests corroborate implementation, not replace
its proof. The result concerns one shared-latent module; it is not a bound for
deep stacks, arbitrary input distributions, independently conditioned gates,
nonlinear targets, finite reporting sets or language-model NLL.

## Explicit candidate and the width tradeoff

The isolated prototype computes z=GELU(Ux+a), then Bz+c+x*(Cz+e), with one feature
bank and two readouts. It needs 3dh+h+2d trainable scalars. Its input/parameter
FP64 gradcheck passes. A constant-gate witness gives identity output and full
output covariance rank with h=1, demonstrating why full output rank alone does
not establish capacity to approximate a general rotation.

| Latent h at d384 | Parameters | Reduction versus wide GELU h384 | Relaxed rotation error floor |
|---:|---:|---:|---:|
{chr(10).join(counts)}

At h=192 the displayed obstruction disappears, while parameter savings fall
from about50% to about25%. A zero lower bound is not an actual learned solution,
convergence guarantee or proof that 192 GELU features attain the affine optimum.
Fixed-width universal approximation must not be inferred from full rank.

## Exact and independent spectral certificates

| d | h | Constructed rank | Exact squared error | Normalized error | Gaussian Monte Carlo squared error | Estimated standard error |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The integer factors satisfy B^T B=2I and U U^T=2I, certifying their full column/
row ranks without floating tolerances. Their product has the declared rank.
Integer residual squares equal max(d-2h,0) exactly. A separate SVD check confirms
the spectrum and skew-projection lower bound. Gaussian checks use 8,192 samples
per witness and a prospectively fixed six-estimated-standard-error tolerance;
these stochastic checks neither prove nor weaken the exact bound.

## Conditional-risk identity checks

| Seed | Coordinate projection | Optimal affine risk | Nonlinear Monte Carlo risk | Mean conditional prediction | Maximum pointwise identity error |
|---:|---|---:|---:|---:|---:|
{chr(10).join(conditional)}

Three random rank-4 observations in dimension12 and one coordinate projection
check the Gaussian decomposition. The coordinate case includes zero conditional
coordinate variances. Two hundred diagonal perturbations per case verify the
quadratic optimality identity; each nonlinear gate simulation uses 32,768 samples.
Prediction discrepancies pass six estimated standard errors. The conditional
risk calculation is independent of the neural prototype and its autodiff test.

## Decision and provenance

Do not allocate a learning run for h128 on the premise that output modulation
alone repairs general approximation. Its full-rank output witness masks a
remaining shared-observation bottleneck. This does not eliminate modulation for
real FFN data or nonlinear targets. Any next hypothesis must address what input
information the gate/base can independently access, or explicitly accept the
larger latent width and measure its actual memory/compute tradeoff.

No historical failed recipe is reopened. No existing maintained file changed;
all prior receipt and maintained-source hashes verify. The earlier read-only
PowerShell inspection suffered a CLR failure before file creation and succeeded
on the next inspection; all mathematical checks completed in one attempt.
There were zero optimizer updates and zero GPU backwards. The broad research
goal remains open.

Prior art includes [FiLM](https://arxiv.org/abs/1709.07871) and
[diagonal-plus-low-rank expressivity](https://arxiv.org/html/2605.05659v1).
The derivation uses standard Gaussian conditioning and Eckart-Young approximation;
independent derivation does not establish priority or novelty.

[Prospective plan](modulated_capacity_plan.md),
[prototype](../results/modulated_capacity_v1/model.py),
[machine-readable checks](../results/modulated_capacity_v1/result.json),
[evidence receipt](../results/modulated_capacity_v1/receipt.json).
"""
reportpath = Path("research/modulated_capacity_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = """## Latest: shared-latent modulation capacity obstruction

[H144](modulated_capacity_results.md): **capacity obstruction verified**.
For Gaussian linear targets, arbitrary output modulation of one h-dimensional
observation reduces to diagonal-plus-rank-h approximation. An explicit rotation
has exact relaxed normalized error max(1-2h/d,0): one third at d384/h128.
Eight integer/SVD witnesses, four conditional-risk checks and candidate gradcheck
pass, with zero GPU work. [Proof](modulated_capacity_theory.md).

Do not allocate h128 fitting merely because its output rank is full. A new
hypothesis must alter the shared observation bottleneck or accept a larger latent
width and measure its resource cost. This is not a language-model bound or a
rejection of all modulation. The broader research goal remains open."""
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = """Latest: [H144 capacity proof](research/modulated_capacity_results.md)
identifies a shared-latent output-modulation limit before GPU fitting. No active
model/default changes; the broad research goal remains open."""
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
    Path("research/modulated_capacity_plan.md"),
    Path("research/modulated_capacity_theory.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H144",
    status="EVIDENCE_VERIFIED",
    gate="CAPACITY OBSTRUCTION VERIFIED",
    training_updates=0,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
