"""Publish the scoped theorem and audited CPU witnesses."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/augmentation_barrier_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "audit_exit.txt").read_text().strip() == "0" and a["status"] == "EVIDENCE_VERIFIED"
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(a["inputs"])
assert sha("results/narrow_coupling_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/narrow_coupling_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
matrix_rows = []
for q, k in p["pairs"]:
    cs = [c for c in a["matrix_checks"] if (c["q"], c["k"]) == (q, k)]
    matrix_rows.append(
        f"| {q} | {k} | {len(cs)} | {max(c['kernel_residual'] for c in cs):.3e} | {min(c['minimum_error_to_bound_ratio'] for c in cs):.4f} |"
    )
recovery_rows = [
    f"| {c['dtype'].removeprefix('torch.')} | {c['scale']:g} | {c['block_relative_errors'][0]:.3e} | {c['block_relative_errors'][1]:.3e} |"
    for c in a["recovery_checks"]
]
report = f"""# H155: small augmentation does not remove the exact even-output barrier

**Verified scoped dimension result; no new GPU training.** For a continuous
injective core operating on (x, zero padding), followed by a linear readout,
exactly representing an even target with q-dimensional affine output span
requires at least q extra state coordinates. The d=q=32 product target therefore
needs at least 32, not merely 4 or 8, under these assumptions.

This is not a lower bound on Gaussian MSE. It neither proves that approximate
learning with fewer coordinates is useless nor explains H154's measured error gap.
The broad goal of lower actual VRAM with competitive quality remains open.

## Proof and scope

[The complete argument](augmentation_barrier_plan.md) writes F(x)=W R(x,0)+b.
Full affine output span forces rank(W)=q. With k<q extra coordinates, ker(W)
has dimension d+k-q<d. Borsuk-Ulam supplies antipodal inputs whose core states
have equal kernel coordinates. Exact even outputs make their state difference
lie in ker(W); equal kernel coordinates then force identical states, violating
injectivity. This deduction uses the classical
[Borsuk-Ulam theorem, Hatcher Corollary 2B.7](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf#page=185).
It is not claimed as new topological theory or established research priority.

The dimension bound is sharp for unrestricted continuous cores: at k=q,
R(x,z)=(x,z+T(x)) is invertible by subtraction, and selecting the z coordinates
at input z=0 gives T(x). This inserts the target function explicitly; it is a
capacity witness, not a learned model or a parameter-efficiency result.
Input-plus-output dimensional constructions are already in
[Zhang et al., ICML 2020, Theorem 7](https://proceedings.mlr.press/v119/zhang20h/zhang20h.pdf).
The necessity argument here is specific to exact even targets and linear readouts.

For full-row-rank W, k<q, and antipodal core separation at least 2*m*r on a
radius-r sphere, the same argument gives the conditional uniform-error bound

    sup_(||x||=r) ||F(x)-T(x)|| >= sigma_q(W) * m * r.

At the selected pair the core difference lies in W's row space, so applying W
retains at least its smallest nonzero singular value times that separation.
The two equal targets then give the bound by the triangle inequality. Both
constants are model-dependent; collapse of either weakens the bound. No fixed
positive approximation floor across all parameter settings is established.
Nonlinear readouts, additional input bypasses, noninjective cores and different
domains can evade these hypotheses. Transformer residual branches, normalization
and final classifiers require their own analysis.

## Numerical checks and independent audit

Three fresh seeds, 27 FP64 orthogonal matrices, and 162 scaled witnesses verify
the linear examples. Input dimension is 32; output dimension q and added state k
vary below. Radii are 0.5, 1 and 2; readout scales are 1 and 0.01. An explicit
null vector supplies each antipodal pair. The observed worst error divided by
the conditional bound is at least one in every case. These examples do not
numerically prove the topological theorem for arbitrary nonlinear maps.

| Output q | Extra state k | Matrices | Largest kernel residual | Smallest worst-error/bound ratio |
|---:|---:|---:|---:|---:|
{chr(10).join(matrix_rows)}

The cyclic product's full affine span is verified with exact integer witnesses:
T(0)=0 and T(e_i+e_(i+1))=e_i; signed pairs have identical outputs. Mixing outputs
by an invertible matrix, translating them, or applying nonzero scalar target
normalization preserves the required span and parity. The explicit product shear
passes FP64 input and parameter finite differences. The NumPy audit independently
checks all saved matrices, scaled errors and both reconstructed state blocks.
The numerical witness matrices also regenerate exactly from their recorded seeds.

## Exact inverse versus floating-point recovery

At sufficient augmentation, the product shear recovers x unchanged and computes
z=(z+T(x))-T(x). Its forward evaluation at z=0 gives the product exactly in the
tested arithmetic. Inversion with arbitrary small z can nevertheless lose it
when T(x) is large. The 257-sample stress uses x scaled as below and unit-scale
random z. Errors are relative L2 norms for each block, never pooled together.

| Precision | Input scale | Recovered x error | Recovered auxiliary z error |
|---|---:|---:|---:|
{chr(10).join(recovery_rows)}

At input scale 1,000, FP32 auxiliary recovery loses about 2.5% relative accuracy;
FP64 error is below 5e-11. This is a large-input stress, not a claim about ordinary
normalized activations. It rules out assuming reliable arbitrary-scale inversion
solely from the real-arithmetic formula. No reconstructed training pass was run.

## Consequence for the VRAM objective

At batch 2,048, d=384 and FP32, one uncompressed d-wide tensor is 3 MiB; a 2d-wide
state is 6 MiB. These are exact tensor-byte calculations, not measured peak VRAM.
They exclude parameters, optimizer state, reconstruction temporaries and other
layers. Full augmentation could still beat storing every hidden activation, but
it must be compared against equally checkpointed standard networks.

Do not allocate another GPU sweep on the claim that tiny padding fixes exact
full-vector even capacity. A sufficiently augmented core or a different output
map needs an explicitly budgeted learning and resource test. Approximation with
small padding remains possible; the proof is a design constraint, not a universal
candidate-elimination rule. Parameter count, learnability, latency and real-data
quality are still unproven for any proposed replacement.

## Reproduction and integrity

UV-managed Python ran launch.py stages check, audit and finish. No optimizer
updates or CUDA initialization occurred. Source/maintained-code hashes, the H154
receipt, saved arrays and all NumPy checks verify. The existing output directory
refuses overwrites; use a fresh directory with adjusted ROOT/module paths to
reproduce. The frozen plan, protocol.json, witnesses.pt, result.json and audit.json
retain the complete assumptions, values and tolerances.
"""
report_path = Path("research/augmentation_barrier_results.md")
report_path.write_text(report, encoding="utf-8")
for path, title in [
    (Path("README.md"), "# Memory- and parameter-efficient FFN research"),
    (Path("research/CURRENT_STATE.md"), "# Current research state"),
]:
    old = path.read_text(encoding="utf-8")
    assert old.startswith(title)
    if path.name == "README.md":
        rest = old[len(title) :].lstrip().replace("Latest:", "Earlier:", 1)
        intro = "Latest: [H155 augmentation dimension proof](research/augmentation_barrier_results.md):\nexact even output with full affine span requires at least as many extra state\ncoordinates as output dimensions, under the stated reversible/linear-readout assumptions.\nCPU witnesses verified; no training or VRAM-performance claim.\n\n"
    else:
        rest = old[len(title) :].lstrip().replace("## Latest:", "## Previous:", 1)
        intro = "## Latest: augmentation dimension barrier\n\n[H155](augmentation_barrier_results.md): scoped exact-representation necessity\nk >= q, with a matching unrestricted shear construction and conditional uniform\nerror bound. 27 matrices / 162 scaled CPU checks and six inverse stresses audited.\nNot a Gaussian MSE bound; FP32 large-product cancellation is exposed separately.\nZero GPU training. Tiny padding is not an exact-capacity repair; VRAM goal stays open.\n\n"
    path.write_text(title + "\n\n" + intro + rest, encoding="utf-8")
files = [
    f
    for f in ROOT.iterdir()
    if f.is_file() and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    report_path,
    Path("research/augmentation_barrier_plan.md"),
    Path("README.md"),
    Path("research/CURRENT_STATE.md"),
]
receipt = dict(
    study="H155",
    status="EVIDENCE_VERIFIED",
    result="Scoped augmentation dimension proof and CPU witnesses verified",
    training_updates=0,
    gpu_used=False,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
print({k: v for k, v in receipt.items() if k != "files"})
