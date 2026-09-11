"""Publish inverse proof, capacity limit and audited reconstruction evidence."""

import statistics as st
from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/reversible_softsign_v1")
p, r, a = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit.json")]
assert not (ROOT / "receipt.json").exists()
assert (
    (ROOT / "audit_exit.txt").read_text().strip() == "0"
    and a["status"] == "VERIFIED"
    and r["passed"]
)
hashes(p["sources"])
hashes(p["maintained_files"])
hashes(read(ROOT / "audit_protocol.json")["inputs"])
assert sha("results/modulation_depth_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/modulation_depth_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
rows = []
stats = []
for dtype in p["dtypes"]:
    for depth in p["depths"]:
        for c in p["strengths"]:
            cases = [
                v
                for v in r["cases"]
                if (v["dtype"], v["depth"], v["c"]) == ("torch." + dtype, depth, c)
            ]
            assert len(cases) == 3
            metrics = {
                k: dict(
                    mean=st.mean(v["errors"][k] for v in cases),
                    median=st.median(v["errors"][k] for v in cases),
                    sample_variance=st.variance(v["errors"][k] for v in cases),
                    maximum=max(v["errors"][k] for v in cases),
                )
                for k in ("input", "cycle", "gradient")
            }
            stats.append(dict(dtype=dtype, depth=depth, c=c, metrics=metrics))
            rows.append(
                f"| {dtype} | {depth} | {c} | {metrics['input']['maximum']:.3e} | {metrics['gradient']['maximum']:.3e} | {cases[0]['lower']:.4f} | {cases[0]['upper']:.4f} | {cases[0]['lower'] ** 2:.4f} |"
            )
write_json(ROOT / "summary.json", stats)
worst32 = max(v["errors"]["input"] for v in r["cases"] if v["dtype"] == "torch.float32")
report = f"""# H147: closed-form reversible scalar geometry passes numerical preflight

**Numerical preflight passed; no resource or quality claim.** All36 GPU stress
cases pass through128 layers in FP32/FP64. Independent NumPy forward-retained
VJPs verify input, shape and bias gradients separately. This earns investigation
of a memory-efficient implementation; the current reference is not such an
implementation. The broad research goal remains open.

## Scalar map and stable inverse

For rho=c/L<1 and a=rho*tanh(theta), define

    phi_a(x) = x + a*x/(1+abs(x)).

It is continuously differentiable, strictly increasing, onto the real line and
has derivative 1+a/(1+abs(x))^2, including derivative1+a at zero. Its slope is
in[1-rho,1+rho], its tails are asymptotically linear, and a=0 is the identity.
The correction is bounded by abs(a). This does not initialize a ReLU network.

Because phi is odd and monotone, sign(x)=sign(y). Put v=abs(y), t=abs(x) and
B=1+a-v. Rearranging v=t+a*t/(1+t) gives

    t^2 + B*t - v = 0.

The nonnegative root can be computed as 2v/(sqrt(B^2+4v)+B) when B>=0, and
(sqrt(B^2+4v)-B)/2 otherwise. The first branch rationalizes subtraction that
would cancel near zero; the second avoids cancellation in the denominator for
large v. The implementation uses hypot(B,2*sqrt(v)) and safe unused-branch
denominators. sign(y) restores x; zero maps to zero. Inverse is used under
no_grad; its autodiff behavior at sign/branch boundaries is not an API promise.
The tested range is finite: signed1e-12 through1e12, plus zero. No claim covers
all representable floating values, overflow or arbitrary mixed precision.

## Orthogonal stack, input gradients and a capacity cost

A layer is phi(Qx+b), with fixed orthogonal Q and per-channel theta,b.
For two inputs u,v, scalar monotonicity gives the layer distance bounds

    (1-rho)||u-v|| <= ||layer(u)-layer(v)|| <= (1+rho)||u-v||.

Orthogonal Q preserves distances and biases cancel before activation. Composing
L layers therefore gives m=(1-rho)^L and M=(1+rho)^L as lower and upper distance
bounds, and every input-Jacobian singular value lies in[m,M]. With fixed c and
increasing L these bounds tend to exp(-c) and exp(c); this is the reason for
scaling each layer's deformation by1/L. It bounds input VJPs, not parameter
gradients, optimization trajectories or numerical roundoff. Stored floating Q
is only approximately orthogonal; its residual is independently checked.

Strong inverse conditioning also limits contraction. For iid isotropic X,X',
let H=F-G where F has lower distance factor m and target G is k-Lipschitz, k<m.
Reverse triangle yields ||H(X)-H(X')|| >= (m-k)||X-X'||. Thus

    E||F(X)-G(X)||^2 >= Var(H(X))
                        = E||H(X)-H(X')||^2/2
                       >= (m-k)^2*d.

Here Var denotes total centered second moment, and Cov(X)=I; Gaussianity is
not required. Finite second moments follow for Lipschitz maps. For constant G,
k=0 and the per-dimension MSE floor is m^2. The claim concerns this whole
bijective stack as the predictor. An external unconstrained projection, scale,
readout or subtractive residual can evade it; such modifications also require
new gradient/resource analysis. It is not a bound on all reversible networks.
This is a lower bound, not a claim that the finite scalar family attains it.

## Reconstruction and derivative experiment

Three seeds131/149/167, d16, batch32, depths8/32/128, c0.5/2 and FP32/FP64.
Q is generated by CPU FP64 QR then cast; theta is uniform[-1.5,1.5], biases
normal with sd.1. Input and output cotangent are independent standard Gaussian.
Native autograd supplies one VJP per case. The analytical reverse reconstructs
z=phi_inverse(y), then x=Q^T(z-b) and uses

    dz = dy*(1+a/(1+abs(z))^2)
    dtheta = sum_batch(dy*z/(1+abs(z)))*rho*(1-tanh(theta)^2)
    db = sum_batch(dz), dx = Q^T*dz.

The implementation uses row-vector equivalents. It retains only the current
reconstruction and cotangent inside that reverse loop, plus parameters and their
gradients. The surrounding native reference graph is used for verification;
its peak allocation must not be advertised as optimized training memory.

Worst relative errors across three seeds; the input reconstruction and
concatenated-gradient tolerances were1e-4 FP32 and1e-10 FP64. Forward cycles and
analytical full-Jacobian singular values also passed. The table's m/M are global
real-arithmetic bounds, not observed extremes.

| Dtype | Depth | c | Input error | Gradient error | m | M | Constant-target MSE floor/d |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Maximum FP32 input reconstruction error is{worst32:.3e}. CPU scalar inverse
checks and independent quadratic residuals pass for a=-.95,0,.95, plus a
finite-difference input/shape gradient check at nonzero shape including x=0.
The separate NumPy audit retains forward activations and never calls the inverse,
so its VJP does not inherit reconstruction errors. All36 cases pass the same
precision tolerance for each gradient family separately, as well as output and
reconstruction. Seeds' means, medians and sample variances are in summary.json.

## Scope, cost and evidence

This dense-orthogonal prototype has2Ld learned scalars and Ld^2 fixed matrix
entries. Fixed matrices occupy memory even though they are not trainable. At
L128/d16, this is4,096 parameters and32,768 matrix entries (128KiB FP32).
Using dense reference mixing does not prove an efficient structured implementation
exists with equal quality. QR construction cost is outside the GPU diagnostics;
no latency comparison, VRAM reduction or training convergence is claimed.

Thirty-six native GPU VJPs, zero optimizer updates,37 clean allocated/reserved
CUDA boundaries. The independent CPU audit uses zero backward calls and no CUDA.
RTX4070 Laptop GPU, driver610.62, PyTorch{a["torch_version"]}, CUDA build
{a["cuda_build"]}. No clock/power changes. Model defaults and maintained code
remain unchanged; all prior receipts and frozen source hashes verify.

[RevNets](https://arxiv.org/abs/1707.04585) established activation reconstruction
for memory-efficient training. [Linear rational spline flows](https://proceedings.mlr.press/v108/dolatabadi20a.html)
establish analytically invertible nonlinear geometry as prior art. The formula
and bounds here were independently derived; that does not establish priority.
This scalar residual differs from the previously rejected centered pair twist,
but novelty and practical benefit remain unproven.

Next: implement reconstruction under a real training autograd boundary and compare
against equally checkpointed conventional FFNs. Include a fixed-shape/affine
control and account for all mixing buffers. Only a resource pass earns a separate
quality study; inverse stability alone does not justify extensive training.

[Plan](reversible_softsign_plan.md), [reference](../results/reversible_softsign_v1/model.py),
[raw evidence](../results/reversible_softsign_v1/result.json),
[independent audit](../results/reversible_softsign_v1/audit.json),
[receipt](../results/reversible_softsign_v1/receipt.json).
"""
reportpath = Path("research/reversible_softsign_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: reversible scalar reconstruction preflight passed\n\n[H147](reversible_softsign_results.md):36 GPU reconstruction/VJP cases through\n128 layers pass, with independent NumPy gradient checks. Maximum FP32 input\nerror{worst32:.3e}. Closed-form inverse, depth-scaled slope bounds and a\ncontraction-capacity limit are documented. Zero optimizer updates. This earns\na resource implementation study, not a quality/VRAM claim. Goal remains open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = "Latest: [H147 reversible scalar preflight](research/reversible_softsign_results.md)\npasses reconstruction and gradient checks through128 layers. Resource savings\nand learning quality remain unproven; research continues."
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
    Path("research/reversible_softsign_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H147",
    status="EVIDENCE_VERIFIED",
    gate="PASS numerical reconstruction preflight",
    training_updates=0,
    native_backwards=36,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
