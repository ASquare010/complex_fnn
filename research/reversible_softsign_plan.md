# H147: reversible softsign residual — inverse and conditioning preflight

Previous turn: progress. H146 closes eager/checkpointed modulation stacks.
New hypothesis: use globally invertible scalar geometry, so internal activations
can be reconstructed instead of retained. This is a numerical/theory preflight,
not a quality or GPU-memory claim. RevNets and rational normalizing flows are
prior art; no novelty claim. Earlier centered pair twists remain rejected.

Define phi_a(x)=x+a*x/(1+abs(x)), a=rho*tanh(theta), rho=c/L<1.
Layer y=phi_a(Qx+b), Q orthogonal, per-channel theta and bias. Each scalar
slope lies in [1-rho,1+rho]. The stack has global lower/upper Lipschitz constants
m=(1-rho)^L and M=(1+rho)^L. This bounds input-Jacobian singular values,
not parameter gradients, optimizer trajectories or floating reconstruction error.

For v=abs(y), B=1+a-v, D=sqrt(B^2+4v), inverse magnitude is
2v/(D+B) for B>=0, (D-B)/2 otherwise; restore sign(y). Use hypot for D and
safe branch denominators. This avoids cancellation in the positive quadratic root.
Inverse is used under no_grad; no differentiable-inverse API is promised.

Capacity limitation: for independent iid X,X' and any globally m-co-Lipschitz F,
E||F(X)-EF(X)||^2 = E||F(X)-F(X')||^2/2 >= m^2 E||X-EX||^2.
Consequently a constant target cannot be fitted below m^2*d for isotropic input.
More generally, for target G with Lipschitz constant k<m, reverse triangle gives
E||F(X)-G(X)||^2 >= (m-k)^2*d. Any external unconstrained projection or scaling
can evade this statement and must be analyzed separately; this is not a universal
obstruction to reversible models or feature extractors.

Freeze seeds131/149/167, depth8/32/128, c0.5/2, d16, batch32, FP32/FP64:36 GPU
stress cases, no optimizer updates. Fixed orthogonal matrices are dense numerical
references, not an efficient mixing implementation. Record matrix buffer bytes.
Per-channel theta uniform[-1.5,1.5], biases normal sd.1; standard Gaussian input
and independent output cotangent. Compare ordinary autograd VJP with backward
that reconstructs each layer from its output and uses the analytical local VJP.
GPU results must have finite outputs/reconstructions/gradients; relative input
reconstruction and concatenated input/parameter gradient error <=1e-4 FP32,
<=1e-10 FP64. Report every case, no retries or threshold tuning. One native
backward per case; analytic reverse runs under no_grad and performs no autograd.

CPU preflight checks scalar inverse on signed logspace[1e-12,1e12] plus0,
a=-.95,0,.95 in both dtypes using relative-scaled error <=1e-6/1e-12;
FP64 input/shape gradcheck of phi; independent quadratic residual certificate.
For each GPU case, independently check reconstructed input through forward and
small d16 full Jacobian singular values against [m,M] at one sample. Analytical
Jacobian products require no backward. Native GPU output is saved to CPU along
with inputs, parameters, VJPs and errors for independent CPU replay.

No speed/memory promotion until numerical correctness passes, then only a new
fair resource comparison can evaluate reconstruction's compute/memory tradeoff.
Retain source hashes and H146 receipts; do not change maintained modules/defaults.
References: https://arxiv.org/abs/1707.04585 and
https://proceedings.mlr.press/v108/dolatabadi20a.html.
