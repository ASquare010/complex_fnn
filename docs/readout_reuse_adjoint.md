# Shared-projection adjoint and activation-storage audit

An independent algebra audit during execution v1's frozen resource screen.
This changes no live source, recipe, gate or allocation. It is not a second
execution registration or a speed claim.

For one token, use column vectors:

```text
z = P x; r = phi(z); u = C r; y = alpha P.T u.
g = derivative of loss with respect to y.
g_u = alpha P g
g_r = C.T g_u
g_z = phi'(z) * g_r
grad_x = P.T g_z
grad_P = u (alpha*g).T + g_z x.T
grad_C = g_u r.T
```

Batch gradients sum the two outer products over tokens. C is block diagonal,
so only each group's corresponding grad_C block is a learned gradient. Both
uses of P contribute; combining them is essential. A diagonal-only learned
correction takes the diagonal of grad_C. An even-response mask multiplies r
before C and also multiplies its adjoint after C.T. Calibration/signs remain
fixed. Zeroed correction flags must return zero correction gradients; masking
off-diagonal corrections also masks those gradient entries.

For phi(z)=scale*(z*SiLU(z)-mean), with s=sigmoid(z):
phi'(z)=scale*(2*z*s + z^2*s*(1-s)). This is the same derivative as the v1
ordinary operations. The known independent untied control instead has separate
readout Q, grad_Q=g*u.T and only the input contribution to grad_P.

The formulas permit storing projected z and reconstructing r/u in backward,
instead of checkpointing both entire projections. They require x/P/C references
for projection and group gradients, but do not require retaining a token-sized
FP32 nonlinear bank across all layers. Whether this is faster and stays below
the VRAM limit depends on real temporary allocations and dispatches.

These real-valued expressions are not a bit-exact BF16 implementation: forward
projection/output casts, the output gain's rounding, projection adjoint casts,
and separate master-weight contributions must agree with ordinary autocast.
Any later execution must test FP32/BF16 against v1 and the independent equations
at full shapes before profiling. No approximation to backward is authorized.

Preregister a small CPU FP64 audit: eleven tokens, five input coordinates,
eight detectors, two groups of four, arbitrary nonzero corrections, alternating
signs, and with/without even-response removal. Check joint x/P/C finite
differences and each separate variable direction, at epsilon 1e-6. Require
absolute error <=1e-8 or relative error <=1e-6. The tiny sample checks the
equations only; it cannot identify a bottleneck or predict language quality.
Save records/readout-reuse-adjoint.json with every error and the audited source.
