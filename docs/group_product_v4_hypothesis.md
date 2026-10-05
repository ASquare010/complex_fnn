# Group-product v4: fuse the activation and its derivatives

Registered after v3's terminal hardware screen and before this implementation.
V3 saved 23% memory versus full GELU but missed the slowdown limit against full
SwiGLU on TinyStories. Retire its exact execution recipe before language training.
The useful lesson is implementation overhead, not a demonstrated quality failure.

Keep hidden 512, 32 groups of 16 and the v1 feature-group equation. Fuse direct
SiLU, the learned tanh coefficient, neighbor product and addition into one
elementwise CUDA operation, using PyTorch's bundled runtime compiler if available.
There is no new third-party dependency or change to attention/backbone. Separate
group reductions remain ordinary PyTorch operations. Save projected features,
the coefficient vector and the small group sum, rather than full q/neighbor arrays.

For incoming output gradient t and c_i=tanh(a_i), define R=sqrt(15):

```text
f_i = SiLU(z_i) + c_i*z_i*(S-z_i)/R
local_i = t_i*(SiLU'(z_i) + c_i*(S-2*z_i)/R)
pooled_i = t_i*c_i*z_i/R
da_i = t_i*z_i*(S-z_i)*(1-c_i*c_i)/R
dz_i = local_i + sum_j(pooled_j) within the group
```

The gradient of a sums over token/batch positions only. Compute group sums and
fused activation/gradient arithmetic in FP32 for BF16 projected features, then
cast the activation/input gradient to BF16. This reduces rounding relative to the
earlier native BF16 sequence; FP64 CPU equations remain identical. Numerical
comparisons must use a reference with the same FP32 accumulation and final casts.

First test runtime-compiler availability and exact fused elementwise calculations;
retire this implementation route if unsupported. Verify forward/gradient behavior
against independent dense group adjacency and ordinary autograd at nonzero a,
including CUDA FP32 and BF16 reference tolerances, causality, counts, save/load and
CPU exact resume. Record maximum numerical differences. No higher-order gradient
support is claimed. CPU uses the same equations without runtime CUDA compilation.

Profile the fused full Transformer, plain SiLU and both full dense controls on both
corpora: three alternating-order rounds, 20 warmup plus 100 measured updates.
Warmup includes kernel compilation; record cold compilation separately before a
training-time claim. Apply the unchanged >=20% memory OR >=1.2x throughput screen,
<=5% deterioration in the other resource, <6 GiB reserved, maximum allocated peak
and median throughput. If this fails, retire before language; no accepted near miss.

If it passes, integrate a readable complete Transformer in one production file,
verify production equivalence/resume, then run the seven v1 variants on both corpora
with the same seed 101, 2,000 updates, rate 0.0006, warmup 200 and decay 0.1.
The full/compact language margins, plain/square controls, coefficient-zero inference
removal, equal tuning and independent confirmation still apply. Test loss stays
unscored. This is a kernel/precision implementation of the proposed primitive,
not evidence that the primitive improves quality or has verified originality.

Before any FFN profiling, add an engineering control: fuse full SwiGLU's SiLU and
gate multiplication and their derivatives using the same runtime compiler.
Include both its existing fused up/gate GEMM version and this fused-activation
version, plus full GELU's already native unary kernel. The resource gate must
pass against all three full controls; use the fused-activation compact SwiGLU
in the subsequent quality screen. Verify this control's mathematical equation
and gradients independently. This avoids attributing a kernel-only gain to the
candidate's feature interactions. This addition precedes all v4 FFN measurements.
Include the ordinary full fused-up/gate SwiGLU as an eighth language variant too;
select the strongest full language reference across all three full controls.
Use FP32 activation arithmetic for the square residual control as well, matching
the fused candidate's numerical path before casting back to BF16. These choices
precede all v4 language training and do not change the observed hardware decision.
