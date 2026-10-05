# Conditional sketch v2: one feature stream and a cached spectrum

Registered after sketch v1's terminal hardware screen, before v2 implementation
or training. The fixed size, quality and resource targets are unchanged.

V1 saved roughly 24% allocated memory versus full GELU, but repeated timings
showed about 20% lower throughput. Its eight real FFT calls per forward/backward
FFN invocation and repeated gathers were too costly. Retire that fixed recipe
before language training; preserve its equations, prototype and measurements.

## Simpler hypothesis

```text
z = Ux
v = sign * z[permutation]
V = FFT(v)
q = inverseFFT(V * V) / sqrt(2h)
FFN(x) = D (SiLU(z) + tanh(a) * q)
```

Save V in the autograd operation, with its full FP32-complex precision. In
backward, reuse it rather than recompute transforms of two feature streams:

```text
dv = 2 * inverseFFT(FFT(dq) * conjugate(V)) / sqrt(2h)
dz = inverse_permute(sign * dv)
```

This uses four real FFT calls per forward/backward invocation versus v1's eight,
and one signed permutation. Caching costs additional activation memory. It may
trade enough of v1's memory margin for speed to pass both resource gates.

The feature map also changes: products use one signed feature stream, so
symmetric pairs collide together. The `sqrt(2h)` scaling approximately matches
variance for independent zero-mean features; it is not an exact normalization
for trained correlated features. The map retains many pair products but loses
independent sketching streams. No unbiased polynomial-kernel theorem is claimed.
It may capture fewer useful distinctions; reduced execution cost is not quality.

## Fixed experiment

Use the v1 backbone and size: width/hidden 512, four layers, 16 heads, context
256, vocabulary 4096, batch eight, BF16; 2,099,200 FFN weights and 75.17% fewer
than full SwiGLU. Use the same full and compact dense controls. Additional
controls are the plain SiLU path and a squared-feature path `v*v/sqrt(2)` in
place of convolution, with the same learned coefficient count and initial zero
coefficients. The square control's feature means differ; report that confound.
The retired two-stream sketch's quality remains unknown.

Before GPU profiling, verify the new adjoint against direct convolution and
finite differences, and verify full-Transformer causality, gradients, initial
control equivalence, save/load, counts and exact CPU resume. Profile single sketch,
plain SiLU and both full dense controls on both corpora: 20 warmup plus 100
measured steps, three alternating-order rounds. Use maximum allocated peak and
median throughput. Require >=20% less allocated memory OR >=1.2x throughput
against both full controls, no more than 5% deterioration in the other resource,
and less than 6 GiB reserved. Retire before quality if it fails.

Only if hardware passes, integrate the numerically equivalent prototype into
the simple production model layout and repeat production resume checks. Then
train seven variants (single sketch, square, plain SiLU, compact SwiGLU h344,
compact GELU h512, full fused SwiGLU h1376, full GELU h2064) on both corpora,
2,000 updates, seed 101, rate 0.0006, warmup 200, AdamW decay 0.1. Compare all
completed eligible validation windows. Require the active 1% full-model margin
and 1% compact-model advantage on both corpora; no test-loss selection. Also
require improvement over plain and square controls to justify convolution.

Zero learned sketch coefficients after training for an inference-removal test;
compare with the retrained plain control to separate inference dependence from
training effects. Survivors still need equal tuning, fresh seeds, sustained
resource measurements and independent confirmation. Prior art and the limits
of originality are those in [sketch v1](polynomial_sketch_hypothesis.md).
