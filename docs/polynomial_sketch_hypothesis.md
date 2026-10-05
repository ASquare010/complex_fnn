# Next conditional hypothesis: a compact bank of feature products

Registered before implementation or training on 2026-10-03. This is a proposed
next experiment; the current shared-bank campaign must finish before any new
GPU work or production source edits. Implement mathematical checks in staging
first. It does not replace the active FFN acceptance contract.

## Evidence and the change in direction

The smaller shared bank passed the short hardware screen, but its completed
TinyStories result is 2.603695 NLL versus full SwiGLU 2.538486 and compact SwiGLU
2.626184. The 2.57% full-model loss cost and 0.86% compact gain miss both targets.
WikiText and the untied control remain pending. Those results are not assumed.

Earlier local feature exchange did not establish an inference benefit, and the
new gate-transport implementation exceeded the memory target. Rather than
assume more interactions help, test a different allocation of the feature bank:
represent many pair products in a compact sketch while retaining a direct path.

## The mathematical connection

For projected features z, an explicit outer product contains h squared pair
products. Storing that tensor is impractical. Circular convolution collects many
products into each of h channels:

```text
q_k = sum_i v_i w_(k-i mod h) / sqrt(h)
v = sign_1 * z[permutation_1]
w = sign_2 * z[permutation_2]
q = inverseFFT(FFT(v) * FFT(w)) / sqrt(h)
FFN(x) = D (SiLU(Ux) + tanh(a) * q(Ux))
```

Two fixed independently drawn signed permutations rearrange the learned feature
coordinates. Each output channel combines many feature conjunctions. Learned
input features and output maps may discover useful combinations despite collisions.
The direct SiLU path preserves individual features; coefficients a start at zero.
This is a mathematical route from polynomial feature maps and signal processing
to neuron interactions. It does not add information absent from x, preserve all
pair products, or guarantee better language modeling. Collisions can destroy useful
distinctions. Unlike a CNN over an image, there is no given spatial locality among
latent channels. The model must learn a useful representation for this constraint.

The fixed permutations are not unrestricted random CountSketch hashes. Do not
claim TensorSketch's unbiased kernel approximation theorem for this variant.

## Memory and backward computation

Use an explicit autograd operation for the convolution only. Save z and the fixed
signs/permutations, then recompute Fourier transforms in backward. For upstream
gradient t, the derivatives are circular correlations:

```text
dv = inverseFFT(FFT(t) * conjugate(FFT(w))) / sqrt(h)
dw = inverseFFT(FFT(t) * conjugate(FFT(v))) / sqrt(h)
dz = inverse_permute(sign_1 * dv) + inverse_permute(sign_2 * dw)
```

This avoids retaining several complex Fourier intermediates at every layer.
It trades backward work for memory. Count both, and compare actual end-to-end
time. Perform FFT calculations in FP32, cast the result to the feature dtype,
and verify its gradients against direct convolution in FP64. This custom backward
is an implementation choice, not a new mathematical operation or novelty claim.

## Fixed experiment and failure criteria

If shared-bank v2 does not pass, prototype this at width 512, four layers, 16 heads,
context 256, vocabulary 4096, batch eight, BF16. Hidden 512 keeps transforms a
power of two. FFN weights: `L(2dh+h)` = 2,099,200, 75.17% fewer than full SwiGLU.
Projection FLOPs are 4,194,304/token, 75.19% fewer than full; include FFT and
pointwise arithmetic separately. Every layer's U, D and a are independent.
Attention, embeddings, norms and the residual backbone stay fixed.

Controls: plain SiLU hidden 512 (the exact direct path without the interaction),
signed-permuted pointwise product instead of convolution (one pair per feature,
same signs/permutations and scale matched by variance, no FFT), compact SwiGLU
hidden 344, compact GELU hidden 512, full fused SwiGLU hidden 1376 and full GELU
hidden 2064. Compact SwiGLU has 2,113,536 FFN weights, 0.68% more than the sketch;
compact GELU has 2,097,152, 0.10% fewer. Report those small differences explicitly.

Numerical gate: direct equation and gradient agreement, token locality, causality,
counts, finite CUDA updates and exact resume. Check the custom gradient at nonzero
coefficients; zero initialization alone can hide errors. If these fail, repair
before profiling. Profile sketch, direct-path control and both full controls on
both corpora, three alternating-order rounds, 20 warmup plus 100 measured steps.
Reject above 6 GiB reserved. Apply the unchanged resource screen against both full
controls: 1.2x throughput OR 20% lower allocated peak, with at most 5% deterioration
in the other resource. Retire before language training if this fails.

Only after the hardware gate, train all seven variants on both corpora: seed 101,
2,000 updates, rate 0.0006, warmup 200, AdamW decay 0.1, identical sampled windows.
Require within 1% of the strongest full dense loss and at least 1% better than the
strongest compact dense loss on both corpora. The interaction must also improve
over its direct-path and pointwise-product controls to justify the mechanism.
Do not use test losses for selection.

For frozen trained weights, zero a and score full eligible validation. Compare
with the retrained direct-path control to distinguish inference dependence from
training dynamics. Removing the fixed permutations after training changes the
feature coding and is not a clean ablation; the trained pointwise control is the
registered comparison for pooled versus individual products. A survivor still
requires equal rate/seed tuning, longer repeated resource measurements and frozen
independent confirmation. No single seed qualifies a replacement.

## Prior work and originality

[Fast and Scalable Polynomial Kernels via Explicit Feature Maps](https://www.rasmuspagh.net/papers/tensorsketch.pdf)
introduced efficient TensorSketch constructions in 2013.
[Compact Bilinear Pooling](https://openaccess.thecvf.com/content_cvpr_2016/html/Gao_Compact_Bilinear_Pooling_CVPR_2016_paper.html)
applies compact polynomial feature representations in computer vision.
[Polynomial Tensor Sketch](https://proceedings.mlr.press/v119/han20a.html) further
uses FFT sketches to approximate element-wise nonlinear functions of low-rank
matrices. These precede the main mathematical idea. This is a proposed use of a
signed-permutation product sketch as a residual feature primitive inside a
language-model FFN. That placement and training recipe have unverified originality;
they must not be described as an undiscovered invention. A useful contribution
requires controlled evidence of better capacity at the stated cost.
