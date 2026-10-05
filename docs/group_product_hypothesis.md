# Hypothesis: inexpensive feature-group products

Registered after both sketch hardware screens finished and before implementation
or training. Keep the same active size, quality, resource and confirmation gates.

## Why this experiment

The two-stream and cached single-stream sketches saved more than 20% allocated
memory but exceeded the allowed slowdown. Fewer projection FLOPs did not make
their FFTs and data movement cheap. Their fixed recipes are retired before
language training; their language quality is unknown. The shared-bank language
screen also failed. We need interactions that execute as simple reductions and
elementwise operations, rather than assume algebraic complexity means speed.

## Computation and hypothesis

Project each token to h features z, split them into 32 groups of 16, and form:

```text
z = Ux
s_g = sum(z_j for j in group g)
q_i = z_i * (s_group(i) - z_i) / sqrt(15)
FFN(x) = D (SiLU(z) + tanh(a) * q)
```

Coefficients a are learned per feature in each layer and start at zero. Each
feature can interact with the other 15 in its group, without constructing a
pairwise matrix. Algebraically, q_i is a sum of cross-feature products; the group
sum computes a complete graph's neighbor signal in linear work. Learned U can
place related features in the same group. D learns how to use their conjunctions.
There is no given semantic meaning to the groups; useful organization must be
learned and demonstrated. The map compresses many products into h outputs and
does not preserve every distinction or create information absent from x.

This links graph aggregation, quadratic feature maps and gating. Excluding z_i
from the group sum removes the direct self-square term. `sqrt(15)` approximates
variance scaling for independent features, not exact normalization of correlated
trained activations. The direct SiLU path preserves unary features. Coefficients
may remain unused, or unbounded products may destabilize training; neither is
assumed away. Input normalization and gradient clipping remain the common ones.

## Fixed study

Same width 512, hidden 512, four layers, 16 heads, context 256, vocabulary 4096,
batch eight, BF16. `groups=32` means 32 groups, each containing 16 features.
Independent U/D/a in every layer. FFN weights 2,099,200: 75.17% fewer than full
SwiGLU. Projection FLOPs 4,194,304/token: 75.19% fewer; report reduction/product
work separately. Attention, embeddings and normalization stay fixed.

Controls: plain SiLU hidden 512; square residual `z*z/sqrt(2)` with the same a
count; compact fused SwiGLU hidden 344; compact GELU hidden 512; full fused
SwiGLU hidden 1376; full GELU hidden 2064. Dense compact count differences are
those reported in the sketch hypothesis. The square control's means and feature
statistics differ; it tests an alternative quadratic transformation, not a
perfectly isolated semantic intervention. All paths have identical initial logits
when a=0; check nonzero coefficients and gradients before profiling.

Numerical gate: independent dense group-adjacency equation and gradients, finite
differences, causality, token locality, counts, save/load, exact CPU resume and
finite GPU updates. Then profile group products, plain SiLU and both full dense
controls on both corpora, three alternating-order rounds, 20 warmup plus 100
measured steps. Apply >=20% allocated-memory reduction OR >=1.2x throughput
against both full controls, at most 5% deterioration in the other resource and
less than 6 GiB reserved. Retire before language if the hardware screen fails.

If it passes, integrate the equivalent full Transformer into the simple production
layout, repeat production Trainer resume, then train all seven variants on both
corpora. Seed 101, 2,000 updates, rate 0.0006, warmup 200, AdamW decay 0.1,
identical sampled windows. Require within 1% of strongest full dense NLL and
>=1% better than strongest compact dense NLL on both corpora. Also require gain
over plain and square controls to motivate group interactions. Test loss stays
unscored. No group-size search after viewing these results.

Zero a after training to remove interactions at inference. Compare with the
retrained plain model to distinguish inference dependence from training effects.
Feature removal and training-mean interventions follow the shared-bank procedure.
Survivors require equal tuning, fresh seeds, sustained resource measurements and
independent confirmation. A one-seed gain is a development result only.

## Prior work and originality

[GLU variants](https://arxiv.org/abs/2002.05202) precede multiplicative FFNs.
[Rewrite the Stars](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html)
studies multiplicative feature interactions and compact networks.
[Gated Channel Transformation](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.html)
uses inexpensive channel context and normalization in vision. Polynomial feature
maps, grouped operations and graph aggregation are established ideas. Our proposed
token-local group cross-product residual and its FFN compression recipe have
unverified originality. This is a falsifiable candidate, not an undiscovered
invention or a claim of attention-equivalent reasoning.
