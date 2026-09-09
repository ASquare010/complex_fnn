# H018-H020: paired nonlinear features - frozen first screen

The previous goal turn made progress: it completed three-seed BlockShuffle
replication and established its remaining quality and runtime failures. The
next candidate targets useful nonlinear features with two ordinary dense
projections, avoiding a new chain of small grouped matrix operations.

## Mechanism and exact accounting

Let z = U x = [a, b], with a and b each of width r. A learned dense output
projection D maps 2r features back to model width d. Test three feature maps:

- H018, antipodal: [SiLU(a) * b, SiLU(-a) * b].
- H019, reciprocal: [SiLU(a) * b, SiLU(b) * a].
- H020, duplicate control: [SiLU(a) * b, SiLU(a) * b].

The duplicate control collapses to ordinary SwiGLU of width r with output
weight D_left + D_right. It has redundant learned coefficients and altered
aggregate optimizer dynamics, so it is a capacity diagnostic rather than an
identical-optimizer reimplementation of the narrow control.

Each candidate has 4dr = 2dh weights per layer, where h=2r is the projected
and output-feature width. Choose h=228, r=114, d=192, L=4: 350,208 unique FFN
weights and 1,728,192 total weights. This exactly matches narrow SwiGLU h=152
and GELU h=228 in parameters and FFN matrix FLOPs. It reduces both FFN quantities
by 70.3125% relative to full SwiGLU. No extra learned coefficients, factor-aware
learning rates, checkpointing or output caches are added.

All pair modes and matched GELU share identically initialized U and D tensors
through the existing name-derived generator. Use the original dense Gaussian
initialization, uniform AdamW learning rate, residual-output scaling, data,
precision and decoder. Pair modes change only the nonlinear feature map.

## Structural claims, with explicit limits

Write s(t)=t*sigmoid(t). Since s(t)-s(-t)=t,

f_plus - f_minus = a*b,
f_plus + f_minus = a*b*tanh(a/2).

Thus the antipodal feature pair spans an exact bilinear feature and a separate
smooth gated feature, without relearning a second gate/value projection.
For fixed b, s'(a)+s'(-a)=1 gives

||d[f_plus,f_minus]/da||_2^2
    = b^2 * (s'(a)^2 + s'(-a)^2) >= b^2/2.

This is a scalar gate-direction bound before D. It becomes uninformative at
b=0; D can cancel features, and the full input Jacobian can be singular.
Indeed its 2-by-2 feature Jacobian determinant is -b*a^2*sigmoid'(a).
There is no claim of a globally nonvanishing Transformer gradient.
The reciprocal pair supplies two directed nonlinear products, but shares its
quadratic Taylor term; extra feature count alone does not prove independence
or improved language modeling.

## Prior work and novelty boundary

[SwiGLU](https://arxiv.org/abs/2002.05202) supplies the gated primitive.
[CReLU](https://arxiv.org/abs/1603.05201) already uses paired positive/negative
phases for parameter efficiency. [MGLU](https://arxiv.org/abs/2506.23225)
shares a projection through complementary learned masks, a different mechanism
with related goals. This signed/shared-projection experiment has no verified
priority claim. The identities above are elementary algebra, not new theorems
of general network superiority. A narrow search did not establish novelty.

A reread of [BTT](https://arxiv.org/abs/2406.06248) confirms that rectangular
factorization and normalization remain important stronger controls. We defer
its implementation this round because more intermediate tensor movement does
not directly resolve the measured runtime gap; this is a hypothesis about our
implementation setting, not a negative BTT experiment.

## Bounded experiment sequence

1. Independently verify the feature identities, first/second derivatives,
   parameter/FLOP equality, common initialization and CUDA backward. Run an
   actual tiny-batch overfit test for the two candidates.
2. Reuse the existing interaction benchmark at 300 steps, seed 17, on all
   three polynomial/additive/product targets. Pair width 6 exactly matches
   narrow SwiGLU width 4 in learned parameters. Include the duplicate control.
   Require finite gradients and successful fitting before LM promotion; these
   selected targets are diagnostics, not general capability evidence.
3. Run five 200-step LM screens at seed 17: all pair modes, narrow SwiGLU
   h=152, and matched GELU h=228. Preserve every run.
4. Promote at most one pair mode to an 800-step check if it improves NLL by at
   least 1% against matched SwiGLU, stays within 1% of matched GELU, and avoids
   catastrophic memory/gradient behavior. The output must also improve on the
   duplicate control. Later evaluation must retain the same token budget for
   candidate and controls. If it fails, record the mechanism and move on.
5. A useful 800-step result must earn three seeds and equal-execution serving
   comparisons. No candidate is accepted from these screens alone. The original
   quality, parameter, runtime, scale and broader-corpus requirements stay open.