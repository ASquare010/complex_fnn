# Paired-feature FFN

Hypothesis: reuse each projected gate/value pair to produce two useful nonlinear
features, retaining only two dense projections and improving on a narrow FFN
at the same parameter and matrix-compute budget.

The projected vector Ux is split into a and b. A second dense matrix maps one
of these feature banks back to the residual stream:

| CLI variant | Features |
|---|---|
| paired_antipodal_swiglu | [SiLU(a) b, SiLU(-a) b] |
| paired_reciprocal_swiglu | [SiLU(a) b, SiLU(b) a] |
| paired_duplicate_swiglu | [SiLU(a) b, SiLU(a) b], a redundant control |

For projected width h=2r, each FFN has 2dh weights. At d=192, h=228 and four
layers this is 350,208 unique FFN weights and 1,728,192 total weights. It exactly
matches narrow SwiGLU h=152 and GELU h=228 in weights and FFN matrix FLOPs.
The reduction against full SwiGLU is 70.3125% for the FFN, 32.43% for the model.

The antipodal pair has an exact bilinear difference and a smooth gated sum.
Its gate-direction derivative squared norm is at least b^2/2 before the output
projection. This does not guarantee a nonsingular full Jacobian: its determinant
is -b*a^2*sigmoid'(a), and the output projection can cancel features. Extra
features may still be redundant or difficult to optimize.

The [frozen experiment and derivation](../../../../paired_feature_plan.md)
records prior work, limitations, matched controls and promotion thresholds.
[SwiGLU](https://arxiv.org/abs/2002.05202),
[CReLU phase pairing](https://arxiv.org/abs/1603.05201) and
[masked shared projections](https://arxiv.org/abs/2506.23225) are related prior
art. No verified novelty is claimed. Verdict: REJECTED from the frozen LM
promotion screen. At 200 steps, antipodal NLL is 4.320257 and reciprocal NLL
is 4.333520, versus 4.287006 for the duplicate control and 4.192404 for matched
GELU. Their useful scalar identities did not produce the required LM gain.
Read [all results and plots](../../../../paired_feature_results.md).

Common initialization, training, data, evaluation and diagnostics remain in
src/core. The module contains no special optimizer or execution machinery.