# H106: an affine path removes the rank obstruction at sufficient residual width

The capacity screen qualifies **GELU h256 and SwiGLU h170** for a separate fitting
experiment. The smaller h128/h85 recipes fail the fixed allocation gates. This
is a matrix-approximation result, not a learned model, VRAM saving or language
quality result. The broader research goal remains unmet.

The experiment reuses the 18 independently audited H105 datasets: full GELU
and SwiGLU teachers at depths 0/3/7, with three calibration-window seeds. The
teachers themselves each have one training seed. All 126 rank comparisons and
eight qualification checks pass an independent QR/rectangular-SVD audit.
No neural optimizer update or new GPU capture was needed.

## What the proof establishes

Consider the established model family

\[
f(x)=Wx+b+D\phi(Ux+a).
\]

Only the nonlinear correction has output rank at most h. A full-rank affine path
can preserve directions that H105's global low-rank output bottleneck discarded.
This differs from the retired H058-H061 architecture, which added a low-rank
matrix to each block-diagonal projection inside SwiGLU. Those failed recipes stay
closed; their results are not reinterpreted here.

For a finite dataset, let X1 augment input rows with a column of ones, P project
onto its column space, and R=(I-P)Y. Then

\[
\min_{W,\,\mathrm{rank}(E)\le h}\|Y-X_1W-E\|_F^2
=\sum_{j>h}\sigma_j(R)^2.
\]

Projecting any candidate error onto the orthogonal complement of X1's column
space cannot increase its norm. The projected correction still has rank at most
h, so Eckart-Young gives the lower bound. Taking E as R's truncated SVD and
X1W=PY attains it. This is a consequence of established matrix approximation,
not a new theorem. An arbitrary rank-h matrix E need not be expressible or
learnable by h actual scalar neurons.

The reporting-data affine fit is used **only for this optimistic capacity
bound**. It is not a model initialization. A separate diagnostic fits an affine
map and residual output basis on calibration data, then allows arbitrary oracle
reporting coefficients in that fixed basis. Those coefficients use reporting
labels, so that diagnostic is also not a deployable predictor.

For compatibility with H105's FP32 teacher target, apply the recorded BF16/FP32
drift using the reverse triangle inequality for the lower bound and the ordinary
triangle inequality for the fixed-basis upper diagnostic. Both use H105's
reference output variance. No NLL theorem follows from either quantity.

## The fixed gate and result

Each predeclared width had to meet both conditions: every dataset's corrected
oracle lower bound at most 0.05, and each teacher's mean fixed-basis oracle upper
diagnostic at most 0.05. Passing allocates fitting; failing closes that width.

| Residual family | FFN parameters | Saving vs full FFN | Basis upper, GELU teacher | Basis upper, SwiGLU teacher | Decision |
|---|---:|---:|---:|---:|---|
| GELU h128 | 246,272 | 79.12% | 0.02892 | 0.06393 | Closed |
| GELU h256 | 344,704 | 70.78% | 0.00494 | 0.01122 | Fit separately |
| SwiGLU h85 | 245,930 | 79.15% | 0.05261 | 0.11330 | Closed |
| SwiGLU h170 | 344,020 | 70.84% | 0.01643 | 0.03699 | Fit separately |

The two larger widths also pass the all-dataset lower-bound condition; both
smaller widths fail it. Seven ranks (32/64/85/128/170/192/256) are retained in the
complete result. The larger recipes do not retroactively turn smaller failures
into successes. The 5% threshold is a local resource-allocation choice, not a
necessary condition for the project's language NLL target.

At width d384, an ungated affine residual uses d²+d+h(2d+1) coefficients; a gated
correction uses d²+d+h(3d+2). These counts include biases and all trainable
weights. Whether the extra affine matrix launch is worthwhile remains an
empirical question. The separate [H107 plan](affine_residual_fit_plan.md) specifies
fresh windows, matched controls, uniform initialization, training and resources.

## Prior art and reproducibility

[Wide & Deep](https://arxiv.org/abs/1606.07792) is a longstanding linear-plus-deep
precedent. [Whipp's FFN recoverability study](https://arxiv.org/abs/2606.19379)
directly tests affine reconstruction and residual probes, and cautions that local
reconstruction and language perplexity can disagree. The basic decomposition,
least-squares initialization and low-rank bound are not claimed novel.
[Eckart and Young](https://doi.org/10.1007/BF02288367) supplies the approximation
identity used above.

See the [frozen plan](affine_residual_capacity_plan.md),
[source](../results/affine_residual_capacity_v1/source/study.py),
[complete compressed result](../results/affine_residual_capacity_v1/result.json.gz)
and [independent audit](../results/affine_residual_capacity_v1/audit.json).
The audit recomputes all 126 quantities using QR projectors and direct rectangular
SVD, rather than the screen's covariance eigendecomposition. Local tensor data
and source snapshots remain available in the workspace; a compact clone needs
those raw artifacts to rerun verification.
