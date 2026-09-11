# H106: can an explicit affine path remove the real-FFN rank obstruction?

Previous goal turn: progress. H105 established that the fixed global rank-64
output class cannot meet its local reconstruction target. The broad VRAM/quality
goal is unmet. This screen checks capacity before allocating any neural training.

## Distinct question

Consider f(x)=Wx+b+D phi(Ux+a), with a full d-by-d affine path and h nonlinear
features. At d=384, h=128 and GELU this costs 246,272 parameters including biases,
79.12% fewer than the 1,179,648-weight full FFN. A SwiGLU correction with h=85
costs 245,930 parameters. The affine map can be full rank; only the correction
lies in a fixed low-dimensional output subspace. This is an established additive
linear/nonlinear model family, not a novel activation claim.

H058-H061 instead use block-diagonal-plus-low-rank maps inside each of three
SwiGLU projections. Their failed language recipe stays closed. H106 does not
alter those projections or reinterpret their measurements.

## A finite-data capacity bound

For paired rows X and Y, append a column of ones to X to form X1. Let P project
orthogonally onto col(X1), and R=(I-P)Y. For any coefficient matrix W and any
arbitrary rank-at-most-h prediction correction E,

    min_{W, rank(E)<=h} ||Y-X1 W-E||_F^2
      = sum_{j>h} sigma_j(R)^2.

Proof: project the error onto col(X1)'s complement. Rank((I-P)E)<=h, giving the
SVD-tail lower bound. It is attained by E equal to the best rank-h truncation
of R and W an ordinary least-squares solution for PY. This is a consequence of
orthogonal projection and Eckart-Young, not a new theorem. Arbitrary E need not
be representable or learnable by h GELU/SwiGLU neurons, so passing is necessary,
not sufficient. The bound is local to the sampled teacher function, not NLL.

Use the H105 captured reporting matrices. Fit the oracle affine projection on
reporting data only to compute this optimistic lower bound; no model is trained
or initialized with that fit. The deployment/prototype affine fit, if later
allocated, must use calibration data only. Correct BF16-label bounds conservatively
to FP32 targets with H105's reverse-triangle correction. Divide by the same
recorded FP32 output variance. This enables a like-for-like comparison with H105.

Separately fit an affine map and a residual output PCA basis on calibration data
only; report held-out error after an oracle arbitrary residual correction in that
fixed basis. This is another capacity diagnostic, not a deployable predictor:
the arbitrary residual values use teacher reporting outputs. It identifies whether
training-only output directions are competitive with the fully privileged bound.

## Fixed screen and allocation gate

Reuse all 18 audited H105 datasets: two seed-17 pretrained teachers, depths 0/3/7,
calibration seeds 17/29/43. These are sampling replications, not independent
teacher training seeds. Calibration/reporting rows remain 32,768 / 8,192.
Ranks 32,64,85,128,170,192,256. Predeclared fitting candidates are GELU h128/h256
and SwiGLU h85/h170. The larger budgets still meet the original 70% parameter
target: GELU h256 costs 344,704, SwiGLU h170 costs 344,020. Each is evaluated
independently; a larger candidate never converts a failed smaller recipe to a pass.

No SGD, language validation or new teacher collection. CPU FP64 SVD/QR is suitable
for the small output dimension and avoids unnecessary GPU allocation. Record
wall time, data hashes and all spectra; raw tensors remain local.

Before the real-data screen, verify the bound on exactly affine targets, a known
orthogonal residual construction, a planted affine-plus-rank-h matrix and a
rank-deficient design; compare QR/SVD/lstsq projections, not normal equations.

Each predeclared affine-plus-feature family earns a separately frozen fitting comparison
only if its corrected bound is <=0.05 on every dataset and the training-only
residual-basis diagnostic has mean error <=0.05 for each teacher. Otherwise close
this fixed width before training. Smaller/larger ranks show the frontier and do
not retrospectively replace the gate. Passing does not promote a model. The 5%
threshold is a conservative local allocation choice, not a necessary condition
for the actual language NLL target; reconstruction and perplexity can dissociate.

## Prior art and verification

[Wide & Deep](https://arxiv.org/abs/1606.07792) is an established linear-plus-deep
model precedent, with a different recommendation task and feature design.
[Eckart and Young](https://doi.org/10.1007/BF02288367) gives the rank-approximation
identity used in the bound. Read H105's compression precedents as well.
[Whipp 2026](https://arxiv.org/abs/2606.19379) directly studies exact affine FFN
recovery, bilinear residual probes and the distinction between local reconstruction
and downstream perplexity. That precedent prevents a novelty claim for the basic
decomposition and cautions against presenting this local gate as an NLL theorem.
No claim of novel architecture, improved training, GPU speed or measured VRAM
reduction follows from this diagnostic. Independently check the real-data
projectors and tails with a different decomposition before allocating fitting.
