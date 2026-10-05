# Wider detectors with a restricted learned readout

Registered on 2026-10-04 before implementation or measurement. The previous
feature-flow campaign completed 22 runs/removals and failed both language
margins and its cross-neuron comparison. This is a new fixed allocation,
not a revised gate or a continuation of tuning that failed recipe.

## Hypothesis and equations

An independent down projection consumes weights that might instead fund more
input detectors. Reuse the input matrix for output directions, but let each
group learn combinations of its directions:

```text
z = P x                                  P: [1152, 512]
r = a * (z * SiLU(z) - mu)
C_g = diag(s_g) + Delta_g                 36 groups of 32 coordinates
FFN(x) = residual_scale * P.T * C * r
```

Each layer has its own P and Delta. Signs s alternate +1/-1 and are fixed,
balanced in each group. Delta starts at zero. The readout's signs allow both
positive and negative contributions at initialization; they do not create a
new detector, prove independence of directions, or guarantee stability.
P starts with normal standard deviation .02 using its named up generator.
Residual scale is 1/sqrt(2*layers). No learned bias, token state, recurrence,
attention change or cross-layer sharing.

The response retains the previous scalar form, not a new activation claim.
For Gaussian z of variance q=width*.02^2, mu=q/2. Set a with 128-node Gaussian
quadrature so response variance is (1376/512)*(width/hidden)*q*E[SiLU(z)^2].
At width 512, this matches the independent-coordinate output-variance budget
of full SwiGLU 1376. Check with 256 nodes. Tying P to the readout correlates
features and directions, so this scalar calculation does not establish full
function, Jacobian or logit matching. Balanced signs cancel an expectation,
not every sample's linear drift.

This allocation gives 1152 learned input rows instead of compact SwiGLU's
408 gate/value pairs or GELU's 612 rows. More rows do not automatically give
more useful features. For response i, its output direction lies in the span
of P's rows in its group; the readout sacrifices freedom. Its Jacobian is
P.T*C*diag(phi')*P times the fixed scale and is generally nonsymmetric, unlike
the fixed diagonal tied control. C follows the scalar nonlinearity and adds
no additional nonlinear interaction depth. It is a constrained ordinary FFN
readout, not feature attention or a memory retrieval algorithm.

2,506,752 FFN weights across four layers: 4*(512*1152+1152*32), 70.35% fewer
than full SwiGLU's 8,454,144. Forward projection work is 9,732,096 FLOPs/token,
4*(4*512*1152+2*1152*32), plus scalar operations. P is used twice: storage
reuse does not erase either product or either gradient contribution.

## Execution and controls

Execution v1 uses ordinary PyTorch operations and nonreentrant checkpointing
of the FFN computation. FP32 response/group mixing, native BF16 linear
projections under autocast, TF32 disabled. Checkpointing re-executes both
projections and the group operations in backward; conservatively account for
4x forward projection work per training token: 38,928,384 FLOPs across layers
plus scalar/reduction work. Cached ordinary autograd uses the same equations
and initialized weights, with 3x projection work. Resource measurements decide
whether storage savings repay recomputation. No custom kernel claim in v1.

Controls, all registered before training:

* Full kernel SwiGLU 1376, fused-projection SwiGLU 1376, GELU 2064.
* Compact kernel SwiGLU 408 and GELU 612: exactly 2,506,752 FFN weights.
* Calibrated compact kernel SwiGLU 408: multiply only down initialization by
  sqrt(1376/408), retaining the same count/work.
* Untied same-response FFN 612: independent P/Q, the same calibrated centered
  response and alternating signs before Q, exactly the primary's weight count.
  This is an ordinary dense FFN comparator too; select the strongest compact
  reference across all four, not just conventional activations.
* Diagonal readout 1152: learn only 1152 diagonal corrections per layer,
  2,363,904 FFN weights. It starts at exactly the primary's function and removes
  cross-direction readout freedom. It honestly has fewer weights.
* Fixed signed readout 1152: no corrections, 2,359,296 weights; identical
  initial function. Width-preserving readout removal control.
* Fixed signed readout 1224: exactly 2,506,752 weights, more detectors but no
  learned readout corrections. This tests whether C earns its weight budget
  rather than allocating it to extra rows. Different width/calibration means
  it is not an exact initial-function match.
* Cached grouped 1152: ordinary autograd, same weights/function/count, no
  required language superiority comparison against the recomputed primary.

Initialization remains named and identical for all unchanged Transformer
weights. Every model directly defines its full block and Transformer and only
reuses shared attention/normalization/training. First stage prototypes stay in
dump/staging until the mathematical and resource checks pass.

## Frozen protocol and decision

No overlapping GPU work. First independent CPU FP64 full block-matrix equations,
all x/P/C gradients, finite differences and gradient checks. Verify initialization
equality of equal-width tied controls, counts/projection work, quadrature,
token locality/causality, save/load and exact CPU AdamW checkpoint recovery.
CUDA FP32/BF16 compares recomputed against ordinary equations at small and
actual [8,256,512] shapes, nonzero off-diagonal corrections, all modes, and all
parameter/input gradients. FP32 tolerances atol 1e-5, rtol 3e-4; BF16 outputs
atol .004, rtol .02 and gradients atol .0001, rtol .05. Record every maximum
error and any errors; never relax a failed check silently.

Then profile all three full dense and six prototype variants on both prepared
language corpora: primary, diagonal, fixed, fixed-matched, untied, cached.
Three rounds, alternating forward/reverse order; width 512, layers four,
heads 16, context 256, vocab 4096, batch eight, BF16, threads four, seed 101,
AdamW lr .0006, weight decay .1, gradient clip one; 20 warmup and 100 measured
updates. Median throughput and maximum allocated memory across rounds. Primary
must have >=1.2x throughput with memory <=1.05x OR >=20% less allocated memory
with update time <=1.05x against every full dense control. Record reserved memory,
compilation/warmup and all failures. A failed execution is retired before
language; a different execution requires a separate preregistration.

Only if the primary passes resources, integrate without mathematical changes
and check staged/production initialized weights, outputs, gradients, counts and
shared-Trainer exact CPU recovery. Freeze complete source and recipes. Run all
12 variants on both corpora, seed 101, 2000 updates / 4,096,000 targets, learning
rate .0006, warmup 200, weight decay .1, same sampled windows and full eligible
validation. Tests remain unopened. Select strongest full and strongest of the
four compact ordinary dense references per corpus. Primary must be <=1% worse
than strongest full and >=1% better than strongest compact on both corpora,
and beat both diagonal and fixed-matched on both. Retire any failure after
finishing all registered controls/removals. No post-hoc control promotion.

Frozen validation interventions for every tied mode: set Delta to zero, retain
only its diagonal, mask even response coordinates before C, and replace the
whole FFN with its training mean estimated from 32 train batches, seed 2027.
For untied/dense controls use even responses and the same mean replacement.
Retain original calibration and signs throughout. Honest no-op interventions
stay recorded. Report learned diagonal/off-diagonal norms. Removal penalties
do not establish semantic roles and can reflect changed statistics.

Only a survivor enters equal lr tuning at .0003/.0006/.0012 for primary, all
three full and four compact references at seed 101/2000 updates. Freeze each
selected recipe, then paired seeds 211/307/401 must repeat both language margins.
Independent seed 509 runs 8000 updates, warmup 800, then opens tests once; require
both margins on full validation and test. Repeat diagonal/fixed/fixed-matched
controls with the primary's rate on paired and independent runs to distinguish
training-path from inference benefit. Finally repeat isolated resources with
100 warmup/1000 measured updates, three alternating rounds, unchanged thresholds.
Freeze the detailed confirmation protocol before any conditional tuning.

## Prior work and risks

[Readout prior-art notes](ffn_readout_prior_art.md) identify Energy Transformer,
SLlama shared-projection MLP, and Causal Energy Minimization. Transpose tying,
gates, grouped matrices, output preconditioning and recomputation are known.
The research question is this weight allocation and its directed grouped
readout versus width/freedom controls under the stated FFN-only contract.
No claim of an undiscovered primitive or mathematical invention. A positive
result would still require a closer implementation comparison for originality.

Risks: restricted value spans, harmful detector/readout gradient coupling,
correlated residual drift, groups without semantic locality, scalar response
limitations, checkpoint overhead and no quality improvement despite greater
feature width. Failure is evidence against this fixed allocation, not against
every form of weight tying or feature breadth. Each execution and the model
family will have a result.md containing quality, speed, memory and decision.
