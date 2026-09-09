# H099: learn nonlinear residuals, not affine trends

Previous goal turn: progress. H096-H098 verified the matched-sample affine
control and closed the prior lead's general claim. The gold target is unchanged.

## New task distribution and positive control

Draw 73,728 independent N(0,I) inputs of dimension 384, seed 9860. Use 65,536
training, 4,096 rate-selection and 4,096 reporting rows. A fixed permutation
(seed 9861) selects 16 input coordinates q. A fixed orthonormal 16-by-384 output
map (first 16 rows of a QR orthogonal matrix, seed 9862) hides the latent outputs.
Targets are scaled by training-only standard deviation without empirical centering.
Compute targets in FP64 and store FP32. The four 16-dimensional latent targets are:

1. quadratic: (q^2-1)/sqrt(2);
2. cubic: (q^3-3q)/sqrt(6);
3. product: q_i*q_(i+1 mod 16);
4. piecewise: [ReLU(q)^2-1/2-sqrt(2/pi)*q]/sqrt(5/4-2/pi).

Each has zero population mean, zero population linear projection onto x and unit
population variance. Gaussian moments prove the polynomial statements. For the
piecewise target E[ReLU(Z)^2]=1/2, E[Z ReLU(Z)^2]=sqrt(2/pi), and its residual
variance is 5/4-2/pi. Finite sample covariances need not be exactly zero.
These are deliberately small latent problems, not real-data or language benchmarks.
Quadratic/cubic tasks favor their corresponding polynomial bases; no universal
conclusion may be drawn from such a target set.

The feature-aware positive control receives the exact 16 latent nonlinear
features and learns only a zero-initialized 16-to-384 readout. It has privileged
task information and is not a fair candidate comparator. An oracle readout must
first reconstruct generated targets, and gradient training of this small model
must reduce reporting MSE by at least 95% on every task. This validates the
feature/readout pipeline, not the difficulty of discovering the input directions.
Also require a full conventional neural control to halve zero-predictor error on
at least three tasks before interpreting a candidate as promising.

## Candidate: one bank with shared parity mixing

For each preactivation z, let s=z/(1+abs(z)), E=z*s, O=z*s^2, and

    phi_g(z) = E(z) + a_g*O(z),    a_g=2*tanh(theta_g).

Use four activation groups, initialized at a_g=1. This replaces two independent
feature readout banks with a single mixed bank, allowing 456 projected features
at the compressed budget. There is no raw-input matrix or duplicated readout.
The four scalars learn the odd/even balance; they do not give independent readouts
for both bases. Fixed a=1 and fixed a=0 isolate the shape-learning contribution.
E and O have derivative magnitude at most one, hence |phi'|<=3 for |a|<=2.
This is an activation-level bound, not a whole-network gradient guarantee.

All neural forms have ordinary learned projection/output biases. This is a
deliberate stronger baseline than the recent bias-free single-layer assays;
shifting an even function can create a nonlinear odd input response. Biases are
counted separately from activation-shape parameters. All down weights and output
biases start at zero, so every neural model starts with the same zero function.
Other matrices use name-local Gaussian initialization, std 1/sqrt(384); hidden
biases use a separate name-local Uniform[-1,1] initialization. Varied thresholds
avoid the population stationary point that zero readout plus zero hidden bias
can create for a centrally even activation on a pure odd target. This initialization
is shared by all comparable forms and is an explicit changed recipe. Learned and fixed parity versions share initial weights.

Neural forms: full GELU h1536, full SwiGLU h1024, full squared-ReLU h1536;
narrow GELU/squared-ReLU/StarReLU/even-rational/fixed-mix/learned-mix h456;
narrow SwiGLU h304. Add the feature-aware positive control: 11 forms total.
StarReLU uses learned global scale/bias initialized at 1/0. Those parameters
can be absorbed into the biased readout and do not enlarge that function class.
[Primer](https://arxiv.org/abs/2109.08668) uses squared ReLU;
[StarReLU](https://arxiv.org/abs/2210.13452) is established prior art.
The rational mixture also builds on ordinary softsign/products; no novelty claimed.

## Qualification before training

Freeze source, this plan and H098 result anchors. Run exactly 25 checks once:
11 actual parameter/zero-output counts; three FP64 activation formula and numeric
gradient checks (rational, squared-ReLU, StarReLU); one rational parity/derivative
bound check; four feature-aware oracle checks; five CUDA whole-FFN checks for the
new activation forms; one common-zero-initial-function check.

FP64 formula tolerances 1e-11; gradcheck eps1e-6, atol1e-5/rtol1e-3. GPU tests use
nonzero down weights, transposed 2x3x384 probes, FP32 atol/rtol1e-5 and BF16 0.02,
FP32 activation arithmetic with one cast and FP64 preservation. Non-reentrant
checkpoint gradients must match exactly. Save paired tensors and recheck them.
Oracle agreement with FP32 stored targets uses atol/rtol5e-6. Stop on failure;
no automatic retries, numerical tolerance relaxation or hidden training allocation.

## Fixed fitting screen

Four tasks, seeds17/29/43, rates0.001/0.003, 300 updates, batch256: 264 runs,
79,200 neural updates and 20,275,200 presentations. The teacher-aware readout's
updates are included in that total. Same seed-specific sample streams for every
form; rotate execution order. Select rates only on the separate selection split.

AdamW betas(.9,.95), eps1e-8, decay0, global gradient clip1, constant LR. Retain
the narrow down-weight LR multiplier reference_hidden/actual_hidden; all other
parameters use base LR. Feature-aware readout uses base LR without this multiplier.
One GPU, four CPU threads, FP32 CUDA, TF32 off; same precision as previous screens.
Log loss, preclip norms/clipping, activation/gradient samples, learned coefficients,
synchronized forward/backward/update timing after50 warmups, and peak GPU bytes.
Count matrix FLOPs from matrices, excluding biases and activation work explicitly.

Fit a fresh affine least-squares control to each seed's sampled training multiset
using the audited GPU-statistics/CPU-small-solve method. No held-out label enters
the solve. Report these 12 fits separately from gradient updates and timings.

## Prespecified decision for the learned mixture

Require finite complete evidence, qualified oracle/feature-aware controls, and
the full-neural learnability gate above. Then require at least70% fewer total
FFN parameters than full GELU and SwiGLU; aggregate reporting MSE ratio<=0.98
against every narrow control, fixed mixture, even-only mixture and affine fit;
every seed better than each control; no task mean more than5% above its strongest
control; aggregate ratio<=1.01 against both full references; and mean median
update time<=1.25 times narrow GELU. Known-feature control is excluded from fair
candidate comparisons. Squared-ReLU full remains an additional strong comparator.

If learnability gates fail, label the assay inconclusive rather than rewarding
a model merely for predicting zero. Otherwise failed gates close this recipe at
this budget. No automatic refinement, language run, active model integration or
breakthrough claim. Audit data, selections, all checkpoint scores and summaries
independently before publishing a conclusion. Keep raw tensors local and compact
machine-readable evidence in Git.
