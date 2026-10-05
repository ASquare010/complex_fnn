# Registered diagnostic: what a trained full FFN map needs

Registered before any diagnostic activation collection. Execute only after the
current basis-readout language campaign is terminal; no overlapping GPU/CPU-heavy
measurement. This is an explanatory diagnostic, not a new qualifying candidate,
not distillation, and not permission to change the active experiment.

Question: how much of each trained full SwiGLU block's input/output map is affine
on sampled training activations, and how much of that affine map needs more than
128 output directions? Previous weight singular-value energy does not answer
this. The [primary prior work](https://arxiv.org/abs/2606.19379) motivates using
closed-form fits and distinguishes map fidelity from downstream language loss.

Use the completed current campaign's full native SwiGLU 1376, seed 101, 2000-update
checkpoint on each corpus. Read frozen weights into the unchanged dense model.
Collect normalized inputs to each FFN and its outputs in CPU FP32 inference;
report the arithmetic difference from BF16 training. No validation or test
activations, no language-loss scoring, no model training or modifications.

Split the contiguous training token array into two equal halves. Sample 32
context-256 windows entirely inside each half, preserving the existing tokens
and context behavior. Fit on the first half at seed 6001; held-out activation
measurement on the second at 6003. Independently reverse halves at seeds
6007/6011. Windows across each fit/held-out partition do not overlap positions;
near-duplicate text is not guaranteed absent. Save all starting positions and
their hashes. Each set gives 8192 input/output pairs per block. Also fit the
first 16 training windows (4096 pairs) against the same 8192 held-out pairs.

Fit centered affine regression in CPU FP64 using an SVD of the input matrix;
training means supply the intercept. Use pseudoinverse cutoff
eps(float64)*max(matrix dimensions)*largest singular value. Report effective
input rank, extrema/condition number, train/held-out sum-square errors and held-out
variance explained against the held-out mean, plus the mean-only predictor using
the training output mean. No learned probe optimizer. Fixed ridge sensitivity
values 0/1e-8/1e-6/1e-4 times the mean input Gram diagonal; report all values,
do not select a ridge from held-out scores or call an ill-conditioned fit exact.

For every fit, project the fitted training output onto its leading right singular
vectors at ranks 64/128/256/512. Apply those same output subspaces to the fitted
held-out predictions. This is reduced-rank regression on the affine fit, not
truncation of raw weight matrices or an independent nonlinear probe. Retain the
untruncated affine result. Record aggregate R2, per-output R2/variance, quartiles,
worst features and highest-variance features. Report output covariance energy
and whether a few coordinates dominate aggregate fit. Include both sample sizes,
ridge choices, halves and all four layers on both corpora without cherry-picking.

Independent synthetic CPU checks: a noiseless full-rank affine map recovers its
held-out outputs; an actual rank-128 map is retained by rank-128 regression;
controlled quadratic residual is not falsely called an affine fit. Numerical
checks must pass before loading any research checkpoint. Preserve source/data/
checkpoint hashes and save a result.md plus compact records.

Interpretation limits: high map fidelity is not evidence of preserved language
quality; low output rank is not evidence that input information can be discarded.
Results may motivate a separately registered hypothesis about a full-width
affine path and smaller nonlinear correction. Such paths already have mathematical
precedents and must meet the same size/resource/quality controls. This diagnostic
cannot tune the currently frozen basis-readout architecture or override failures.
