# H105: does label-energy feature discovery transfer to actual FFNs?

Previous goal turn: progress. H103/H104 produced audited synthetic evidence and
closed two fixed recipes. The full VRAM/quality goal remains active. This is a
distinct, cheap diagnostic on pretrained language FFNs, not automatic training
of the rejected cubic model and not a new activation claim.

## Question and candidates

Can bounded teacher-output energy select a small input subspace that retains more
of a real FFN's function than input PCA or activation-aware linear compression?
Gaussian population identities do not hold automatically on RMS-normalized
language representations. Constant-norm token labels are not used: labels are
the frozen FFN's vector outputs. All teacher/calibration costs are recorded.

Center training inputs/outputs with means mx/my. Let S be the input covariance,
H its symmetric square root and T its inverse square root. Eigenvalues are
floored at 1e-6 times the largest; report the number floored and condition ratio.
For an orthonormal B in whitened coordinates, encode A=T B and decode E=B^T H:

    x_hat = mx + (x-mx) A E.

PCA and unwhitened methods instead use A=B, E=B^T. Teacher output projection C
is always the rank-r PCA basis of centered training teacher outputs. Evaluate

    f_hat(x) = my + (f(x_hat)-my) C C^T.

Methods (all fixed before the screen): input PCA; input weight SVD using the
stacked up/gate matrix L; activation-aware linear response (top eigenspace of
H L^T L H); bounded centered-output energy in original coordinates; bounded
energy in whitened coordinates; bounded energy of the full affine-fit residual
in whitened coordinates; and shuffled-label whitened energy as a null control.
Energy matrices select the largest algebraic eigenvalues, matching H103.
Residual fitting uses training-only ridge 1e-6 times mean input variance.

The full affine baseline uses all 384 inputs. It reports how much output variation
is already linear; it is not counted as a small-model success. No held-out
targets, loss or task/layer names enter any basis calculation.

## Algebra and scope of proof

For teacher up matrix W, the projected first affine map has weights W E^T and
bias W(mx-E^T A^T mx). The common encoder is A^T. The gate uses the same encoder.
For down matrix D, use C^T D followed by C, with final bias my-C C^T my. This
factorization exactly implements f_hat over real arithmetic, including means.
It needs no d-by-d projector at deployment; all means are folded into biases.

At rank r, SwiGLU has 2dr+3hr+2h+d stored parameters; GELU has 2dr+2hr+h+d.
At d384, r64 and the original full widths (1024/1536), these are 248,192/247,680
versus 1,179,648 weights in each teacher FFN: about 79% fewer. Parameter reduction
does not prove lower peak training VRAM; hidden activations retain full width.

With nonsingular S and no flooring, Z=(X-mx)T has covariance I. Truncating
H L^T L H minimizes squared first-linear-response error among these rank-r
whitened projectors, by the truncated SVD variational identity. This says
nothing about optimal nonlinear output error. Flooring changes that objective.
Whitening can amplify perturbations; no whole-model or nonvanishing-gradient
guarantee is asserted. Double-precision forward/input-gradient equivalence,
rank-full identity, covariance reconstruction and a small linear optimum witness
must pass before collecting the real screen.

## Frozen data and budget

Teachers: the existing audited 3,200-update seed-17 full GELU and full SwiGLU
checkpoints under results/ungated_duration_v1/runs. No teacher retraining.
Analyze zero-based layers 0, 3, 7 to cover early/middle/late depth.
Three calibration-sampling seeds 17/29/43 are not three independently trained
teachers. Preserve this limitation in every generalization claim.

For each seed, randomly permute complete nonoverlapping 128-token windows from
the existing WikiText-2 training cache. First 256 windows: 32,768 calibration
tokens. Next 64: 8,192 reporting tokens, disjoint from calibration. The teacher
previously trained on this corpus. Reporting is an internal compression assay,
not a language test set or evidence of unseen-domain generalization. No rate,
rank or method is selected using reporting values. Neither validation nor the
official test tokens are used in this screen.

Capture BF16 teacher execution, with full model weights FP32 and batch 8 windows;
omit only the final vocabulary projection when collecting internal pairs.
Matrices/means use CPU FP64. Evaluate projected FFNs in FP32, TF32 off, comparing
to fresh FP32 teacher outputs on captured inputs, and report BF16/FP32 target
drift separately. Rank grid 32/64/128; all seven methods; two teachers; three
layers; three sampling seeds: 378 fixed local comparisons, zero SGD updates.
CPU-resident pairs, one GPU at a time, four CPU threads. Record collection and
estimation time/peak allocation, model parameter bytes and inference timing at
batch 256 after warmup. Report pipeline peak including the resident teacher,
not merely compressed inference allocation. This is not end-to-end LM latency.

## Allocation gates

Rank 64 is primary; 32/128 show the tradeoff and cannot rescue a failed primary
recipe after selection. Each of the three energy candidates (unwhitened,
whitened, residual-whitened) is considered separately. To earn a new fitting
experiment it must, for BOTH teachers:

1. Have mean reporting squared error / teacher centered output variance <=0.05.
2. Be at least 5% better in paired geometric-mean error than BOTH input PCA and
   activation-aware linear response, and at least 5% better than the shuffled null.
3. Have no layer's three-seed mean normalized error above 0.10.
4. Produce finite outputs and pass the local factorization checks.

These are cheap allocation gates, not the project's final language/VRAM gates.
Failure closes this fixed initializer/projection recipe, not all possible learned
low-rank FFNs. If no method earns fitting, do not spend training on this branch.
Publish all ranks, methods and failures; independently verify stored bases,
factorized evaluation, data hashes and reporting scores.

## Prior work

[ASVD](https://arxiv.org/abs/2312.05821) and
[SVD-LLM](https://arxiv.org/abs/2403.07378) establish activation-aware low-rank
compression and whitening precedents. [IO-SVD](https://arxiv.org/abs/2605.15626)
adds output sensitivity and rank allocation. The linear-response control here
is a simple stated algebraic baseline, not a reproduction of those complete
systems or a SOTA comparison. H103's spectral/Stein precedents still apply.
