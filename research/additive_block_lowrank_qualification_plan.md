# H060: additive block/low-rank qualification

Frozen before numerical qualification or language-quality scoring. H058 is the
mechanism proposal; H059 completed cleanup. This round qualifies one policy,
not a learning-rate search or a claim of novelty. No corpus validation/test is used.

## Fixed architecture and initialization

Each bias-free projection maps n inputs to m outputs using W=S+LR. S has G=8
block-diagonal groups; L is m by r, R is r by n. At model width d divisible by24,
h=8d/3 and r=d/8. Three projections use exactly19d^2/8 weights/layer.
Use ordinary SwiGLU and optional existing native gate recomputation only.

Let sigma=0.02 for up/value and sigma=0.02/sqrt(2*depth) for down. Fix rho=1/2.
Initialize each S block semi-orthogonally with gain

    s = sigma * sqrt((1-rho)*max(n,m)).

Initialize L with orthonormal columns and R with orthonormal rows, both multiplied
by sqrt(g), where

    g = sigma * sqrt(rho*n*m/r).

Use independent name-local CPU generators keyed by seed, layer/projection path
and factor name. The block and global components have exactly (1-rho)*mn*sigma^2
and rho*mn*sigma^2 squared Frobenius norms in exact arithmetic. The cross term
has expectation zero under independent symmetric draws; total norm and individual
row norms are not fixed per draw. No trained-spectrum constraint is imposed.
All factors start nonzero so neither global factor waits for another to learn.

## Fixed update calibration, with a deliberately limited interpretation

Use AdamW with the shared betas, epsilon and clipping, parameter-level decay0.1.
The candidate-specific multipliers for S,L,R are

    aS=sqrt((1-rho)*G),
    aL=sqrt(rho*n/(2*r*g)),
    aR=sqrt(rho*m/(2*r*g)).

Derivation: suppose independent zero-mean entry perturbations in each parameter
have variances eta^2*aS^2, eta^2*aL^2, eta^2*aR^2. For first-order
DeltaW=DeltaS+DeltaL*R+L*DeltaR, E||DeltaW||F^2/(mn)=eta^2, since its three terms
contribute (1-rho),rho/2,rho/2. Cross terms average to zero by perturbation
independence. This uses actual orthogonal factor norms and does not require the
fixed block mask to be random. It is a Frobenius-average calculation, not equality
for each matrix entry or gradient direction. Actual Adam steps are data-dependent
and correlated; this does not prove Adam invariance, equal training speed or
optimal hyperparameters. Do not silently use the existing grouped LR multiplier8.
No extra product-decay correction, trainable branch scale or optimizer tuning.

## Qualification gates

1. Independent dense W reference: FP64 forward, input and every parameter gradient,
   gradcheck/gradgradcheck on small rectangular maps. No dense W in native forward.
2. Actual parameter counts at d24/96/192/384; h=8d/3,G8,r=d/8; reduction70.3125%.
3. Implement H058's quadratic witness in actual allowed tensors. Test FP64 forward,
   Jacobians and Hessians against analytic I,diag(1..d),ones at d24; forward at96.
   This proves the scoped representation only, not learnability or NLL superiority.
4. Seeds17/29/43: component Frobenius energy relative error <=1e-5 at d384.
   Monte Carlo independent perturbations: 256 FP64 draws at a small rectangular
   map, aggregate first-order update energy within10% of the derivation.
5. Native CUDA BF16 versus native FP32 on the same d384/h1024 weights and inputs:
   relative output RMS error <=0.02, input/parameter gradient relative errors <=0.05,
   finite outputs/gradients; native recomputation must equal eager in BF16 for
   initial and deliberately changed weights. FP64 second derivatives remain valid.
6. Integration, if local gates pass: preserve all six retained CPU/GPU signatures,
   actual complete model total9,099,648 and FFN2,801,664 at d384/L8,V4096.
   New configuration and actual optimizer membership must match declared policy.
7. Isolated full-model GPU workers, shared synthetic tokens B16,T128,V4096,
   seeds17/29/43, AdamW20 updates at fixed peak0.0006 using the shared schedule.
   Compare full SwiGLU, GELU, calibrated narrow, plain BlockShuffle and additive.
   Run one GPU worker at a time, native BF16, no compilation. Candidate allocation
   must be <=110% of each full control per seed; all steps/weights/moments finite.
   Report synchronized training time after10 warmup updates, allocated peak,
   preclip norms and clipping. Synthetic loss is diagnostic, never a quality rank.

No automatic retry of completed cells. Record failures and exact source/config,
environment, hashes and per-worker results before combining. Archive existing
source before integration; preserve all historical checkpoints/results. A passing
qualification earns a separately frozen balanced language screen with both full
controls, calibrated narrow and the strongest existing structured reference.

## Primary implementation audit

SLTrain uses random sparse support, a zero output low-rank factor and Kaiming
input factor, alpha/r scaling, and uniform sparse values. Its custom operation
forms a dense weight transiently rather than retaining it for backward. See
[paper sections3.2-3.3](https://arxiv.org/html/2406.02214v2) and the
[pinned primary implementation](https://github.com/andyjm3/SLTrain/blob/2335d62c75df9b0f0441604eb04fde21b06693e4/splora/splora_linear.py).
We read the linear/model wrappers without executing their code; revision and
file hashes are [recorded](../results/verification/additive_prior_art_v1.json).
The block support, both-nonzero orthogonal initialization, separate GEMMs and
candidate-specific update calibration here are disclosed adaptations. The published
method's memory/quality results are not transferred to this experiment.