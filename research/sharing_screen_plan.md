# Reuse screen — frozen before training, 2026-09-06

Structured GELU seed-17 NLL 4.2241 is only .40% better than narrow GELU 4.2411;
structured SwiGLU 4.6482 is worse than narrow SwiGLU 4.4191. Neither meets the
second-screen promotion criteria. Record these negatives; no larger run.

Next test H006: preserve full dense nonlinear width by reusing the same FFN
weights in each group of four existing Transformer blocks. Attention and
normalization remain independent. Do not add layers, recurrent steps or compute.
Use shared GELU and shared SwiGLU, then compare the original full/narrow controls.
This is an established sharing baseline inspired by ALBERT and
[One Wide Feedforward Is All You Need](https://arxiv.org/abs/2309.01826).
[Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672) is relevant
if strict tying needs layer-specific adaptation. No novelty claim.

For L layers and sharing span S=4, K=ceil(L/S) distinct FFNs:
P_total=Vd + L(4d²+2d) + K*P_FFN + d.
L=4,d=192: one shared FFN has 294912 unique parameters, a 75% reduction in
total FFN parameters. Total model 1672896, down 34.592%. Each invocation
remains full width; forward FFN FLOPs are exactly the full baseline's.
This is not 75% fewer parameters per FFN invocation or lower training FLOPs.

Initialize each shared FFN using the first block's name-derived seed, same
recipe as a full reference. Common attention/embedding tensors must still match.
Optimizer and parameter accounting must deduplicate shared tensors. Gradient
norms of shared parameters aggregate contributions across invocations; do not
mislabel duplicated aggregate norms as per-layer gradient propagation.

First seed 17, 200 steps, otherwise micro_v1. Promote to 800 steps with controls
only if NLL is within 1% of both full references or clearly improves the narrow
frontier without unacceptable speed/memory regression. Three seeds and broader
data remain necessary. Scaling uses groups of four, preserving 75% FFN savings
at layer counts divisible by four; partial final groups are counted exactly.

Seed-17 shared GELU achieved NLL 4.13642, versus full GELU 4.09232 and narrow
GELU 4.24112: a 2.47% improvement over the parameter-matched narrow control,
with full-sequence speed close to the full reference. This meets the prespecified
frontier-improvement promotion route, though the 1% gold quality target is still
missed. Promote shared GELU with full GELU/SwiGLU and narrow GELU to 800 steps,
seed 17 first. This is a separate budget cohort, not a continuation of the
200-step cosine schedule.


The completed full controls at 800 steps change the activation ranking:
full GELU NLL 3.19312, full SwiGLU 3.06505; narrow GELU 3.20449 is within .36%
of full GELU but still 4.55% worse than the stronger SwiGLU reference. Therefore
also evaluate narrow and shared SwiGLU at 800 steps before judging sharing.
This follow-up is motivated by an observed change in baseline ranking, and
receives exactly the same optimizer, seed and token budget.
