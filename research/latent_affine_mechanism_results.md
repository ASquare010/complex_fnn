# H084 - What the internal affine control learned

The internal affine control depends on both learned gain and offset, but its
gain is algebraically redundant with the first weight factor. Removing gain
without compensation multiplies aggregate held-out MSE by 2.972; removing offset
multiplies it by 1.843. Folding the affine operations preserves predictions
within the declared FP64 and FP32 tolerances. This is a checkpoint-mechanism
result, not a new trained model, causal training explanation or speed result.

All 12 selected H083 affine checkpoints pass. Independent verification exactly
reproduces the 24 new gain-only/offset-only resets and their prediction hashes.
There are zero training updates, no new rate selection, and no earned language
or full-model resource trial. H083's curves remain rejected; its affine control
still trails narrow GELU by 6.512% aggregate held-out MSE.

## Algebra and the limited expressivity claim

For a column-vector projection, write

    f(x) = P_out^-1 B2 P_mid (G B1 x + b).

G repeats each learned group gain g=1+0.5*tanh(theta_a) on that group's
first-factor outputs. The vector b repeats 0.5*tanh(theta_b) within each group.
Left-multiplying B1 by G preserves B1's block-diagonal structure and parameter
shape. Therefore gain alone adds no represented functions when B1 is trainable.
It can still affect optimization through a different parameterization.

There are two ways to preserve an already trained function in real arithmetic:

1. Replace B1 by G B1, set gain controls to zero, and retain the offset.
2. Replace B1 by G B1, remove the affine module and add the cached output vector
   c=P_out^-1 B2 P_mid b after the ordinary two-factor projection.

The first removes 24 effective gain degrees of freedom from one FFN; the
diagnostic copy retains their zeroed parameter slots. The second removes all
48 curve parameters and stores 4,480 bias-buffer values across up/gate/down:
17,920 bytes in FP32. It has 350,208 factor parameters plus those buffers.
This preserves the evaluated function, not Adam states, gradients with respect
to a changed parameterization, or its original optimization trajectory. It does
not prove lower runtime, BF16 fidelity or zero deployment storage overhead.

An isolated bias-free SwiGLU with zero-preserving differentiable projections has
F(0)=0 and DF(0)=0, by the product rule and SiLU(0)=0. H083's tanh/sine curves
also preserve zero and retain this restriction. Internal offsets can remove it:
the verified width-one witness U(x)=1/4, V(x)=x and D(x)=x has
F(x)=SiLU(1/4)*x, with strictly positive slope at zero. The actual grouped-factor
implementation reproduces this witness.

This is a local statement at the origin, already related to the earlier H037
and H072 analyses. It is not a whole-Transformer gradient guarantee, a
distributional approximation bound, a proof that every learned Jacobian has
full rank, or a claim that bias is a new mechanism. The
[frozen derivation](latent_affine_mechanism_plan.md) gives its assumptions.

Featurewise affine modulation appears in [FiLM](https://arxiv.org/abs/1709.07871);
our parameters are not outputs of a conditioning network. Bias-only adaptation
appears in [BitFit](https://arxiv.org/abs/2106.10199), which studies fine-tuning
pretrained models rather than training these structured layers from scratch.
The gated baseline follows [GLU variants](https://arxiv.org/abs/2002.05202).
Those papers do not establish the present checkpoint result or our gold target.

## Resetting is different from folding

Ratios are geometric means of paired errors over all 12 task/seed checkpoints,
or over the three seeds within one task. Values above one mean worse error.
Projection weights stay fixed for resets; gain folding changes the first factor
to compensate exactly in real arithmetic.

| Intervention | Aggregate MSE / original | Smooth | Oscillatory | Multiplicative | Piecewise |
|---|---|---|---|---|---|
| Original affine | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Remove gain | 2.971783 | 4.451566 | 1.169175 | 2.137393 | 7.011183 |
| Remove offset | 1.843307 | 5.629432 | 1.001029 | 1.097342 | 1.866969 |
| Remove both | 2.451589 | 6.925839 | 1.205165 | 1.546081 | 2.799238 |
| Fold gain into first factor | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Fold gain and cache output bias | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The two reset effects are not additive. Removing both can be less harmful than
removing gain alone, as in the multiplicative and piecewise tasks. These are
coadapted checkpoints, so a reset is not equivalent to training a gain-only or
offset-only model. The results cannot identify which training mechanism caused
H083's 15.03% advantage over plain. Such a claim requires separately controlled
training and must still beat narrow GELU.

## Numerical folding checks

All 24 original/folded CPU FP64 comparisons pass for output and input gradient
on the fixed 9x384 probe. The maximum absolute output difference is
3.552714e-15; the maximum input-gradient difference is 3.794708e-19.
The tolerance was rtol=atol=1e-10, fixed before execution.

Across both folding modes and all checkpoints, the largest relative FP32
held-out MSE change is 1.621767e-08, below the frozen 1e-5
limit. Original and reset-both scores reproduce H083 exactly. FP32 folding
changes arithmetic order, so bitwise equality is not claimed. No BF16,
compiler, latency, training-gradient or full-model resource qualification was run.

## Response at the origin

For each checkpoint, eight fixed FP64 input directions probe the derivative at
zero. Removing offset produces exactly zero output and JVPs in every case,
whether gain is retained or reset. The table describes finite probes; it does
not certify Jacobian rank or gradient behavior away from the origin.

| Task | Seed | Original output norm at zero | Original eight-JVP norm | After removing offset |
|---|---|---|---|---|
| smooth | 17 | 22.327619 | 38.557533 | Both exactly zero |
| smooth | 29 | 21.877105 | 39.066148 | Both exactly zero |
| smooth | 43 | 22.183989 | 39.774090 | Both exactly zero |
| oscillatory | 17 | 0.058449 | 2.152411 | Both exactly zero |
| oscillatory | 29 | 0.064232 | 2.130044 | Both exactly zero |
| oscillatory | 43 | 0.049380 | 2.062534 | Both exactly zero |
| multiplicative | 17 | 0.645213 | 16.665156 | Both exactly zero |
| multiplicative | 29 | 0.657252 | 16.861950 | Both exactly zero |
| multiplicative | 43 | 0.692860 | 16.888943 | Both exactly zero |
| piecewise | 17 | 2.732204 | 36.539920 | Both exactly zero |
| piecewise | 29 | 2.746489 | 36.762693 | Both exactly zero |
| piecewise | 43 | 2.719967 | 36.840053 | Both exactly zero |

## Mean mismatch and remaining error

For residual e=prediction-target, squared error decomposes into

    E[||e||^2]/d = ||E[e]||^2/d + E[||e-E[e]||^2]/d.

The following entries are arithmetic averages over the three seeds. Reductions
use FP64 residuals from FP32 predictions. No target centering or refitting was
performed; this is a descriptive decomposition of the same reporting errors.

| Task | Mode | Residual MSE (FP64) | Mean-error energy | Centered error |
|---|---|---|---|---|
| smooth | Original affine | 0.518223 | 0.005162 | 0.513061 |
| smooth | Remove gain | 2.307400 | 0.608408 | 1.698992 |
| smooth | Remove offset | 2.917414 | 1.900333 | 1.017081 |
| smooth | Remove both | 3.591511 | 2.333075 | 1.258436 |
| oscillatory | Original affine | 1.003295 | 0.000265 | 1.003030 |
| oscillatory | Remove gain | 1.173029 | 0.001004 | 1.172025 |
| oscillatory | Remove offset | 1.004327 | 0.000257 | 1.004070 |
| oscillatory | Remove both | 1.209140 | 0.001134 | 1.208006 |
| multiplicative | Original affine | 0.918279 | 0.001238 | 0.917041 |
| multiplicative | Remove gain | 1.962825 | 0.006769 | 1.956056 |
| multiplicative | Remove offset | 1.007666 | 0.000264 | 1.007402 |
| multiplicative | Remove both | 1.419855 | 0.001108 | 1.418747 |
| piecewise | Original affine | 0.548499 | 0.002652 | 0.545847 |
| piecewise | Remove gain | 3.845675 | 0.058679 | 3.786996 |
| piecewise | Remove offset | 1.024032 | 0.019771 | 1.004261 |
| piecewise | Remove both | 1.535397 | 0.028966 | 1.506430 |

The decompositions reproduce from saved per-coordinate residual moments. Their
sums match the FP64 error to the declared 1e-12 tolerance and the original
FP32-accumulated MSE to 1e-6. Mean correction and remaining functional error
must both be considered; improvements cannot be labeled complex-pattern
learning solely from aggregate MSE.

## Every checkpoint intervention

The interaction is E(no gain,no offset)-E(no gain,offset)-E(gain,no offset)
+E(gain,offset). It is descriptive and uses the fixed trained weights.

| Task | Seed | Original MSE | Remove gain | Remove offset | Remove both | Factorial interaction |
|---|---|---|---|---|---|---|
| smooth | 17 | 0.518193 | 2.314410 | 2.909853 | 3.562306 | -1.143764 |
| smooth | 29 | 0.519482 | 2.362243 | 2.889989 | 3.447189 | -1.285561 |
| smooth | 43 | 0.516995 | 2.245547 | 2.952400 | 3.765039 | -0.915913 |
| oscillatory | 17 | 1.003284 | 1.172454 | 1.004347 | 1.209624 | +0.036107 |
| oscillatory | 29 | 1.003331 | 1.170571 | 1.004325 | 1.204924 | +0.033360 |
| oscillatory | 43 | 1.003268 | 1.176062 | 1.004310 | 1.212871 | +0.035767 |
| multiplicative | 17 | 0.917727 | 1.934777 | 1.007124 | 1.394053 | -0.630121 |
| multiplicative | 29 | 0.918822 | 1.977932 | 1.007909 | 1.428735 | -0.638283 |
| multiplicative | 43 | 0.918289 | 1.975765 | 1.007966 | 1.436778 | -0.628664 |
| piecewise | 17 | 0.548478 | 3.867843 | 1.023288 | 1.543599 | -2.799054 |
| piecewise | 29 | 0.548872 | 3.822383 | 1.024523 | 1.526203 | -2.771832 |
| piecewise | 43 | 0.548149 | 3.846797 | 1.024284 | 1.536388 | -2.786544 |

## Allocation, verification and next decision

The fixed H083 reporting split has 4,096 vectors per task, width 384, scored in
batches of 256. Six modes on each of 12 selected checkpoints give 72 passes and
294,912 example presentations. The study worker took
16.62 seconds, including CPU FP64 probes and GPU FP32 scores.
Independent verification repeated the two new reset modes: 24 passes and
98,304 examples in 8.00 seconds, with exact MSE and prediction hashes.
All original checkpoints, sources, data, selections and completed reports remain
unchanged. Sixty input checkpoint/metadata files are pinned by hash, size and
mtime; the H083 archive anchors remain intact. All 131 scientific sources and
73 frozen plans are retained. There are no numerical retries or training updates.

This establishes a representational distinction between gain and offset and an
evaluation simplification. It does not promote a candidate. A distinct future
training experiment would need gain-only, offset-only, full-affine and
fixed-function controls, both narrow baselines, equal budgets and a material
gain. The old outer-affine failures and H083 curve failures remain closed.
The active tree stays at three model folders, six variants and nine recipes;
the full research goal remains unmet.

[Frozen plan](latent_affine_mechanism_plan.md),
[all observations](../results/latent_affine_mechanism_v1/result.json),
[independent audit](../results/verification/latent_affine_mechanism_analysis_v1.json),
[preceding fitting result](latent_activation_recovery_results.md).
