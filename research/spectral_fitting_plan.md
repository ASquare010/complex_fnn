# H104: small FFNs after supervised feature discovery

H103 is progress: both fixed moment estimators pass every cubic allocation gate
on three independent datasets. Bounded moments recover substantially more of the
hidden subspace than shuffled labels. No learned-model quality follows yet.

## Candidate and proof scope

Test f(x)=D phi(U A x+b)+c, with d=384, rank(A)<=32 and h=128. Every matrix and
bias is trainable; A has no bias because that would be redundant with b. Count:
384*32+32*128+128*384+128+384=66,048 parameters. Phi is either established GELU
or normalized cubic Hermite (z^3-3z)/sqrt(6). This is a factorized input projection,
not a new activation. Cubic has unbounded derivatives; monitor actual gradients.

Initialize A using raw/bounded label-energy eigenvectors or an independent random
orthonormal basis. U has N(0,1/32) entries, b uniform[-1,1], and D/c start at zero.
Thus all models start with the same zero function. Keep A trainable to refine
the finite-sample subspace. The factorization restricts input dependence to32
learned directions, but does not restrict all inputs to a known teacher basis.

An oracle construction with A containing the true directions represents the
cubic task in16 features and pair products using at most64 shifted cubic
features. It is a capacity witness only; training never receives this A.
For quadratic, He3(z+1)-He3(z-1)=6z^2-4. For product, four shifted cubes give
He3(u+v+1)-He3(u+v-1)-He3(u-v+1)+He3(u-v-1)=24uv.
Neither finite GELU sums nor bounded moment estimates inherit an exact global
cubic representation guarantee. No whole-network nonvanishing-gradient theorem.

## Fair controls

Seventeen forms, all trained for the same300 updates and two-rate search:

- Full GELU h1536 and SwiGLU h1024, each random and bounded-spectral initialized.
- Narrow GELU h456, SwiGLU h304 and cubic h456, bounded-spectral initialized.
- Parameter-matched dense GELU/cubic/squared-ReLU h85 and SwiGLU h57, all bounded.
- Factorized GELU and cubic h128/r32, each random, raw and bounded initialized.

Ordinary spectral dense projections start as K B^T, with K entries N(0,1/32).
Their weights subsequently train without a rank constraint. Bias distributions,
zero readouts and name-local seeds match where shapes permit. The same estimator
is available to all strong controls, so initializer gains cannot be attributed
to factorization. Random controls isolate the label-information contribution.
No candidate comparison is based solely on beating a random-initialized baseline.

## Fresh data and compute

Use independent data from H103/H100: input/basis/output seeds
(10800+s,10900+s,11000+s), s in17/29/43. N73728: first65536 training, then4096
rate-selection and4096 reporting. Four H100 target equations and train-only
per-output scaling remain. Each seed changes both data and optimizer sampling;
report these as paired independent datasets/runs, not pure optimizer variance.
Moment estimation sees only the training split. Rank32 and transforms are fixed.
Every spectral recipe pays one full65536-example preprocessing pass; record its
standalone time/peak memory without dividing it by the number of grid runs.
Random controls do not use labels for initialization; label-prepass cost and
extra data access are disclosed. No held-out score selects the estimator.

Exactly17 forms *4 tasks *3 seeds *2 rates =408 fits,122,400 optimizer updates.
Rates0.001/0.003; batch256; independent shared stream seed12000+s. AdamW
betas(.9,.95), eps1e-8, decay0, clip1. Readout matrix rate scales by reference
hidden/actual hidden (reference1536 or1024 for SwiGLU), as in H100. All other
parameters use the base rate. The factorized A is included in this rule's base.
GPU FP32, TF32 off, four CPU threads. H101's BF16 sensitivity motivates keeping
the existing FP32 function-fitting precision. Data stays on CPU; transfer only
the current batch so full-dataset GPU residency does not hide model memory costs.

Use one shared training/evaluation loop. Record end-to-end update time including
batch gather/transfer, model forward/backward time, separate inference latency,
allocated/reserved training peaks, parameter/optimizer bytes, all losses,
pre-clip norms, clipping, activation ranges and per-parameter gradient norms.
Warm up first50 updates before timing summaries. Exclude scoring/export from
training peaks. Record matrix-only FLOPs, excluding activations and optimizer.
Save every final checkpoint; select the rate only on the selection split.

## Qualification and decisions

Before fitting: count every form; prove zero initial output and paired initial
matrices; check folded factor/dense equality for outputs, input gradients and
factor gradients in FP64; numerical gradcheck with nonzero readout; CUDA FP32
forward/backward finiteness; spectral initialization's rank and Gaussian probe
variance; oracle polynomial witnesses on nonzero dense rotations. Tolerance
FP64 rtol/atol1e-10 except gradcheck1e-5/1e-3. Stop on failure and preserve it.

Judge bounded factor GELU and cubic independently. Each must use>=90% fewer
parameters than full controls, halve cubic zero-predictor error on every seed,
stay within1% of both bounded full controls in aggregate geometric-mean MSE,
beat every tiny bounded control and its random/raw ablations by>=2% aggregate,
win every seed aggregate versus each tiny control, and have no task mean>5%
above the strongest tiny control. Training peak allocation must be<=75% of both
bounded full controls; update and inference time<=1.25 times bounded narrow GELU.
These are cheap-screen gates, not proof of the full research goal. Account for
preprocessing separately; no same-total-compute superiority claim without a
budget-matched follow-up. Failed recipes receive no automatic language budget.

Independently reconstruct data, initializers, scores, selections, counts and gates.
All failures and all-rate results stay visible. Any survivor still needs duration,
broader task families, realistic FFN inputs, mixed precision and language-relevant
training/distillation evidence. Label energy is identically constant for ordinary
one-hot LM targets; H104 cannot bypass that applicability limitation.
