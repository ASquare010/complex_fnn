# Candidate ledger

Current allocation after H087: only full/narrow dense controls and plain
BlockShuffle remain active. Native rational BlockShuffle is retired. Earlier PROMISING/IMPLEMENTING
labels are historical and do not override [the current shortlist](CURRENT_STATE.md).
Discarded source and all results remain [archived](archive/README.md).

States and verdicts apply to the recorded setting, not an entire research family.

| ID | Question / decisive control | State |
|---|---|---|
| H001 | Shared bounded cubic versus narrow GELU, identical initialization | ELIMINATED from 200-step promotion; no credible three-seed gain |
| H002 | Eight curve groups versus one and narrow GELU | ELIMINATED in this setting; worse than narrow in 3/3 seeds |
| H003 | Does a static same-coordinate cubic bank expand the function class? | ELIMINATED as an expressivity claim by algebraic collapse |
| H004 | Distinct coordinates/intervals versus extra same-coordinate curves | IMPLEMENTED through H034; distinct coordinates, no LM promotion |
| H005 | Grouped wide projections versus equally sized narrow controls | ELIMINATED from the 200-step promotion screen |
| H006 | Reuse a full FFN across four existing layers | NEEDS_INVESTIGATION as a strong prior-art control; fails gold target at 800 steps |
| H007 | Compressed multiplicative interactions versus narrow SwiGLU | HYPOTHESIS, investigated through H008/H011 |
| H008 | Fixed cross-group input mixing restores missing products | ELIMINATED for LM promotion; constructive proof holds, additive toy control fails |
| H009 | Layer-specific affine scalar gates specialize shared SwiGLU | ELIMINATED from 800-step promotion; tiny NLL change with overhead |
| H010 | Spend the remaining parameter allowance on wider shared SwiGLU | ELIMINATED as a gold result; h608 still misses quality and costs more compute |
| H011 | Optional projected-value coupling preserves the original family | ELIMINATED for LM promotion; strict expressivity separation proved, toy optimization inconsistent |
| H012 | Two-factor structured projections mix features before nonlinearity | PROMISING compressed reference; factor-aware LR essential; quality and runtime gold gates unmet |
| H013 | Match AdamW shrinkage of the represented factored projection | ELIMINATED from promotion on seed 17; NLL 3.118254 versus 3.115322 |
| H014 | Spend the remaining parameter allowance on BlockShuffle width 1,024 | PROMISING; three-seed NLL 3.108014, beats matched narrow in 3/3 seeds; gold target unmet |
| H015 | Recompute only the SwiGLU gate during backward | VALIDATED for the measured memory reduction; complete training is not bitwise identical |
| H016 | Pack up/value factors to reduce matrix launches | ELIMINATED as a speed-gate solution; eager median 0.780 and corrected V2 graph 0.679 times dense |
| H017 | Apply CUDA graph replay equally to candidate and full reference | VALIDATED for fixed-shape correctness; factorized relative speed still fails the gate |
| H018 | Antipodal gates expose an exact bilinear feature with two dense projections | ELIMINATED from LM promotion; product fitting improves, but duplicate control has better NLL |
| H019 | Reciprocal products reuse both roles of each projected pair | ELIMINATED from LM promotion; fails matched GELU and duplicate controls |
| H020 | Duplicate features separate useful nonlinear diversity from redundant output weights | VALIDATED algebraic collapse; its better short-run NLL motivates an optimizer control |
| H021 | Calibrate dense FFN output variance and down-projection learning rate as width shrinks | VALIDATED as a stronger conventional control in three seeds; no architectural novelty |
| H022 | Direct bounded quadratic Bezier activation | ELIMINATED in tested direct/calibrated screens; loses to matched GELU |
| H023 | Three quadratic branches with learned pair products | ELIMINATED in tested screen; function gains did not transfer to LM |
| H024 | Shared/grouped quadratic correction of calibrated SwiGLU | ELIMINATED for promotion; only .095%/.061% gains over calibrated SwiGLU |
| H025 | Transfer locked compressed recipes to a trained model with doubled width and depth | VALIDATED for locked larger-model quality/memory in three seeds; fused serving passes on seed17, broader goal remains open |

| H026 | Apply default fullgraph compilation equally after training | ELIMINATED as the relative-speed solution; numerics pass, compiled candidate/full ratio .632 |
| H027 | Raise weak singular directions in square down factors after training | VALIDATED for post-training projection conditioning: .1 floor passes larger seed17 and all three smaller seeds; full orthogonalization loses quality |

| H028 | Fuse paired factors, activations and permutations into four inference kernels | VALIDATED for measured larger-checkpoint serving; compiled fused/full1.206x, fused/native1.780x, numerical/memory checks pass |

| H029 | Replicate the locked larger-model recipes in two additional seeds | VALIDATED locally; all three seeds pass both full-reference quality limits and beat calibrated narrow |
| H030 | Check previously unscored validation windows after model selection | VALIDATED within this corpus; all tail gates pass across three seeds, no second-corpus claim |

[Initial report](first_screen_report.md), [longer screen](second_screen_report.md),
[interaction proof and experiments](interaction_report.md),
[structured three-seed results](blockshuffle_results.md), and
[execution and ablations](blockshuffle_progress.md).
No mechanism above has verified novelty or met all research acceptance gates.
Preserve failures; compare stronger existing methods before expanding the search.

## H031: Combine the fixed condition floor and fused execution

**Status: VALIDATED locally.** All six original/floored larger-checkpoint cases pass numerical checks. The fixed floor improves worst down conditioning by 190x to 1,136x; combined prefix/tail NLL cost stays below 0.011%, with all baseline quality gates preserved. The [result](joint_conditioning_results.md) separates weight-intervention quality from kernel fidelity. An import-time native failure and explicit unchanged continuation are preserved. No training, new speed or whole-network stability claim.

## H032: Transfer under equal new tuning budgets to WikiText

**Status: ELIMINATED from 800-step promotion in this setting.** All twelve trials completed. Selected BlockShuffle NLL 5.970604 loses to calibrated narrow 5.912434, full SwiGLU 5.907694 and full GELU 5.879278; both 1% full-reference limits fail. Finite/memory gates pass. The [complete report](wikitext_screen_results.md) preserves all rates and boundary winners. The official test remains unscored.

## H033: Published multi-head FFN as a near-budget comparator

**Status: IMPLEMENTING.** The [standalone reference](archive/retired/src/multihead_ffn/README.md) follows the published architecture equations, retains normalization epsilon and has 350,976 weights per layer at the proposed small budget. CPU forward/gradient checks and a scoped routing-Jacobian upper bound are available. It now has registered normal/calibrated initialization controls under H045; it remains without a controlled LM result or flash kernel and does not establish a new primitive.

## H034: Learn shifted quadratic activation shapes on the compressed front-runners

**Status: ELIMINATED from the frozen promotion screen.** Both shifted-coordinate
variants lose to their selected base and fail memory. Distinct coordinates
implement H004 but do not establish a language-model gain. See [all twelve
activation trials](learnable_activation_results.md).

## H035: Learn a positive-denominator rational activation residual

**Status: PROMISING for quality, failed frozen memory promotion.** Rational
BlockShuffle NLL 5.898003 improves over its selected unchanged base by 1.216%
and calibrated narrow by 0.244%, passing both full-reference quality limits.
Only 320 weights are added across eight layers. Eager peak 883.17 MiB fails
memory. Narrow rational's 0.055% gain is too small for promotion. All four
selected reset interventions are complete; rational BlockShuffle's correction
can be zeroed with only 0.00562% NLL cost, so final shape complexity is not an
established explanation of its training gain. One seed, short budget, prior art.

## H036: Compile the rational activation/product to reduce training memory

**Status: VALIDATED for isolated memory; ELIMINATED as the current training backend.**
The [four-worker execution audit](activation_execution_results.md) reduces native
rational peak from 884.35 to 713.81 MiB and passes numerical gates. The [full
200-step repeat](activation_training_repeat_results.md) nevertheless degrades
NLL from 5.898003 to 6.011564 (+1.925%) while passing memory. Sampling RNG,
initial NLL, warmup parameters and the training-loop AST match. This supports
further numerical investigation, not longer training with this backend.

## H037: Is the rational gain mostly learned affine calibration?

**Status: PASSES the frozen short screen.** [Six trials](affine_activation_results.md)
select affine BlockShuffle at NLL 5.894765, 1.270% below its selected base and
0.055% below native rational, with just128 added weights. Native peak 718.08 MiB
passes both memory limits. Narrow affine loses. The rational affine-energy fit
is 96.388% on actual training-prefix inputs, versus 99.768% on the uniform grid.
Resetting the affine BlockShuffle correction costs only 0.001251% NLL. This
supports the simpler training recipe without proving a curvature-based cause,
independent replication, convergence or universal data adaptation.


## H038: Compile only the rational correction

**Status: PASSES the isolated execution audit.** Retain native SiLU and
gate-product backward while fusing the correction. The
[frozen diagnostic](activation_correction_execution_plan.md) tightens global
gradient fidelity to relative L2<=0.0001 before memory profiling. H036 remains
a failed full-training backend; a local pass does not supersede that result.
[H038 measured](activation_correction_execution_results.md) global gradient
relative L2 3.95e-8, exact sampled logits and717.31MiB peak. The qualified H040
[full-training repeat fails](activation_correction_repeat_results.md), so this
local numerical improvement is not sufficient training-fidelity evidence.


## H039: Does affine BlockShuffle retain its gain at 800 steps?

**Status: VALIDATED for its frozen one-seed gates.** All five 800-step trials
complete. Affine NLL 4.749982 improves 1.707% over selected plain BlockShuffle
and passes both full-reference quality/memory gates, earning H041. The
[report](affine_activation_longer_results.md) preserves the preflight accounting
failure and corrected, unchanged-computation launch.


## H040: Does correction-only fusion preserve the full training result?

**Status: FAILED full-training fidelity and quality gates.** The
[repeat](activation_correction_repeat_results.md) finishes at NLL 5.949104,
+0.8664% versus native rational, despite passing memory at 714.70 MiB.
Initial NLL, warmup parameters, optimizer groups, final sampling RNG and source
archives agree. The locally tiny derivative error did not preserve the measured
training trajectory. Both compiler boundaries remain unqualified for the
promoted longer native-architecture comparison.

## H041: Replicate the native affine recipe on WikiText

**Status: VALIDATED for the frozen three-seed comparison.** All ten new trials
complete; mean affine NLL 4.754207, 1.755% below selected plain BlockShuffle and
2.986% below full SwiGLU. Every gate passes in every seed. The
[report](affine_activation_replication_results.md) qualifies selected-seed
statistics, different recipe rates and slower descriptive training throughput.

## H042: Remove the learned activation after training

**Status: VALIDATED for all three frozen checkpoints.** Correction removal
costs .0058%-.0199% NLL, preserves every common weight and all quality gates.
The [deployment copies](affine_activation_removal_results.md) earn a separate
fused-inference audit. This is not a training-mechanism proof.

## H043: Complete the same-rate two-by-two control

**Status: FAILED two-rate robustness.** The [completed control](affine_activation_rate_control_results.md)
adds precisely plain LR .0012 and affine LR .0006 at 800 steps, seed17. It tests
the unresolved rate confound before gain/bias variants or corpus transfer.

At the same LR0.0012, affine helps only 0.0193% in seed 17; at LR 0.0006 it helps
0.337%, but both low-rate cells are worse than the high-rate cells. This qualifies
the larger H041 selected-rate advantage. Do not assign it to activation learning.

## H044: Replicate the stronger plain rate

**Status: COMPLETE; activation material-benefit gate FAILED, stronger plain control VALIDATED locally.** Two new seeds under the [frozen protocol](affine_rate_replication_plan.md)
complete the three-seed LR0.0012 plain/affine comparison. No model implementation
or other training setting changes. This corrective control takes precedence
over additional activation variants or transfer justified by H041's rate-confounded gain.

The [corrected result](affine_rate_replication_results.md) gives plain mean NLL
4.759029 and affine 4.754207. Affine helps only 0.101% and wins 2/3 seeds;
exploratory paired interval crosses zero. Plain retains 70.3125% FFN reduction
and passes every full/narrow quality/memory gate in every seed. H041 remains
a historical selected-recipe pass, with its activation-specific interpretation
qualified. Additional affine transfer/gain/bias variants are not promoted on
that confounded advantage. No GPU cohort remains running.

## H045: Qualify the published multi-head comparator

**Status: VALIDATED for CPU/GPU pipeline qualification only.** Two initialization controls are
registered without changing the 28 prior model snapshots. Gaussian output
scale is restored to 1.11925 times full SwiGLU by the explicit calibration.
Twelve focused tests and the 213-test full suite pass. Both GPU cases complete
at 832.10 MiB peak with finite/nonzero router gradients. The
[verified result](multihead_integration_results.md) earns a separately frozen
training screen. No validation/test data was scored, and no prior-art performance
or novelty conclusion follows yet.

## H046: Screen both multi-head initializations on validation

**Status: ELIMINATED for longer training at this local recipe.** The [frozen six-trial plan](multihead_screen_plan.md)
gives each initialization three rates and reports all same-rate controls.
The first baseline worker failed during PyTorch import before evaluation or
candidate training; a source-identical continuation uses multihead_screen_v2.
The failure and diagnosis remain archived. No partial measurement is reused.


H046 completed all six trials. Both initializations fail quality and memory
gates; calibrated best NLL 6.027381 versus narrow 5.912434, at 832.58 MiB.
See the [verified result](multihead_screen_results.md). No optimizer extension
or published-scale conclusion follows. H047 separately tested the simpler router-free
headwise control under its [frozen plan](headwise_screen_plan.md).

## H047: Exactly matched router-free headwise SwiGLU

**Status: ELIMINATED for longer training at this local recipe; memory component retained.**
The [verified result](headwise_screen_results.md) completes all three fixed rates.
Best NLL is 6.021328, 1.923% above full SwiGLU, 2.416% above full GELU and 1.842%
above narrow. Quality fails. At exactly 2,801,664 FFN weights, training peak is
638.37 MiB, 23.3% below parallel; both full-control memory gates pass.
All 30 older models remain exact. GPU qualification and all trials complete
without retries; the full suite passes 220 tests. Report generation had an
import-only failure followed by a recorded unchanged continuation.

Retain the [budget proof](multihead_budget_geometry.md) and
[fixed-coordinate interaction limit](archive/retired/src/multihead_ffn/headwise.md).
This changes head width and removes routing together; it is not an isolated
router or activation ablation. Both winners at the upper rate boundary remain
unbracketed. No extra rates, new activation shapes or longer run is promoted.
The next evidence priority is equal-budget optimizer/convergence testing for
the stronger plain model and full/narrow controls. No GPU cohort is running.


## H048: Equal extended global-rate search at 800 steps

**Status: VALIDATED for the local selected-recipe gates; lower boundary remains open.**
The [verified result](optimizer_bracket_results.md) completes all eight new
800-step trials without retries or numerical failures. All four recipes retain
LR 0.0012. Plain NLL 4.750900 is 2.653% below full SwiGLU, 2.320% below full GELU
and 2.933% below calibrated narrow. All fixed quality/memory gates pass. At the
same LR 0.0048, plain loses to narrow; the effect is not rate-independent.

All 223 tests pass and 30 prior model snapshots remain exact. The
[decay-only accounting](optimizer_rate_geometry.md) quantifies how global rate
changes the represented-map contraction; it does not prove the cause of the NLL
change. No architectural novelty or convergence follows. A separately frozen
lower-rate check is next; existing independent seeds for unchanged selected
recipes remain historical evidence and need no redundant retraining.


## H049: Close the lower boundary at 800 steps

**Status: VALIDATED for the local four-rate gates; convergence remains open.**
The [verified result](optimizer_lower_results.md) completes all three missing
0.0006 dense-control cells without retries or numerical failures. The sixteen-cell
grid retains 0.0012 for every recipe, now an interior grid point. Plain remains
2.653% below full SwiGLU, 2.320% below full GELU and 2.933% below narrow at seed 17,
with 70.3125% FFN reduction and all memory gates passing. All 225 tests pass.

This is a discrete rate bracket in one seed, not a global optimum or convergence
claim. Existing independent seeds for unchanged recipes need no redundant reruns.
The next requirement is a separately frozen longer-budget comparison with all
full and calibrated narrow controls. No GPU cohort is running.


## H050: Four-times-longer fixed-recipe comparison

**Status: VALIDATED for local final-step gates; longer robustness and convergence NEED INVESTIGATION.**
The [verified result](long_duration_results.md) completes all four 3,200-step
runs without retries or numerical failures. Plain retains 70.3125% FFN reduction
and all local quality/memory gates, but its NLL is 0.9819% above full SwiGLU,
0.4323% above full GELU and only 0.1763% below calibrated narrow. The separate
>=0.2% narrow margin fails. All models improve by another 2.2%-2.8% in the last
800 steps and fail the late-plateau diagnostic. The 800-step advantage therefore
cannot be promoted to converged superiority.

Every source/checkpoint/data/sampler contract verifies. The final suite passes
227 tests with `-p no:anyio`; an earlier optional compiler crash and a separate
plugin-import crash remain documented. No NN or trainer code changed.
Independent-seed longer controls are the next frozen comparison; duration-specific
rate ranking, broader scale and novelty remain open.


## H051: Independent longer-duration seeds

**Status: FAILED longer-duration all-seed promotion; parameter/memory components retained.**
The [verified replication](long_duration_replication_results.md) adds eight
3,200-step trials at seeds 29/43 and retains four seed-17 controls. Mean plain
NLL is 4.147443, 1.256% above full SwiGLU, 0.637% above full GELU and only
0.146% below narrow. Seed 43 exceeds the 1% full-SwiGLU allowance and loses to
narrow. The primary rule fails; a mean cannot rescue a failed seed. Every model
continues improving and fails the late-plateau screen.

All twelve source/data/checkpoint/sampler contracts verify. Two pre-training
import failures are retained; the qualified minimal worker completes all eight
new trials, changing only startup dispatch and no NN computation. All 233 tests
pass. Figure regeneration later hit additional import-only failures, retained
separately; isolated Matplotlib plotting without Torch imports completed the
consistent-color figure from verified records. The earlier 800-step result remains duration-qualified. A separately
frozen optimizer investigation precedes promotion; no GPU cohort is running.


Analytical follow-up only: [structured headwise mixers](structured_headwise_budget.md)
would reallocate the same 2,801,664 FFN weights from dense mixers to larger
private nonlinear subnetworks. Counts and bounded-input Jacobian upper bounds
are derived, with no registered architecture, new LM score or novelty claim.
This did not alter the frozen H051 protocol.


H051 operational record: two pre-training import failures are retained. The
qualified minimal worker avoids report imports, uses the identical frozen
configurations and calls the unchanged trainer. Two sampler reconstructions
and four tamper/overwrite checks pass; all eight actual trials subsequently completed.
The [CPU duration diagnosis](duration_optimizer_geometry.md) adds no optimizer
updates or validation scores and does not change the cohort.


The [Hessian witness](headwise_hessian_obstruction.md) rules out an explicit
vector quadratic for any single square-mixer additive headwise module, despite
arbitrary learned head functions. Full SwiGLU has an explicit representation;
CPU construction/Hessian checks pass. The argument uses established function
decoupling ideas and is not claimed novel. The derived overcomplete alternative
has the same FFN budget but remains unimplemented and untrained.

## H052 - Longer-duration product-decay control (ELIMINATED from promotion)

The frozen [single-variable control](duration_decay_plan.md) completes one
seed-17, 3,200-step WikiText trial (6,553,600 new tokens). Only existing factor
decay changes; four H050 references are retained. [Final NLL 4.152418](duration_decay_results.md)
is 0.161% worse than parameter-decay BlockShuffle and 1.145% above full SwiGLU.
Both local full-SwiGLU quality and the >=0.2% material-gain gate fail. No
independent seeds or further decay tuning are earned. Prior H013 negative
results and H051's longer-training failure remain.

Clipping rises from 30.50% to 42.41%; allocation remains 714.12 MiB. The initial
losses, exact source/data/sampler, changed optimizer coefficients, finite
weights/layers/gradients and checkpoint are verified. An interim advantage at
800/1600/2400 steps reverses by 3200; the full trajectory is retained. The last
interval still improves by 2.439%, failing the late-plateau diagnostic.

All 235 tests pass; all 31 registered models and the trainer/optimizer/data
implementations stay unchanged. The minimal worker completes on its first
attempt, without numerical or native/process failures. This rejects one
mechanistic optimizer ablation, not all factor regularization or headwise
alternatives. The untrained overcomplete proposal still requires constructive
representation and calibration checks before any new LM budget.

## H053 - Overcomplete structured-headwise qualification (VALIDATED locally)

The [frozen qualification](overcomplete_qualification_plan.md) passes for the
standalone 384->512->384 module, G=16, H=8, head width 64, private hidden 200.
It uses exactly 350,208 weights/layer. An explicit construction represents the
three-output quadratic excluded by square-mixer additive headwise layers,
using the actual structured A/B factors. FP64 forward error 1.718e-16 and
maximum Hessian-vector error 3.740e-14 support the checked construction.

The [model notes](archive/retired/src/multihead_ffn/overcomplete.md) derive rectangular
covariance accounting and isometric A / scaled row-isometric B initialization.
Three-seed Gaussian RMS ratios to full SwiGLU are 1.01356/1.01821/1.01643.
BF16 forward and maximum gradient errors are .007889/.008171; ten fixed
synthetic-target updates stay finite. All 238 tests pass. No language training
or scoring occurs and no new Transformer variant is registered.

Isolated eager forward/backward allocation is 69.366 versus 57.626 MiB for
full SwiGLU (+20.37%). Full-model integration/memory must therefore precede
any language-training allocation. This is a scoped expressivity/numerical
qualification, not a novelty, family-containment, easy-learning, convergence
or quality result. It earns only separately frozen integration and memory
checks; prior headwise, activation and decay failures remain.

## H054 - Complete-Transformer integration (VALIDATED); eager memory FAIL

The [frozen integration](overcomplete_integration_results.md) preserves all 31
older weights/logits/losses/gradients/FLOPs and registers a 32nd variant. Each
layer exactly matches its standalone calibrated initialization. All 241 tests
pass. Three fresh 20-update fixed-batch qualifications use 122,880 repeated
training-token exposures and zero validation targets.

Native candidate peak 730.389 MiB is 3.347% above full SwiGLU 706.733 MiB and
10.647% above full GELU 660.108 MiB. It misses the latter allowance by 4.270 MiB
and does not earn a quality screen. All weights/gradients/moments/layers stay
finite and checkpoint logits restore exactly. The small memory-only failure
justifies a separately frozen test of the existing native gate recomputation,
not extra activation or routing complexity. H054 itself adds no recomputation.

## H055 - Native overcomplete gate recomputation (VALIDATED locally)

The [frozen execution ablation](overcomplete_recompute_results.md) passes.
Peak 680.014 MiB is 50.375 MiB / 6.897% below eager; relative costs are -3.781%
versus full SwiGLU and +3.016% versus full GELU. Both memory limits and the
70.3125% FFN-weight reduction pass. No parameter, function, initialization,
LR/decay policy, new activation or router is added. The existing native
SwiGLU backward is reused, opt-in through the shared trainer helper.

All 32 default model signatures remain exact; 243 tests pass. Full-model initial
loss and every BF16 parameter gradient match eager, and all 9,099,648 final
weight elements plus the entire 20-step history are bitwise equal in these
runs. This is a local execution result, not a universal precision guarantee.
One new cell adds 40,960 repeated training-token exposures and zero validation
targets. It earns a separately frozen balanced quality screen; H054's eager
failure and the broader unmet research target remain.

## H056 - Overcomplete headwise quality screen (ELIMINATED at this budget)

The [frozen three-rate screen](overcomplete_screen_results.md) completes all
three seed-17 200-step trials. NLL .0003/.0006/.0012 is
6.303587 / 6.134152 / **6.043891**. The selected boundary rate loses full SwiGLU
by **2.305%**, full GELU by **2.800%**, narrow by **2.223%**, selected plain
BlockShuffle by 1.227% and square headwise by .375%. All original quality gates
fail; parameter and actual 676.931 MiB memory gates pass. No longer training,
activation repair or automatic rate search is earned.

All updates and final weights/moments/layer outputs remain finite; clipping is
42% / 20% / 7%. All 245 tests pass and existing model/training computation is
unchanged. The three new cells use 1,228,800 sampled targets and score all
322,688 validation targets per evaluation; official test remains unscored.
A diagnosed pre-training diagnostic-source assertion failure and its exact
source are retained. The corrected preflight verifies old/current diagnostic
values, fifteen retained control artifacts, actual groups and exact sampler
states. All GPU cells complete first attempt.

Reusable component: H053's explicit quadratic construction survives as a scoped
representation result. Rejected inference: that escaping this headwise
obstruction with matched initialization RMS suffices for better short-budget
language learning. The selected candidate's layer-0 FFN output RMS grows to
13.7203 (full SwiGLU 3.8765), while later FFN parameter-gradient norms are smaller.
A separate residual-scale/factor-geometry checkpoint diagnosis is justified;
these measurements do not prove a causal explanation or justify automatic repair.

## H057 - Residual geometry diagnosis (VALIDATED locally); causal remedy unproven

The [frozen diagnosis](residual_geometry_results.md) matches all 340 local norm
VJPs to an independent finite-epsilon formula, maximum relative error 9.882e-8.
Native candidate/control later gain ratios are .7071 SwiGLU, .7742 narrow,
1.0368 GELU, .1711 BlockShuffle and .4113 square headwise; CPU ordering agrees.
Full GELU is a counterexample to a simple larger-gain/better-quality story.
Candidate lower rates have larger gains and better B conditioning but worse NLL.
The diagnosis does not establish the cause of the quality deficit or earn a fix.

A disposable exact value/down rescaling changes corresponding gradient norms
by 1/8 and 8, with zero measured logit/loss/predicted-gradient error. All weights
are restored and checkpoints unchanged. Initial isometry is lost during training;
selected B-map conditions span 129.26-3978.29 while numerical rank stays 384.
These observations do not prove whole-network vanishing or optimization failure.

All 248 tests pass; 14 CPU and six BF16 probes plus one gauge check make zero
optimizer updates and score no validation/test targets. All phases complete
first attempt, with all old computation/configurations retained. Stop this
analysis branch; no activation, normalization or optimizer repair is earned.

## H058 - Additive block/global SwiGLU (HYPOTHESIS; unimplemented)

The [proposal](additive_block_lowrank_proposal.md) uses W=S+LR in all three
ordinary SwiGLU projections: G=8 block support, d=384, h=1024 and rank48.
Its count is exactly 350,208 weights/layer, or 70.3125% below full FFN, matching
all current compressed controls while retaining full hidden width. A direct
block path and additive global path are distinct from private headwise networks.
An explicit quadratic construction is described, but no implementation or
numerical qualification has occurred and no training budget is frozen.

Sparse-plus-low-rank pretraining and diagonal additions are established prior
art, including SLTrain (2024), low-rank passthrough networks (2016) and DLoR
expressivity work (2026). This is not a verified novel combination. Next: audit
primary implementation conventions, derive fixed calibration/optimizer geometry,
implement independently verified structured execution, and earn actual full-model
memory before balanced language training. Alternative dense self-guidance and
H057-inspired optimizer repairs are deferred; do not revive the failed recipe.

## H059 - Finish the active shortlist (VALIDATED as repository cleanup)

The [cleanup](cleanup_results.md) reduces 11 model folders/32 variants to three
folders/six variants. Nine old folders and 138 obsolete individual files moved to
a verified archive. The reusable grouped operator remains in core; rational
BlockShuffle now has its own descriptive folder. No new LM trial or quality
promotion occurred. All 89 active tests pass and all six retained variants match
pre-cleanup CPU/GPU weight, prediction, gradient and update signatures exactly.
Plain inference adapters reject learned activations atomically, preventing silent
shape removal. H058 remains the next unimplemented hypothesis; no failed branch
is revived by this housekeeping result.

## H060 - Additive block/low-rank qualification (VALIDATED locally)

The [fixed policy](additive_block_lowrank_qualification_results.md) passes17 local
checks and the full110-test suite. The actual model preserves70.3125% FFN weight
reduction; the vector quadratic is represented using actual supported tensors.
All15 synthetic GPU workers finish20 updates with finite weights/moments. Candidate
allocation591.438MiB is7.03% below full GELU and14.01% below BlockShuffle, but20.67%
above narrow. Its median step78.814ms remains slower than both dense controls.
All six previous CPU/GPU model signatures are exact. The mechanism earns a balanced
language screen, not a quality/novelty/convergence claim. The initialization and
update calibration were frozen before scoring and will not be retuned after it.

## H061 - Additive block/low-rank quality screen (ELIMINATED at this budget)

The [three-rate screen](additive_block_lowrank_screen_results.md) completes all
three seed-17, 200-step cells. NLL at .0003/.0006/.0012 is
6.243238 / 6.093809 / **6.032030**. The boundary winner loses to full SwiGLU by
**2.105%**, full GELU by **2.598%**, narrow by **2.023%** and selected BlockShuffle
by **1.029%**. Every original quality gate fails; 70.3125% FFN reduction and
617.806 MiB actual training allocation pass. It loses to all four controls at
each matched rate too. No longer training or automatic repair is earned.

All weights, moments and diagnostics remain finite; clipping is 54.5% / 31% / 10%.
The three cells add 1,228,800 sampled targets; official test remains unscored.
113 tests pass. A filename-collision preflight failure is preserved and repaired
before training, with no model or protocol change. The corrected preflight
matches twelve full-size controls to four archived source trees, verifies target
order and final sampler states, and all new training cells finish first attempt.
H060's construction and numerical qualification survive as scoped evidence;
competitive language learning does not follow. Retire this tested recipe.

## H062 - Finalize shortlist after additive rejection (VALIDATED as cleanup)

[Retirement](additive_retirement_results.md) returns the active tree to three
model folders, six variants, nine recipes and the verified H059 shared code.
One rejected model folder and five standalone driver/test/configuration files
move to the H061 archive. All 171 LM/profile runs and their checkpoints survive.
89 tests pass; twelve retained CPU/GPU signatures are exact. No new corpus
training/scoring or quality promotion occurs. The constructive proof remains
historical evidence, while the tested additive recipe earns no further tuning.

## H063 - Native recomputation (VALIDATED locally; memory repair REJECTED)

The [equally checkpointed audit](native_recompute_results.md) completes 15 workers,
300 synthetic updates and zero corpus scoring. Both scopes preserve initial
full-model gradients, all 20 losses/norms, and final weights/moments exactly for
all five recipes. The full 104-test suite passes; six variants and all twelve
historical default CPU/GPU signatures remain intact.

Rational peak memory is 855.194 MiB existing, 636.207 MiB whole FFN and 460.333 MiB
whole block. These save 25.61% / 46.17% against itself, at 20.31% / 25.61% update-time
cost. Equivalent full controls also improve: both rational scopes fail their
same-scope 110% memory caps. Whole-block rational is about 40% above either full
control and 208.670 MiB above plain BlockShuffle. No full training repeat or
additional boundary search is earned. The generic native execution method is
locally qualified; it is not a new primitive or a quality result.

A preflight CPython AST SystemError and a passing independent diagnostic are
preserved. One explicit unchanged-source preflight retry passes; every GPU worker
finishes first attempt. The remaining peak gap is measured but not attributed
per operation. Stop this repair, retaining all prior activation and long-run failures.

## H064 - Rational allocation diagnosis (VALIDATED accounting; partial attribution)

The [frozen four-worker diagnosis](rational_memory_results.md) completes with zero
optimizer updates and 16,384 synthetic target exposures. Traced/untraced loss,
logits, every gradient and allocated peaks are exact. Replay reconciles phase
counters and terminal block layouts with native allocator rounding and splitting.
Plain peaks at 251.975 MiB; rational at 460.332 MiB, a 208.357 MiB one-pass gap.

Eight simultaneous 16 MiB allocations originate in FP32 rational evaluation during
checkpoint recomputation. This supports the transient pointwise-cost hypothesis;
it does not prove 128 MiB is removable. Rational also has 184.989 MiB of unmatched
new runtime allocation and 16.250 MiB unidentified before tracing. C++ stacks are
unsupported on the installed Windows build. Close the diagnostic with partial
attribution; no automatic execution repair, compiler retry or training follows.

108 active tests pass. An existing Triton test initially fails in PTXAS; one
unchanged targeted rerun and then a fresh full suite pass. Cause remains unknown,
and all records survive. Every H063 source/configuration/test file is unchanged,
four new parser tests are added, and all four GPU workers finish first attempt.
The three-folder, six-variant shortlist and previous negative results remain.

## H065 - Staged rational recomputation (LOCALLY QUALIFIED)

The [fixed four-region partition](rational_staged_results.md) passes all twelve
native-versus-staged product comparisons: CPU FP32 and full-shape CUDA BF16,
zero and trained coefficients, three input seeds. All 60 output/gradient tensor
comparisons are bitwise exact. FP64 direct-formula and independent gradcheck pass;
parameters and RNG remain unchanged. There are zero optimizer updates or corpus
targets. This is known checkpointing applied to the same rational formula, with no
active variant added. It earns a separately frozen full-model resource and update
comparison; memory savings, quality and novelty remain unproven.

## H066 - Full-model staged rational repair (REJECTED memory; exact fidelity)

The [six-worker study](staged_resource_results.md) preserves all initial gradients,
twenty losses/norms and final weights/moments exactly, including both additional
dense activation-checkpoint controls. Rational memory falls 460.333 -> 428.333 MiB
(32 MiB, 6.9515%), with 8.8451% more update time. It misses the 10% self-reduction
and both 110% dense memory caps (361.666 / 361.753 MiB). No language repeat or new
partition search is earned. H064's 128 MiB live attribution was not a removable
memory guarantee. Close this fixed repair; keep its code only with experiment
artifacts and leave the three-folder/six-variant active shortlist unchanged.

All workers finish first attempt: 120 synthetic updates, 245,760 training targets
and 12,288 initial-probe targets, with no corpus scoring. Every active source,
configuration and test file remains at H064 bytes. Local exactness is preserved;
adequate resource behavior and the broader research goal remain unmet.

## H067 - Token-conditioned residual value activation (LOCALLY QUALIFIED)

[Twenty-one fixed checks](token_activation_results.md) pass for one scalar router
per token/layer and its static control. Dynamic adds 3,080 weights at d384/L8,
retaining 70.2799% FFN reduction. Zero initialization preserves all 96 measured
shared output/gradient tensors, with finite nonzero router signals; four nonzero
checkpoint comparisons add 38 exact tensors. A full-size structured construction
and independent FP64 gradients pass. The [proof](token_activation_theory.md)
separates one smooth exact function from every finite single SwiGLU/exact-GELU
FFN, without an approximation-rate or learning claim. MoA and pole-analysis prior
art are explicit. No optimizer updates or corpus targets occur.

The prototype stays outside the active factory. It earns a separately specified
plain/static/dynamic fitting and resource comparison, including baseline-favorable
targets. It does not earn language training or change the completed rejection of
H066. Stale FlashMHF documentation is corrected; its failed local screens survive.

## H068 - Token activation fitting (REJECTED at the frozen budget)

[Seven-target fitting](token_activation_fit_results.md) completes 252 trials with
three seeds, two equal-budget rates and 300 updates. Dynamic's paired geometric
mean error improves 0.1213% over plain and 0.0628% over static. Static improves
0.0585% over plain. Both fail the predeclared 2% plain-improvement gate; dynamic
also fails 1% over static and the every-seed requirement. Both beat calibrated
narrow overall, but remain worse than the full controls in aggregate. A valid
exact-function proof and nonzero gate learning signal did not yield a sufficient
finite-budget learning improvement.

At d48/h256/G8, dynamic has 5,521 weights (70.0467% reduction), static 5,473 and
plain/narrow 5,472. Selected median updates are 6.080 / 5.833 / 5.031 ms for
dynamic/static/plain; the full dense controls are faster at this small FP32 shape.
These are standalone, not full-model resource results. Resetting learned gates
raises error only 0.4436% dynamic / 0.3740% static after coadaptation.

Four harness checks pass. The first launch is interrupted with no completed
endpoint and an unknown 0..300 executed updates. A separately recorded one-time
unchanged recovery completes all 75,600 updates in 359.72 s; no scientific choices
change. Data and sampler tensors match before fitting. Independent CPU rescoring
checks 126 selected checkpoints and 42 ablations (max relative MSE difference
7.93e-8), while all 252 checkpoint/moment states, histories and selections pass.
No corpus targets or resource workers occur. Close both fitting recipes without
adding an active model, tuning amplitude/rate or extending their training.

## H069 - Internal factor balance (LOCALLY EXACT; optimizer motivation REJECTED)

[The diagnosis](factor_balance_results.md) covers 216 projection records from
initialization and existing 800/3,200-step checkpoints, seeds17/29/43. Correctly
permuted power-of-two row/column rescaling preserves the represented linear map.
A weighted Frobenius-energy calculation proves the rounded representative is
within 1.25 times its positive continuous optimum and reduces weighted norm
ratios to [1/2,2]. This is coordinate bookkeeping, not a better projection condition
number, new expressive function or optimizer convergence theorem.

All 24 CPU FP32/CUDA BF16 FFN pairs pass: 192 exact output/input/mapped-factor
comparisons. Independent inspection rederives all 82,944 channel records and
reloads every raw paired tensor. The three trained seeds reduce weighted energy
only 12.5759..12.7591% (required20%); no channel exceeds fourfold imbalance
(required10% of channels). Both motivation gates fail in every trained seed.
No optimizer-state qualification, learning allocation or new active variant is
earned. Existing H052/H057/H068 rejections are unchanged.

Four original checks pass before a BF16-to-NumPy hashing error interrupts the
first diagnosis after one CPU pair. Preserve that failed attempt. A separately
frozen serializer-only correction passes four logging tests and completes the
unchanged diagnosis in 11.90 s; the first CPU artifact reproduces exactly.
Both attempts have zero optimizer updates and zero corpus targets. Factor-gauge
and balancing prior art are explicit; no architecture/optimizer priority claim.

## H070 - One rotated shuffle (LOCALLY QUALIFIED; learning unmeasured)

[One Givens stage](rotated_shuffle_results.md) crosses input origins and output
groups between the retained linear factors. A same-cost control shares input
origins and is provably absorbable into the first factor. Zero angles include
the baseline; the cross-origin form constructs a canonical rank 7 block where
all original projections have rank <= 6. Its explicit matrix witness has an
original-family relative Frobenius error floor 1/sqrt(13). This is a projection
result, not strict full-FFN separation, a learning guarantee or novelty priority.

Both forms add 4,608 weights at d384/L8 (FFN 2,806,272, total 9,104,256; 70.2637%
FFN reduction). All 21 checks pass first attempt: 140 exact output/gradient
comparisons, 36 finite nonzero angle signals, counts/topology, rank witness,
absorption, FP64 gradcheck and component isometry. Independent raw-tensor and
matrix reconstruction pass. No optimizer updates, corpus targets or resource
workers occur. It earns only separately frozen fitting/resource qualification
with matched controls; no active model or recipe is added.

H068's small width has 16 missing canonical block connections, unlike the full
k384 setting. Its documentation now records this limitation; frozen data/source,
metrics and rejection are preserved. H069 and earlier hypotheses remain closed.
A separate CPython report-import failure is recorded and avoided by removing
an unnecessary Torch dependency from the report helper; scientific checks are
not rerun. Established structured/orthogonal-transform prior art is explicit.

## H071 - Full-size rotated-shuffle fitting (REJECTED at this budget)

The [matched screen](rotated_shuffle_fit_results.md) uses actual d384/h2048/G8
projection topology, seven targets, three seeds and two equal rate budgets.
All 252 cells complete 300 updates: 75,600 updates and 19,353,600 examples.
The expressive shift1 is 0.0025% worse than plain and 0.0780% worse than shift0.
Absorbable shift0 improves plain by only 0.0755%. Both fail the 2% gain criterion;
shift1 additionally fails every-seed improvement and 1% improvement over shift0.
They beat calibrated narrow in aggregate, but that does not satisfy all gates.
No full-model resource qualification, active variant or automatic tuning follows.

Same-cost rotation control separates added optimization coordinates from the
projection-family expansion. Selected update time rises 5.1483 ms plain to
9.1889 ms shift0 / 10.0927 ms shift1; these are standalone FP32 measurements.
Both retain 70.2637% FFN reduction. Resetting angles changes aggregate error by
only 0.0106% / 0.0653%, respectively; this is post-training dependence, not a
causal learning analysis. H070's scoped rank witness and derivative checks stand.

A post-hoc zero-predictor diagnostic shows all selected forms/seeds worse than
zero on five of seven held-out target families, despite lower training errors.
This limits the interpretation of tiny relative differences in this fixed
learning regime. It changes no decision gate and is not a reason to reopen
these recipes automatically. Neither poor held-out fitting nor this budget
rejects every possible use of orthogonal transforms.

All checkpoints are finite and independently audited. CPU data regeneration,
126 selected endpoint rescores and 42 angle-reset ablations pass. Four isolated
harness checks pass before fitting. One audit tool observation was interrupted
with no returned handle; follow-up found no live process/output, and earlier
audit execution is unknown. The subsequent durable CPU audit passes; no training
is repeated. Active sources, prior results and frozen plans are preserved.

## H072 - Existing ungated structured control (LOCALLY QUALIFIED; untrained)

The [qualification](ungated_blockshuffle_results.md) uses the existing ungated
BlockShuffle path, with no active source change. Two projections permit h3264
at the h2048 SwiGLU budget: 350,208 weights per FFN, 70.3125% FFN reduction.
The h2048 GELU control uses 233,472 weights. All 18 fixed CPU/GPU checks pass,
including 72 exact output/input/factor-gradient comparisons and four independently
reconstructed projection matrices. No optimizer update, corpus score or full-
model resource measurement occurs. No new active model or recipe is added.

The scoped parity proof shows linear odd parts for any finite single bias-free
GELU FFN and homogeneous quadratic even parts for SwiGLU. Full-size sparse
witnesses exclude exact containment in either direction. For cyclic multiplicative
uniform-input targets, every such GELU layer has population relative squared-error
floor 12/17; exact moments and 81-node polynomial quadrature agree. These facts
provide no Transformer separation, finite-held-out floor or learning guarantee.

Passing earns a separately frozen learning comparison with same-width and matched
controls, explicit positive-control generalization and absolute-error diagnostics.
H037's affine result and H068-H071 rejections remain closed. This is a conventional
control and elementary analysis of established components, not novelty priority.
A report-only font/ABC failure and one successful source-identical rendering
attempt with a fresh local Matplotlib configuration are preserved. Its cause is
unknown; scientific qualification is not repeated.

## H073 - Ungated structured fitting (PROMISING; earns resource qualification)

The [full-size comparison](ungated_fit_results.md) completes 294 trials first
attempt: seven tasks, seven forms, three seeds, two equal rates and 300 updates.
Both same-width and parameter-matched GELU pass every frozen learning gate after
the full-GELU positive-control assay passes. Aggregate error improves plain by
44.893% and 50.271%; both beat calibrated narrow GELU/SwiGLU and improve plain
in every seed. Matched GELU improves same-width GELU by 9.759%. FFN reductions
are 80.2083% and 70.3125%, respectively. Both earn only a separately frozen
full-model resource comparison; no active variant or language run is added.

The linear positive control contributes heavily to the aggregate gains. Post-hoc
subsets do not change selections or gates: on four generic tasks, matched GELU
improves plain by 19.442% and narrow GELU by 4.605%, while roughly tying full GELU
(0.148% lower error). Excluding only linear, it is 0.178% worse than full GELU.
Oscillatory learning remains around predicting zero. Training is not converged,
and the three optimization seeds share one fixed data/teacher realization.
This conventional control establishes neither novelty nor universal expressivity.
H072's single-layer restrictions and 12/17 multiplicative population floor stand.

Selected standalone updates take 4.418 ms same-width / 4.398 ms matched, versus
5.589 ms plain / 2.274 ms full GELU. Fewer weights have not established practical
speed superiority. Transformer resources and language quality remain unmeasured.
The 88,200 updates and 22,579,200 example presentations finish in 453.13 s, with
zero corpus targets. Five isolated harness checks pass; independent CPU audit
regenerates data/streams, verifies all 294 checkpoint weights/moments and rescores
147 selected endpoints. All prior active sources, results and frozen plans remain
preserved; earlier rejected recipes are not reopened.

## H074 - Ungated full-model resources (INCOMPLETE: import failure)

The [resource attempt](ungated_resource_results.md) completes17 of21 workers,
then stops before same-width GELU's inner-checkpoint probe or updates. AdamW
construction triggers a Torch configuration import and CPython tokenization
raises `SystemError: Unmatched paren in format`. The failed worker directory is
empty; all three matched-width workers are unlaunched. The source snapshot and
all completed measurements are preserved, with no scientific retry.

Same-width GELU's completed whole-block case uses243.913 MiB at108.476 ms/update;
plain uses254.038 MiB at127.437 ms and full GELU331.538 MiB at87.196 ms. These
fresh-initialization synthetic measurements do not qualify a candidate with
missing execution modes. H073's fitting evidence remains promising, but neither
form earns language training. The failure is not an architecture rejection.

Five isolated harness tests pass. Independent CPU audit verifies17 checkpoint
pairs and11 available within-form comparisons exactly, with finite weights and
moments and regenerated token streams. Completed work is340 updates,696,320
training targets and34,816 probe targets; zero corpus. A one-pass read-only
AST/tokenizer probe succeeds on the implicated source files, but the cause of
the original import-state failure remains unknown. Any operational recovery
requires explicit specification and must preserve completed work and gates.
Active models, prior evidence and frozen plans are unchanged.

## H075 - Ungated resource recovery (QUALIFIED; earns short language screen)

The [bounded recovery](ungated_resource_recovery_results.md) passes three fresh
construction-only probes and completes the four missing H074 workers, using its
unchanged loop and original gates. Both GELU forms pass resource qualification.
Matched GELU has70.3125% fewer FFN weights and selects255.225 MiB /107.857 ms;
same-width has80.2083% fewer weights and selects243.913 MiB /108.476 ms. Both
select whole-block checkpointing. Full GELU uses331.538 MiB /87.196 ms and full
SwiGLU327.788 MiB /95.701 ms: lower memory does not establish faster dense training.
No active variant is added, and language quality remains unmeasured.

Independent CPU audit reloads21 initial/final checkpoint pairs, reproduces all14
within-form comparisons exactly and independently rederives every selection/gate.
The three probes perform zero forwards/backwards/updates; two reproduce the
original full-size initial weights and optimizer state/groups. The earlier import
failure's cause remains unknown, and its original trace/empty directory survive.
There is one explicitly specified retry of the pre-update failed cell, three
first launches of missing cells and no rerun of17 completed measurements.

New work adds80 updates,163,840 training targets and8,192 probe targets. Combined
work exactly matches the original420-update allocation. The time gap between
cohorts limits timing interpretation. All original sources, prior results and
frozen plans remain preserved. Passing earns a separately frozen short language
screen with full/narrow controls and equal budgets, followed by the original
quality, convergence, replication, scale and broader-data requirements. No
novelty priority or research-goal completion is claimed.

## H076 - Ungated GELU language screen (matched PROMISING; same-width REJECTED)

The [frozen language comparison](ungated_lm_screen_results.md) completes all 21
fresh trials: seven forms, peak rates 0.0003/0.0006/0.0012, seed 17 and 200 updates.
Matched-budget GELU h3264 passes every original quality/count/memory gate and
alone earns a separately frozen longer comparison. Same-width h2048 fails the
full-GELU 1% allowance and both narrow quality comparisons. Its fixed recipe is
closed without an automatic rate, width, activation or budget repair.

Selected matched NLL is 5.895275541 with 2,801,664 FFN / 9,099,648 total weights,
70.3125% FFN reduction, 280.017 MiB and 111.154 ms per mean timed update. Full
GELU selects 5.879277652 at 393.245 MiB / 91.249 ms; full SwiGLU selects
5.907693898. Matched is 0.2721% worse than full GELU and 0.2102% better than
full SwiGLU. Its narrow-GELU gain is only 0.1167% against selected LR 0.0006,
NLL 5.902164305; narrow SwiGLU reaches 5.912504605. The 0.2% margin is descriptive,
not a replacement gate. Plain selects 5.970604333, making matched 1.2617% better.
Same-width selects 5.942922671 and has 80.2083% fewer FFN weights.

This candidate reduces actual memory 28.79% relative to full GELU but uses
21.81% more update time. Whole-block checkpointing is frozen from H075 for all
except narrow GELU's block-plus-inner mode. Both narrow controls use calibrated
down initialization/LR and product decay, following the established language
recipe; H075's synthetic probes used parameter decay. All other shared training,
data and evaluation source remains unchanged. Two-projection counts/FLOPs and
all scalar observers are isolated, with no registered model or recipe addition.

Five harness checks pass. Independent audit verifies all 21 checkpoints and
AdamW states, initial signatures, complete step logs/schedules, source/data hashes,
exact sampler states and validation order. Every stored BF16 validation NLL
rescores exactly. Work is 4,200 language updates / 8,601,600 training targets,
47,435,136 original validation exposures and 6,776,448 independent rescore targets.
Eight separate CPU qualification updates present 256 training targets. There are
no scientific retries, optimizer updates during rescoring or official test scores.

The one-seed margin is small, selected rates can lie at the search boundary,
all models still improve in the final interval, and validation has been heavily
reused during development. H051's longer plain failure and all other negative
results remain. No convergence, novelty, universal gradient property or research-
goal completion is established. The active three-folder/six-variant shortlist,
171 prior language/profile runs and all prior fitting/resource evidence survive.

## H077 - Longer three-seed GELU replication (PROMISING; fixed-budget replication passes)

The [complete comparison](ungated_lm_replication_results.md) finishes all 18
fresh 800-step trials across seeds 17/29/43, using six fixed H076-selected
recipes. Matched-budget GELU passes every seed's primary gates. Only a separately frozen duration/convergence comparison is earned.
Its mean NLL is 4.834511375 +/- 0.013382428
(sample SD), with 2,801,664 FFN / 9,099,648 total weights, 70.3125% FFN reduction.

Relative mean NLL is -0.6456% versus full GELU,
-1.3468% versus full SwiGLU,
-2.8830% versus narrow GELU,
-0.8867% versus narrow SwiGLU and
-0.0959% versus plain. Negative favors matched.
Matched beats plain in only 1/3 seeds; the mean NLL benefit does not
establish a dependable plain-quality improvement. Mean update time changes by
-14.47% versus plain.
All per-seed gates, paired differences/sample SD and exploratory df=2 intervals
are in the report. A passing mean cannot rescue a failed seed. The 0.2% narrow
margin is descriptive. Late-plateau diagnostics pass 0/18;
no convergence or statistical-significance claim follows.

Matched mean allocated peak is 280.017 MiB, mean timed update
105.271 ms. Full GELU uses 393.245 MiB /
86.057 ms. Actual corpus memory, full-sequence inference,
all clipping and diagnostics are retained; fewer weights do not imply faster
training or a global gradient guarantee.

Only steps/logging/seed change from H076's selected recipes. Fresh 800-step
schedules have 80-update warmup; initial weights, native execution, structured LR,
narrow product decay and all shared trainer/data/evaluation sources are unchanged.
Five integration checks pass with zero optimizer updates. All 18 final BF16 NLLs
rescore exactly; initialization, moments, logs, sampler and source/data hashes
are independently verified. Work is 14,400 updates / 29,491,200 training targets,
40,658,688 original validation exposures and 5,808,384 audit validation targets.
Read-only sampler checks construct 4,915,200 unscored targets with no model forwards.

All scientific trials finish first attempt. The earlier implementation-write
tool observation interruption left no file or live writer; it is preserved and
is not a scientific retry. The current-state page is shortened for readability;
its full prior version survives in before_docs.zip. The active three-folder/
six-variant/nine-recipe tree and all earlier evidence remain unchanged.

A separate [comparator note](ungated_comparator_note.md) reads BLAST, StructuredFFN
and BTT-MoE primary sources and derives an unallocated equal-parameter BLAST
control. It changes no H077 hypothesis or decision. The reused development split,
short-selected rates, missing convergence/scaling/broader-data/comparator evidence
and known-component novelty limits prevent completion of the research goal.

## H078 - Fixed-recipe 3,200-step GELU duration test (REJECTED; fixed duration recipe fails)

The [complete report](ungated_duration_results.md) finishes all six fresh seed-17
trials at 3,200 updates. Matched GELU fails the frozen primary gates, with NLL
4.242013727, 2,801,664 FFN / 9,099,648 total weights, 70.3125% FFN
and 42.1700% total reduction. This fixed duration recipe is closed. Further training needs a distinct, justified hypothesis.

It exceeds both full-control NLL allowances and fails to beat either narrow control. Candidate relative NLL differences are
+2.7649% versus full GELU, +3.3272% versus full SwiGLU,
+0.8601% versus narrow GELU, +2.1422% versus narrow
SwiGLU and +2.3576% versus plain. Candidate peak is 280.017 MiB,
mean timed update 104.317 ms, -14.46% versus plain. All controls' resource
and inference measurements remain in the report. No replicated plain advantage,
whole-network gradient property or universal speed benefit follows.

Late-plateau diagnostics pass 0/6 models. Each final-800 loss change is
reported, and the separate 0.2% narrow margins do not change primary gates.
This one-seed duration comparison supplies no confidence interval, significance
or convergence proof. H077's three-seed result applies only to 800-step schedules.

Rates, models, initialization, optimizer calibration, execution, data and all
shared training/evaluation code match H077. Only steps/logging change; warmup
stretches to 320. These are fresh schedules with exactly matching initial weights,
NLL and first pre-update training loss; later short/long trajectories differ.

Five preflight checks pass with zero updates/model forwards, including all six
initial states and a 3,200-batch CUDA sampler reconstruction. All six final BF16
NLLs rescore exactly. Independent checks verify every checkpoint/moment, 3,200
step records per run, source/data hashes, sampler and full validation order,
parameter/operation counts and all decisions.

Work is 19,200 updates / 39,321,600 training targets and 13,552,896 original
validation exposures. Independent rescoring adds 1,936,128 validation targets,
zero updates. Read-only sampler qualification constructs 6,553,600 unscored target
elements. All trials finish first attempt. Three active model folders, six variants,
nine recipes, 171 original runs, 21 H076 and 18 H077 trials, 798 fitting cells and
all prior resource evidence/failures remain preserved.

Heavily reused development validation, short-selected rates and missing long-budget
replication/convergence/scaling/broader-data/published-comparator evidence leave
the gold research goal unmet. H051's plain failure and the local proof's limits
remain evidence; this result does not establish novel components or state-of-the-art.

## H079 - BLAST qualification (NEEDS_INVESTIGATION; saved-artifact integrity failed)

Original process reports PASS in 13.826 seconds, but 25 files are zero-filled;
only 18/34 observation JSONs parse and 24/32 tensor archives pass integrity.
Source/protocol/archive and H078 evidence remain intact. Cause is unknown; original
bytes are preserved and qualification remains incomplete. No optimizer update,
corpus target or Transformer was allocated. [Integrity report](blast_operator_results.md).

## H080 - Explicit BLAST recovery (LOCALLY QUALIFIED; learning untested)

One frozen recovery repeats unchanged operator/test bodies, seeds, shapes,
precision and thresholds. Only artifact location and durable writing change.
All 34 checks and 32 saved tensor files pass; 114 eager/checkpoint tensor pairs
are exact, and 60 FP64 projection comparisons meet the original tolerances.
Independent analytic-gradient error is at most 8.882e-15; Gram error 1.998e-15.
Finite differences use ten forwards, zero updates.

GELU h3200 and SwiGLU h1984 each have 350,208 weights/FFN, 2,801,664 across eight
layers and 9,099,648 total by count. Four same-width embeddings and four
Hadamard/rank-collapse certificates pass. The local rank-6 matrix squared-error
bound is 7/8; same-width inclusion costs 3,072 extra weights/projection.
Reducing hidden width by 64 pays the budget but prevents an equal-budget nonlinear
containment inference. Initial partial isometry is not a trained-network guarantee.

All 24 readable original tensor payloads match exactly. H079's damaged evidence
remains unchanged. A failed CPU/GPU scalar-norm audit is corrected by checking on
the original device, keeping exact equality. Its failure and a pre-execution
report-writer syntax error are preserved. Neither changes a numerical threshold
or repeats qualification/training. Across both stages: two qualification attempts,
one explicit repetition, zero optimizer updates/corpus targets/Transformers.

BLAST is prior art with local permutation/initialization adaptations. Only a
separately frozen learning comparison with explicit optimizer calibration and
equal tuning budgets is earned. Active code remains three folders, six variants,
nine recipes. [Complete report](blast_operator_recovery_results.md).

## H081 - BLAST language and factor calibration (SCREEN COMPLETE)

H081 completes all 24 fresh 200-update WikiText-2 runs and all final checkpoint
rescores match exactly. BlockShuffle calibrated: fails fixed screen; BLAST SwiGLU: fails fixed screen; BLAST GELU: fails fixed screen.
This is one-seed short-budget evidence; longer, multi-seed, scale and broader
data requirements remain open. [Complete screen](blast_learning_screen_results.md).

| Model | Selected LR | Final NLL | Peak MiB | Mean update ms | Clip % |
|---|---:|---:|---:|---:|---:|
| Full SwiGLU | 0.0012 | 5.907694 | 384.56 | 102.25 | 10.5 |
| Full GELU | 0.0012 | 5.879278 | 393.25 | 87.15 | 12.0 |
| Narrow SwiGLU | 0.0012 | 5.912505 | 279.45 | 93.05 | 8.5 |
| Narrow GELU | 0.0006 | 5.902164 | 279.02 | 91.39 | 17.5 |
| BlockShuffle legacy | 0.0006 | 5.970604 | 279.77 | 126.89 | 16.0 |
| BlockShuffle calibrated | 0.0012 | 6.073010 | 279.77 | 139.54 | 7.5 |
| BLAST SwiGLU | 0.0006 | 6.165411 | 281.20 | 153.90 | 38.5 |
| BLAST GELU | 0.0006 | 5.951733 | 280.95 | 125.34 | 11.0 |

## H082 - Internal activation preflight (INCOMPLETE PREFLIGHT)

Learned tanh/sine squared residuals and an equal-count affine control add48
weights per standalone BlockShuffle FFN, inside its existing factor products.
Local positive slope/identity/Bezier calculations and CPU/GPU gradient tests
pass, but the adapter's bulk patch collides with a replacement named target.
Seven checks pass; one cannot run. No fitting data or optimizer update exists.
[Failure and preservation](latent_activation_results.md).

## H083 - Binding-only recovery and nonlinear fitting (COMPLETE: CURVES REJECTED)

Both internal curves improve synthetic fitting over plain BlockShuffle, but
neither qualifies: tanh changes aggregate held-out MSE by -5.757%
and sine by -7.069%, while the equal-count affine control changes
it by -15.034%. Both curves lose to affine and narrow GELU.
The positive-control assay passes. These fixed recipes are closed; no language
or full-model resource run is earned.

All eight checks pass after one explicit adapter-only preflight recovery. All
192 fitting cells finish on their first attempt. Independent analysis reproduces
192 selection scores, 96 selected held-out scores and 36 reset ablations exactly.
The curves add 48 weights per FFN but take about 9.8 ms/update versus plain's
5.3 ms and narrow GELU's 2.4 ms in standalone FP32 fitting. Learned corrections
affect final predictions; that dependence is not a causal training explanation.
Affine's gain/offset contributions remain unresolved, and it also loses to
narrow GELU. No automatic extension, new active model or language trial follows.
[Complete result](latent_activation_recovery_results.md),
[recovery plan](latent_activation_recovery_plan.md), [local proofs](latent_activation_theory.md).

## H084 - Internal affine mechanism (CHARACTERIZED; NO PROMOTION)

The internal affine control depends on both learned gain and offset, but its
gain is algebraically redundant with the first weight factor. Removing gain
without compensation multiplies aggregate held-out MSE by 2.972; removing offset
multiplies it by 1.843. Folding the affine operations preserves predictions
within the declared FP64 and FP32 tolerances. This is a checkpoint-mechanism
result, not a new trained model, causal training explanation or speed result.

All 12 checkpoints, 24 FP64 folding pairs, 48 origin cases and the nonzero-slope
witness pass. Independent checks exactly reproduce the 24 new reset scores and
prediction hashes. Gain folding changes coordinates; uncompensated reset changes
the function. Gains and offsets cannot be assigned causal training credit from
these interventions. No new training, rate search, BF16 or runtime claim follows.
[Complete analysis](latent_affine_mechanism_results.md),
[frozen plan](latent_affine_mechanism_plan.md).

## H085 - Controlled internal offset training (COMPLETE)

H085 completes 240 fresh runs and all independent score checks pass.
Learned offset changes selected aggregate held-out MSE by -13.149%
versus plain; offset with the fixed first-factor LR correction changes it by
-14.529%. Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed.
The broad research goal remains unmet.

Eight qualification checks pass; all240 selection and reporting scores and48
resets reproduce exactly. The two-rate robustness condition is independently
verified alongside selected-recipe gates. The fixed first-factor LR control is
history-informed and all forms train on fresh inputs. No old curved recipe is
reopened. [Full result](latent_offset_fit_results.md), [plan](latent_offset_fit_plan.md).

## H086 - Compact fixed2:4 sparse operator (NATIVE PASS; BACKEND INCOMPLETE)

Sixteen native checks pass, including48 independently audited tensor pairs and
36 exact activation-checkpoint pairs. The first CUTLASS inference case fails as
unsupported;11 hardware cases remain unexecuted. No training, timing conclusion
or promotion follows. Source and original failure are preserved.
[Report](compact_sparse_operator_results.md).

## H087 - Retire failed rational branch and prepare compact Git release

Archive rational BlockShuffle, its active recipe and candidate-only audit/tests.
Retain two model folders, five variants, eight recipes, the historical source
snapshots and scientific reports. Ignore local datasets/tensors/caches and
publish their content-hash inventory. Active-source changes are intentional;
frozen old receipts must be reproduced against recorded source snapshots.
[Release plan](release_cleanup_plan.md), [artifact policy](ARTIFACTS.md).

The next proposed mechanism is a coupled feature-pair activation using a bounded
rational rotation. No learning claim or novelty conclusion is established.
[Equations, potential failure modes and prior work](research_direction_2026_09_10.md).

## H088?H090 - Coupled pairs and shared local curves (CLOSED AT THIS BUDGET)

H088 fails its first BF16 independent-gradient qualification after 21 CPU checks;
zero fitting updates occur. H089's 32-evaluation diagnostic identifies the
residual-rounding boundary. H090 makes one explicit FP32-region recovery,
retains tolerances, passes all 27 checks and preserves H088's evidence.

The unchanged fresh FP32 grid completes 336 runs, 600 updates each, batch 256,
four analytic tasks, three seeds and two rates. Independent audit reproduces
all 672 scores exactly and verifies every selection, initialization and summary.
Learned/identity-start pair twists finish +1.298%/+0.715% MSE versus narrow
GELU; Bezier1P/grouped Bump2P finish +6.676%/+6.460%. All four fail the quality
and task-regression gates. Their fixed recipes are rejected, with no automatic
longer run, language allocation or new active model folder.

The learned twists beat fixed twist but take about 2.25 times narrow GELU's
native update time. A simple four-shift GELU control wins this small-model
comparison (-3.227% versus narrow GELU) while still losing to full GELU by
11.18%; retain it as a control. Learnable geometry is not demonstrated to
replace width. [Full result](neuron_geometry_results.md),
[precision investigation](neuron_geometry_precision_results.md).

## H091-H094 - Direct-input nonlinear pairs (CLOSED AT THIS BUDGET)

H091's partial-GELU draft was superseded without execution; its exact source and
plan remain in the archive. H092/H093 qualify direct input plus rational, Hermite
or trigonometric even/odd feature pairs at 350,208 weights. All25 unchanged checks
pass in an explicit single-process recovery after a Windows DLL import failure.
No numerical test or optimizer update ran in the failed launch.

The direct feature map preserves input distances before the final readout, but
the readout can remove directions. Transformer residual connections already
preserve an input route; this theorem is not evidence of language-model advantage.

H094 allocates a fixed300-step, batch256 screen:14 forms, four fresh synthetic
tasks, three seeds and two rates. Simple duplicate/linear/antipodal and direct-GELU
controls test whether the nonlinear basis earns its cost. No automatic larger
allocation follows. [Qualification](input_basis_lift_plan.md),
[frozen fitting plan](input_basis_fit_plan.md).

H094 completes336 runs in5.82 minutes and its independent audit reproduces672
scores exactly. Rational/Hermite/trig lower aggregate MSE7.21-9.31% versus
narrow GELU but lose4.15-6.56% to duplicated-even; all fail the frozen gates.
These fixed richer recipes are eliminated without a larger allocation.

## H095 - Fold the stronger even-feature control (UNVALIDATED LEAD)

The observed duplicated-even control is12.92% below narrow GELU and8.89% below
full GELU at300 updates. Folding V/W into V+W reduces inference weights to282,624
(76.04% below full); training used350,208 (70.31% below full). All12 checkpoint
output/Jacobian/MSE checks pass, maximum MSE change below1e-8, zero training updates.
Native inference improves from0.297 to0.261 ms but narrow GELU takes0.129 ms.
Retain this simpler mechanism for a distinct future hypothesis, not automatic
refinement or promotion. [Report](input_basis_results.md), [fold plan](input_basis_fold_plan.md).
