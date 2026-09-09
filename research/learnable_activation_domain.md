# Learnable activations: what can adapt, and what it costs

**Corrected result:** At the same peak LR 0.0012, the
[three-seed comparison](affine_rate_replication_results.md) finds only
0.101% lower mean NLL with affine and wins in 2/3 seeds. This fails the frozen
material-benefit gate. The earlier 1.755% advantage compared different rates.
Only native rational BlockShuffle remains an active learned-activation candidate,
with memory still unresolved. The affine and shifted implementations and their
proofs are [historical alternatives](archive/README.md); they have not earned
promotion over the stronger plain control. See the [current rational notes](../src/rational_blockshuffle_ffn/model.md).

A conventional FFN learns its projections while keeping an activation such as
GELU or SiLU fixed. A learnable activation also trains the shape of that scalar
function. For example, one feature group can learn a gentler negative tail,
while another learns a bend around moderately positive inputs. The task loss
supplies the training signal; no labels for the desired curve are needed.

This adapts to the training distribution. It does not automatically find the
best function for arbitrary future data. Generalization, regularization, optimizer
conditioning, sample coverage and hardware cost still matter. A curve shared
within a group stays the same function across inputs after training; an
input-conditioned activation would be another model with additional computation.

## The design space

| Family | What learns | Useful bias / main limitation | Primary source |
|---|---|---|---|
| Parametric slopes and adaptive piecewise linear functions | Slopes, hinge weights and locations | Cheap shape changes; a small number of hinges limits detail | [Agostinelli et al., 2014](https://arxiv.org/abs/1412.6830) |
| Learned splines | Local knot values and possibly locations | Local flexibility; sparsely visited intervals receive weak evidence | [Unser, 2019](https://jmlr.csail.mit.edu/papers/v20/18-418.html) |
| Rational functions | Numerator and denominator coefficients | Smooth flexible shapes; denominator constraints and tail behavior matter | [Molina et al., PAU](https://arxiv.org/abs/1907.06732) |
| Periodic residual functions | Oscillation scale/frequency in suitable parameterizations | Periodic inductive bias; not automatically useful for language | [Ziyin et al., 2020](https://arxiv.org/abs/2006.08195) |
| Edge-wise learned functions | A univariate function on each connection | More localized expressiveness, potentially substantial evaluation/storage cost | [Liu et al., KAN](https://arxiv.org/abs/2404.19756) |
| Our bounded shifted Bezier residual bank | Quadratic controls plus shared centers/slopes | A few smooth changes in different input regions; finite bank and saturating coordinates |

Learned activations, including rational functions, are established prior art.
Unser's spline representer result concerns an objective with a particular
functional regularizer; it does not prove that arbitrary learned activations
are universally optimal or easy for gradient descent to discover. KAN changes
edge parameterization and is not the same operation as replacing a node
activation in an otherwise unchanged Transformer. Function-fitting or PDE
results do not establish language-model superiority.

## Directly related Transformer work

[KAT / Group-Rational KAN](https://arxiv.org/html/2409.10594v1) already combines
group-shared rational functions, variance-aware initialization and GPU kernels
inside vision Transformers. Its default rational degrees are 5/4 and its
reported experiments concern vision. Our constrained 3/2 residual and gated
compressed bases differ, but group sharing and rational Transformer activations
are clearly not new ideas. Its emphasis on efficient execution is directly
relevant to the measured eager overhead here.

[Wang et al., May 2026](https://arxiv.org/html/2605.26647v1) study fixed learned
activation mixtures and token-dependent mixtures inside conventional and gated
FFNs, including one-branch, two-branch and paired-product forms. Their language
experiments span 0.12B to 2B models. This is a particularly relevant comparator
for the user's idea of mixing small nonlinear branches. Their finite-width
separation statements use specific activation dictionaries, biases, domains
and a function/derivative norm. They do not establish that our constrained
smooth residuals train faster or are optimal for all data. An input-dependent
mixture would require a separately budgeted comparison here; it is not what
our group-shared variants implement.

Both papers reinforce the need to distinguish local implementation experiments
from a verified new primitive. These sources were audited during the frozen
screen; they do not change its allocated trials or selection rules.

## Why apply small shapes to these two FFNs?

Calibrated narrow SwiGLU is the stronger compressed reference on the completed
WikiText screen. BlockShuffle is the stronger compressed architecture in the
larger three-seed TinyStories study. Neither wins everywhere. We test the same
initial two shape families on both, preserving their projection initialization and
optimizer treatment. The new correction starts at zero, so a poor starting
shape cannot explain away a failed comparison.

Group sharing spends O(GK) or O(G) shape weights rather than one new function
per input-output edge. Each layer retains independent shapes. Projection and
gating still supply interactions between input features. More scalar curvature
alone cannot recover every interaction removed by a narrow or factored matrix.

The activation implementations are in [learnable_activation_ffn](archive/retired/src/learnable_activation_ffn/README.md).
The [model notes](archive/retired/src/learnable_activation_ffn/model.md) prove bounded
activation slopes under explicit constraints. These upper bounds do not prevent
vanishing gradients, establish a well-conditioned full FFN, or guarantee rapid
training of complex patterns. The [screen protocol](learnable_activation_plan.md)
measures whether the added capacity earns its cost.

## What would make this research convincing?

A useful result must beat the calibrated conventional control at matched new
tuning effort, preserve the parameter target, and survive independent seeds and
corpus changes. Measure learned curves and slopes alongside loss, activation
RMS, clipping, memory and speed. If a richer shape wins, ablate learnable versus
fixed coordinates and a simpler affine correction before attributing the gain
to complex curvature. If it loses, keep that result and identify whether the
shape learned appreciably, whether optimization failed, or whether execution
cost dominates. No claim of a new primitive follows merely from combining
known components.

## What this repository measured

The [four-variant screen](learnable_activation_results.md) finds a useful
single-seed rational BlockShuffle result: 1.216% lower NLL than its selected
base, with 320 extra weights. Its eager memory cost is excessive. The
[compiled execution audit](activation_execution_results.md) passes local memory
and derivative checks, but a [full training repeat](activation_training_repeat_results.md)
loses the quality gain. This directly illustrates why a flexible function and
a local derivative bound are not guarantees of easy optimization.

The [six-trial affine retraining ablation](affine_activation_results.md) now
finds NLL 5.894765 for BlockShuffle with just128 added weights and718.08MiB
training peak. This matches the richer rational result and passes all frozen
short-screen gates. Narrow affine loses, showing that a learned correction is
not a universal improvement. The real-input rational affine-energy fraction is
96.388%, lower than the 99.768% uniform-grid fit. Reset costs are tiny, so the
training mechanism remained unresolved; the later same-rate replication above
substantially reduces the apparent advantage.

The additional affine control learns `SiLU(z) + a*z + b`, spending two weights
per group. Through the multiplicative gate this supplies linear and bilinear
paths. The original bias-free SwiGLU layer has zero Jacobian at the origin;
the learned offset can make that Jacobian nonzero. This is a precise layer-level
expressivity extension, not a guarantee of complex-pattern learning. See the
[proof](archive/retired/src/learnable_activation_ffn/model.md#h037-affine-correction-and-an-explicit-linear-path)
and [frozen ablation](affine_activation_plan.md).

## Longer-budget evidence and its limits

The [three-seed WikiText study](affine_activation_replication_results.md)
finds mean NLL 4.754207 for affine BlockShuffle, 1.755% below selected plain
BlockShuffle, with only 128 extra weights. All frozen quality/memory gates pass.
The recipes use different selected learning rates; the
[completed same-rate control](affine_rate_replication_results.md)
reduces the measured mean gain to 0.101%. This does not establish an activation-specific training cause.

[Removing the correction after training](affine_activation_removal_results.md)
costs less than .020% NLL in every seed and preserves quality gates. That result
motivates studying the learned common weights and optimization path. It does
not show that the final scalar shape is necessary for the observed performance.
No all-data optimum, convergence, training-speed improvement or full-network
nonvanishing-gradient guarantee follows. The two richer compiler variants
remain failed training backends despite passing local derivative checks.


## Learned shape and learned interactions are different

Changing a scalar activation adjusts how one projected feature responds. It
does not automatically introduce dependence on features that the subnetwork
never sees. For fixed head coordinates, a headwise FFN with a linear output
mixer is a sum of head-local functions; every mixed second derivative between
distinct heads is zero, even with arbitrary learned scalar shapes. The
[headwise notes](archive/retired/src/multihead_ffn/headwise.md) prove this statement and
explain its limits: learned input mixers change the coordinates, and stacked
layers can create interactions. This is not a whole-network expressivity
impossibility or an explanation inferred from NLL alone.

This distinction matters when choosing the next experiment: more activation
coefficients may be ineffective if feature mixing or optimization is the
limiting factor. In this repository the corrected three-seed affine gain is
0.101%, so additional shape complexity has not earned promotion on that result.


The [structured-headwise budget proposal](structured_headwise_budget.md) makes
this concrete: fewer mixer weights would permit 64-coordinate nonlinear heads
at the same FFN budget as the earlier 16-coordinate control. This is an
untrained composition of established components. Wider private subnetworks
and restricted mixers trade different kinds of capacity; no dominance follows.
A flexible activation should be tested as a separate extension only after the
base interaction structure earns its cost. H050's longer native comparison
also shows that the earlier quality advantage depends on training duration.


A [stronger square-mixer witness](headwise_hessian_obstruction.md) now closes
part of the learned-coordinate caveat for a single FFN. Three quadratic outputs
with incompatible input Hessians cannot share any nontrivial head partition,
even after freely learned square input/output mixing and arbitrary head-local
functions. Full SwiGLU represents that witness with 2*(d+1) hidden units.
The proof and CPU construction do not establish a language-model loss bound,
a whole-Transformer limitation or novelty. They show precisely why learning
scalar activation shapes alone cannot guarantee the best function for every data.


## A token-conditioned prototype: local proof, rejected fitting recipe

[H067](token_activation_results.md) now implements one scalar residual router per
token/layer and a static control, outside the active model factory. The value
branch is v + 0.25*tanh(w^T*x+b)*SiLU(v), followed by the existing SiLU gate and
output mixing. It starts at the plain model and adds 3,080 parameters at d384/L8.
Its [exact-function proof](token_activation_theory.md) does not imply a practical
approximation or training advantage. All 21 local checks pass. The subsequent
[H068 fitting screen](token_activation_fit_results.md), with seven target families,
three seeds and equal two-rate budgets, improves error by only 0.1213% over plain
and 0.0628% over its static control. Static improves 0.0585% over plain. Both miss
the frozen 2% gain gate and remain outside the active factory. Dynamic's small
FP32 update time is 6.080 ms versus plain 5.031 ms; this is not a full-model
resource qualification. Being able to represent a new exact function does not
guarantee that useful functions will be learned more efficiently.

## Internal placement: completed synthetic comparison

H082/H083 place learned curvature between the two BlockShuffle factors, before
the middle permutation. They leave the outer SwiGLU in place. Eight groups share
two scalar controls each, independently for up/gate/down:48 extra weights per
FFN, or384 across eight FFNs. This is a different placement from the earlier
outer activation/value-branch experiments; it is currently isolated research.

With a=0.5 tanh(theta_a) and s=exp(log(4) tanh(theta_b)), the scalar function is

    phi(z) = z + a*s*q(z/s)^2,  q=tanh or sin.

Both controls start at zero, so the initial FFN exactly matches plain. Scale has
zero gradient until amplitude departs from zero. A same-count affine correction
tests whether learned gain/offset suffices. The sine form is a constrained
[Snake-related control](https://arxiv.org/abs/2006.08195), not a new periodic idea.

In real arithmetic, the tanh scalar slope lies in approximately[0.6151,1.3849];
the sine slope lies in[0.5,1.5]. These local bounds preserve a nonzero scalar
input slope, not all matrix directions, parameter gradients or a full network's
conditioning. Both corrections also have a constrained quadratic Bezier form.
See the [derivation and limits](latent_activation_theory.md).

The original adapter failed before fitting. One documented binding-only recovery
passes all eight unchanged checks and completes 192 runs. Tanh and sine reduce
aggregate held-out MSE by 5.76% and 7.07% versus plain; equal-count affine reduces
it by 15.03%. Both curves lose to affine and narrow GELU and take about twice
plain's standalone update time. Neither earns full-model resources or language
training. All selected held-out and reset scores reproduce exactly.

Resetting the learned curves worsens final error, so the models use them. That
does not establish that curvature is better than gain/offset or explain training
causally. The affine control itself remains behind narrow GELU. The outcome is
evidence for studying the simpler mechanism before adding more curve complexity;
it does not authorize another tuning sweep. [Completed comparison](latent_activation_recovery_results.md).

## Separating an affine function from learned curvature

The [H084 checkpoint analysis](latent_affine_mechanism_results.md) shows internal
affine gain can be absorbed into a block-diagonal factor without increasing its
shape or function family. An internal offset can change the isolated SwiGLU's
zero-input output and derivative. This distinction explains what each parameter
can represent, but does not explain its optimization effect causally.

Both components matter to the selected trained checkpoints: deleting gain or
offset without compensation multiplies aggregate error by 2.972 or 1.843.
Compensated folding preserves FP32 error within a relative 1.7e-8. It adds cached
bias storage and has no measured runtime or BF16 qualification. A useful learned
parameter need not enlarge the function family, and folding it after training
does not reproduce training without it. Controlled fitting remains necessary
before claiming a better mechanism; narrow GELU still wins the H083 comparison.

## Controlled internal-offset training result

H085 completes 240 fresh runs and all independent score checks pass.
Learned offset changes selected aggregate held-out MSE by -13.149%
versus plain; offset with the fixed first-factor LR correction changes it by
-14.529%. Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed.
The broad research goal remains unmet.

Partial controls retain the original affine arithmetic with the unused controls
fixed as zero buffers. The experiment includes both gain-only and fixed-LR
controls, and tests the offset effect at each learning rate. All final reporting
scores, including unselected rates, reproduce exactly. These trained comparisons
address a question that resets alone cannot answer, within the stated synthetic
scope. [Detailed controlled comparison](latent_offset_fit_results.md).
