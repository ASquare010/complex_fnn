# H084 - Separate internal affine gain, offset and deployment algebra

Freeze before implementation and new checkpoint interventions. H083 is complete:
both learned curves fail against equal-count affine and narrow GELU. Affine is
a diagnostic control, not a promoted candidate. This study investigates its
mechanism with zero optimizer updates; it does not repeat or extend fitting.
The previous goal turn made PROGRESS by completing and independently auditing
all 192 H083 cells. Original training and analysis handles are terminal.

## Questions and local derivations

Write a projection as P_o^-1 B2 P_m [G B1 x + b], where G is diagonal with
group-shared g=1+0.5 tanh(theta_a), and b repeats 0.5 tanh(theta_b) within each
first-factor output group. The diagonal entries of G lie in (0.5, 1.5).

1. G B1 is still a block-diagonal first factor with the same shape. Thus gain
   alone does not enlarge the represented function family when B1 is freely
   trainable. It changes parameterization and can change optimizer trajectories.
2. For fixed trained tensors, set B1'=G B1 and theta_a'=0, retaining b. This
   exactly preserves the function in real arithmetic. Removing gain without
   compensating B1 is a different operation and can change the function.
3. Alternatively use B1'=G B1, remove both curve controls and add the fixed
   output buffer c=P_o^-1 B2 P_m b. This preserves the projection in real
   arithmetic, with two factor multiplications and one output-bias addition.
   It is an evaluation copy, not an optimizer-equivalent training checkpoint.
   Caching c costs the sum of output widths: 2*2048+384=4480 float values per
   FFN (17,920 bytes in FP32). Do not count these buffers as learned weights or
   claim zero storage. Removing all 48 curve parameters leaves 350,208 stored
   factor parameters plus these buffers; folding gain alone leaves 24 effective
   offset parameters, even if the diagnostic module retains zeroed gain slots.
4. For a single bias-free SwiGLU FFN with differentiable zero-preserving
   projections U,V,D and SiLU(0)=0, F(0)=0 and DF(0)=0. This follows from the
   product rule: both factors of SiLU(U(x))*V(x) vanish at zero. It includes
   H082/H083 tanh/sine internal curves. This is a restriction at one point,
   not a distributional error bound, global vanishing-gradient claim, or a
   statement about a Transformer residual block with attention and residuals.
5. An internal offset can remove that restriction. A width-one/group-one
   witness sets U(x)=1/4, V(x)=x, D(x)=x, so F(x)=SiLU(1/4)*x and
   F'(0)=SiLU(1/4)>0. Use the actual grouped-factor/affine implementation for
   this witness. The claim is nonzero slope, not full rank in all trained FFNs.

The origin restriction was already discussed in H037 and H072. This is an
application to the new internal placement, not a new activation-family claim.
Featurewise affine modulation is established in [FiLM](https://arxiv.org/abs/1709.07871).
Our controls are fixed learned parameters rather than conditioning-network
outputs. [BitFit](https://arxiv.org/abs/2106.10199) studies bias-only fine-tuning
of pretrained models, distinct from this from-scratch structured-layer study.
The gated baseline follows the [GLU-variant literature](https://arxiv.org/abs/2002.05202).
None of those papers verifies this checkpoint mechanism or our research target.

## Fixed read-only allocation

Use exactly the 12 affine checkpoints selected in H083: four tasks, seeds
17/29/43, unchanged recorded selected rates. Pin all selected checkpoint and
selection/held-out/ablation records before execution, and H083's source, data,
result, analysis and final archives. Never write inside the completed H083 tree.

For each checkpoint evaluate all six modes on the same 4096 reporting examples:
original; reset gain only; reset offset only; reset both; compensated gain fold;
and full fold to two factors plus cached output bias. All resets act on all
three projections together. No per-projection search, new data, rate selection,
training, threshold tuning or activation fitting occurs. Totals: 72 reporting
passes, 294,912 example presentations, zero language targets and zero updates.
Original and reset-both MSE must reproduce H083 exactly in native FP32 CUDA,
TF32 off, batches of 256, four CPU threads, using the same MSE-sum accumulation.

For both folding modes, require FP32 reporting MSE relative difference <=1e-5
and CPU FP64 forward/input-gradient agreement with rtol=1e-10, atol=1e-10 on
the fixed probe below. These thresholds are declared before new scores; do not
silently relax them. BF16, fused kernels and training-gradient equivalence are
outside this study. A folded copy is not qualified for those modes.

CPU FP64 probe: one zero row and eight standard-normal rows, generator 9291,
shape 9x384. Scalar output probe is linspace(-0.7,1.1,numel(output)), reshaped
like output, with mean dot-product loss. Save original and both folded outputs
and input gradients for every checkpoint: 12 tensor cases, 24 paired checks.
The source checkpoint's FP32 tensors are cast to FP64 before computing the
fold. This tests algebra from the actual trained controls and factors.

For the original and the three reset forms, additionally evaluate the FP64
origin output and eight origin JVPs (directions from generator 9292, shape
8x384). Save those tensors. Reset-offset and reset-both must have exactly zero
origin outputs/JVPs; original/reset-gain are measured without a nonzero gate.
These finite probes complement the analytic proof and do not certify rank.

Construct the width-one offset witness once in FP64 and verify forward and
autograd input slope at zero against SiLU(1/4). Save the witness tensors and
weights. It uses no fitting data, optimizer or trained initialization.

For each reporting mode record MSE and residual mean per output coordinate,
using FP64 residual reductions from saved FP32 predictions. Record mean-error
energy and centered residual energy; check their sum against the FP64 residual
MSE to rtol=1e-12, atol=1e-12, and against the primary FP32-accumulated MSE to
rtol=1e-6, atol=1e-6. These summaries distinguish mean mismatch from remaining
error, without centering or refitting training labels. Save per-coordinate
residual mean, second moment and prediction tensor hash, not all predictions.

## Interpretation and decision

Report every task/seed/mode, geometric reset/original ratios, compensated-fold
errors, and the 2x2 factorial interaction E(no gain,no offset)-E(no gain,offset)
-E(gain,no offset)+E(gain,offset). This is a descriptive interaction of changes
to one coadapted checkpoint. It does not estimate causal effects of training
with or without a mechanism. Compare existing H083 plain and both narrow
errors as context; do not replace trained controls with reset models.

No activation, affine recipe, full-model resource run or language trial is
promoted by this diagnostic. A distinct later fitting hypothesis must separate
gain-only, offset-only, full affine and fixed-function controls, include narrow
GELU, equal budgets and both rates, and satisfy meaningful quality/resource
gates. H083's failed curve recipes and old outer-affine failures remain closed.
No next training allocation is implied by these observations alone.

## Execution, preservation and closeout

Use UV and one bounded worker, CUDA only for reporting passes. Use an isolated
results/latent_affine_mechanism_v1 tree with algebra.py, study.py and launch.py
beside the evidence. Pin the prior 128 scientific sources plus those three new
sources (131 total), the prior 72 plans and this plan. Save source archive,
environment, git state, PID/UTC/log/return records. Write completed tensor/JSON
artifacts with fsync, ZIP/readback validation. Stop and preserve any failure;
no automatic retry, repair, extra probe or tolerance change follows.

Independently verify every saved case and checkpoint hash, all mode counts,
FP64 comparisons, moment decomposition and original/reset-both anchors; repeat
the two new uncompensated gain/offset reporting modes for all 12 checkpoints
without updates, exactly. This adds 24 passes / 98,304 examples. Verify the
scope of the math against the source, not just a passing flag. Update the
report, current state, overview, activation guide and ledger, retaining three
active model folders, six variants and nine recipes. The full research goal
remains unmet regardless of diagnostic success.
