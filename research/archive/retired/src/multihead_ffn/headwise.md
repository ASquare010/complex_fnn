# Router-free headwise SwiGLU

`headwise_swiglu` is a calibrated, single-subnetwork control in the same
architecture folder as the parallel multi-head reference. It is implemented by
`CalibratedHeadwiseFFN` in [headwise.py](headwise.py). It is a simplification of
the prior-art-inspired multi-head family, not a claimed new activation primitive.

## Equation and parameter budget

For a token x, compute q = A*x and split q into H heads of width dh=d/H.
Each head computes

    z_h = D_h [SiLU(U_h q_h) * V_h q_h]
    f(x) = B concat_h(z_h).

A and B are learned d-by-d matrices. U_h and V_h have de-by-dh weights;
D_h has dh-by-de weights. The code stores the private weights in transposed
layout for batched contractions. Every matrix is bias-free; there is no router.
The exact count is

    P = 2*d^2 + 3*H*dh*de = 2*d^2 + 3*d*de.

At d=384, H=24, dh=16, de=48, this is 350,208 per layer and 2,801,664 over eight
layers. Total model weights are 9,099,648. Both equal the BlockShuffle and
calibrated-narrow counts; FFN reduction versus full SwiGLU is 70.3125%.
Logical FFN matrix work is 5,603,328 forward FLOPs per token across eight layers,
excluding nonlinearities and other non-matrix work. This is not a speed claim.
There are H*de=1,152 hidden features per token, versus 2,304 for H046's parallel
model. No cached dense expansion or copied training weights are used.

## Initialization and limits

A is orthogonal; B is orthogonal times c=1/sqrt(2*layers). Their norm identities
are exact at initialization: ||Ax||=||x||, ||Bx||=c||x||. They are ordinary learned
matrices thereafter and are not constrained to remain orthogonal.
Private gate/value standard deviation is 0.02*sqrt(H); private down standard
deviation is 0.02*sqrt(h_full/de), where h_full=floor(8*d/3). These are H045's
calibrated scales with one subnetwork. The down residual scale is carried by B.
No optional finite-head moment correction is applied.

Under isotropic unit-variance inputs and independent private weights, gate/value
preactivation variance is dh*H*0.02^2=d*0.02^2. Down scaling approximately matches
full SwiGLU output variance if gated-feature second moments agree. Finite heads
produce a correction, so this is not exact distributional equality. The
[budget analysis](../../../../multihead_budget_geometry.md) gives the Gaussian
leading-order calculation. Calibration is specified before validation scores.
Changing parameter scales also changes uniform AdamW's optimization geometry.

## A structural limit that a scalar activation cannot remove

Define G(q)=B concat_h(g_h(q_h)), holding the mixers fixed and treating q as the
input coordinates. Every output coordinate is a sum of head-local functions.
For any twice-differentiable head functions and two different heads h and k,

    d^2 G_j / (d q_h,a d q_k,b) = 0.

Proof: differentiating the sum with respect to q_h,a leaves only the h term;
that term does not depend on q_k,b. This remains true if each head learns an
arbitrarily expressive scalar activation. It also holds for H046's parallel
head-local router, since its router sees only its own head.

Thus shape flexibility and cross-head interactions are different architectural
properties. For a fixed q coordinate system, a target q_h,a*q_k,b has mixed
partial 1 and cannot equal an additive head map on an open set. This is a
fixed-coordinate statement: a learned A can change the coordinates, and later
Transformer layers can combine them. It is not an impossibility theorem for the
trained network, an ordering against BlockShuffle, or a diagnosis of the measured
NLL. The generic FFN Jacobian has no positive global lower bound.

## Evaluation scope

The [H047 protocol](../../../../headwise_screen_plan.md) fixes three rates,
200 steps, complete validation targets and all required controls. It first
requires exact preservation of all 30 older models, an independent FP64 loop
and derivative check, common initial non-FFN tensors, Gaussian RMS qualification,
and ten finite GPU updates on a fixed training batch. Both runtime and quality
must be measured. Removing routing while doubling head width is a matched-budget
recipe comparison, not an isolated router ablation.


The [completed H047 result](../../../../headwise_screen_results.md) gives
best NLL 6.021328 and peak 638.37 MiB. Memory passes; quality fails. All 30
older variants remain exact and the full suite passes 220 tests. This local
recipe is not promoted to longer training.
