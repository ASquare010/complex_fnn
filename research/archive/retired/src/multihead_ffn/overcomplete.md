# Overcomplete structured-headwise SwiGLU

**Status: registered variant; constructive and numerical checks pass, native
training memory passes, H056 language quality fails. REJECTED at the tested budget.** [Frozen H053 plan](../../../../overcomplete_qualification_plan.md).
The [module](overcomplete.py) uses exactly the BlockShuffle parameter budget
and has a constructive interaction result beyond the square-headwise restriction.
It is not a verified new primitive or a general dominance theorem.

## Architecture and parameter count

For x in R^d, compute q=A*x in R^m, split it into H heads, and return

    F(x) = B * concat_h D_h[SiLU(U_h*q_h) * (V_h*q_h)].

A and B are existing two-factor BlockShuffle maps with G groups and intermediate
width d. Private U/V have shape k by m/H and D has shape m/H by k. All maps are
bias-free. There is one subnetwork per head, no router or added activation shape.

    P = 2*d*(d+m)/G + 3*m*k.

At d=384, m=512, G=16, H=8 and k=200, mixer weights total 43,008 and private
weights total 307,200: **350,208/layer**, or **2,801,664 over eight layers**.
That is 70.3125% fewer FFN weights than full SwiGLU. There are 1,600 hidden
features. H054 registers this module as `overcomplete_headwise_swiglu` in the shared
decoder; there are now 32 variants. ModelConfig.groups denotes H, mixer groups
are 2*H and expanded width is 4*d/3. The experiment explicitly uses hidden=200.

## A constructive representation using the actual factors

The [square-mixer obstruction](../../../../headwise_hessian_obstruction.md)
excludes the vector quadratic with first three outputs

    Q1(x) = .5 * sum_i x_i^2,
    Q2(x) = .5 * sum_i (i+1)*x_i^2,
    Q3(x) = .5 * (sum_i x_i)^2,

and remaining outputs zero, with i=0,...,383. Here is an explicit representation
inside this overcomplete module. All matrix/array indices below are zero-based.
The construction specifies both structured mixers; dense unrestricted A/B are
not substituted for them.

1. Set A's first sixteen 24x24 blocks to identity. After the fixed middle
   shuffle, call the sixteen blocks of 24 coordinates z_a. In each 32x24
   second-factor block, set row 0 to all ones. For h=1,...,6, set rows
   (2h,2h+1,16+2h,17+2h) to coordinate basis rows
   (4(h-1),4(h-1)+1,4(h-1)+2,4(h-1)+3). All other rows are zero.
   The output unshuffle places all sixteen block sums at even positions
   0,2,...,30 of head 0. Heads 1,...,6 each receive 64 distinct original
   coordinates, covering all 384 exactly once. Head 7 is zero.

2. For each coordinate z in heads 1,...,6, use two signed hidden units with
   both gate and value linear forms z and -z. Their sum is exactly z^2, since
   SiLU(z)*z + SiLU(-z)*(-z) = z^2. Set their down coefficients to .5 in
   local output 0 and .5*(i+1) in local output 32, where i is the original
   coordinate index given by the known shuffle. Each such head uses 128 of
   its 200 units. In head 0, sum its sixteen block sums and use one signed
   pair with down coefficient .5 into local output 1. This produces Q3.

3. B's first factor has sixteen 24x32 blocks. For h=1,...,6, set entry
   (row 0,column 0) to one in blocks 2h and 2h+1. Also set entry (2,1) in
   block 0 to one. In B's second factor, set block 0 entries (0,2h) and
   (16,2h+1) to one for h=1,...,6, and block 1 entry (8,8) to one.
   All other entries are zero. After the output unshuffle, output coordinates
   0/1 sum the six Q1/Q2 contributions and coordinate 2 receives Q3.

Thus the proposed constrained module represents Q exactly in real arithmetic,
using 770 active signed hidden units. This function is absent from every single
square-mixer additive headwise layer in the earlier theorem, regardless of its
head-local activation shapes or hidden widths. It establishes one represented
function outside that family; it does not show that either family contains all
functions of the other, that gradient descent finds these weights, or that an
entire Transformer has this limitation. The sparse construction does not prove
that all learned weights are necessary, optimal or well conditioned.

[FP64 checks](../../../../../results/overcomplete_qualification_v1/cpu.json) use the actual
module, 31 Gaussian inputs and 24 Hessian-vector products at a nonzero point.
Forward relative L2 error is 1.718e-16; maximum Hessian-vector error is 3.740e-14.
The [script](../core/overcomplete_qualification.py) and saved witness weights
make the permutation and every coefficient inspectable.

## Initialization with correlated overcomplete coordinates

Initialize A's first square blocks orthogonally and tall second blocks with
orthonormal columns. Their product, including permutations, satisfies A^T A=I.
For B, the wide first blocks have orthonormal rows and the second blocks are
scaled orthogonal, giving B B^T=c^2 I, c=1/sqrt(2*layers). The existing factor
initializer's total gain is .02*sqrt(m)*residual_scale, so supplying
1/(.02*sqrt(m)) for A and c/(.02*sqrt(m)) for B gives these gains exactly in
real arithmetic. This is an initialization property, not a maintained constraint.

For Gaussian x with covariance I, q has covariance A A^T, a rank-d projector.
Its coordinates are correlated and their mean variance is d/m. Write Sigma_h
for head h's covariance, rho_h=trace(Sigma_h), and nu_h=trace(Sigma_h^2).
Our shuffle gives each head four rows of each tall block. Under the randomized
orthogonal initializer E_A[rho_h]=d/H=48, but individual heads need not equal 48.

Set private U/V standard deviation sigma=.02*sqrt(H)=.0565685. Conditional on q,
the independent gate/value weights produce independent centered normals with
variance r_h=sigma^2*||q_h||^2. Consequently

    E[r_h]   = sigma^2*rho_h,
    E[r_h^2] = sigma^4*(rho_h^2 + 2*nu_h).

The ensemble mean gate variance matches the full dense value .02^2*d=.1536.
This does not exactly match the gate distribution: the head norm is a different
Gaussian quadratic form from ||x||^2. Let psi(r)=E[SiLU(Z)^2], Z~N(0,r), and
kappa_h=E_x[r_h*psi(r_h)]. Averaging over independent private down weights of
variance tau^2 gives the exact conditional output-energy formula

    E_private,x ||F(x)||^2 = k*tau^2 * sum_h kappa_h * ||B[:,head_h]||_F^2.

The symmetric-Gaussian identity gives r/4 <= psi(r) <= r/2, hence
E[r_h^2]/4 <= kappa_h <= E[r_h^2]/2. We use the simple dense-width calibration
tau=.02*sqrt(1024/k)=.0452548, and qualify its actual scale numerically rather
than claiming an exact variance or optimization match.

Across seeds 17/29/43, FP64 isometry errors are at most 1.333e-15 for A and
6.939e-17 for B. The realized head covariance traces range from 46.709 to 49.451
and effective ranks rho_h^2/nu_h range from 60.256 to 61.701. On 4,096 Gaussian
inputs per seed, output RMS relative to same-seed full SwiGLU is
**1.01356 / 1.01821 / 1.01643**, within the frozen [0.8,1.25] band.
This checks initialization scale, not loss convergence, a Jacobian lower bound
or trained stability. The bias-free SwiGLU origin Jacobian is still zero.

## Native numerical and memory qualification

At batch 16/context 128 on the RTX 4070 Laptop GPU, the module's native BF16
forward relative error against its FP32 execution is **0.007889**. The maximum
input/parameter gradient relative error is **0.008171**. All parameters receive
nonzero initial gradients; ten fixed synthetic-target AdamW steps stay finite.
These are arithmetic checks, not a synthetic or language-quality comparison.
[GPU record](../../../../../results/overcomplete_qualification_v1/cuda.json).

| Isolated eager FFN | Allocated forward/backward peak MiB |
|---|---:|
| Overcomplete headwise | 69.366 |
| Full SwiGLU | 57.626 |

The candidate is about **20.37% higher** in this isolated memory measurement,
despite fewer weights. Both use the same shapes and native BF16, with no optimizer
state in that measurement. CUDA workspace/baseline allocations are retained.
This is neither the complete Transformer's memory gate nor an inference-speed
claim. Whole-model memory qualification is required before spending a language
training budget; no gate recomputation or custom kernel was added here.

## Prior art and next boundary

[Monarch](https://arxiv.org/abs/2204.00595) establishes products of structured
block matrices, and [Flash Multi-Head FFN](https://arxiv.org/abs/2512.06989)
establishes headwise subnetworks and specialized execution. Combining them is
not by itself a verified new primitive. The explicit witness and covariance
accounting are scoped local results, with no established literature-priority
claim. A separately frozen integration/memory check and balanced quality screen
must precede promotion. Earlier failed headwise and learnable-activation studies
remain valid for their recorded settings.

## Complete-model integration result

[H054](../../../../overcomplete_integration_results.md) preserves all 31 older
model signatures and matches every new layer to the qualified standalone
initializer. Native BF16 complete-model checks and 20 fixed-batch AdamW updates
stay finite, but allocated peak 730.389 MiB exceeds full GELU 660.108 MiB by
10.647%, missing the 10% allowance by 4.270 MiB. Full SwiGLU uses 706.733 MiB.
No validation targets are scored and no quality screen is earned. The existing
native gate-recomputation method is a separately justified execution control;
it was not applied in H054.

## Native gate recomputation

[H055](../../../../overcomplete_recompute_results.md) reuses the existing
RecomputedSwiGLU backward. Set TrainConfig.recompute_gate=True and
TrainConfig.gate_recompute_method="native"; the shared trainer applies the
same setup as qualification. Eager remains the default. Parameters,
initialization and the represented function do not change.

Complete-model peak drops from 730.389 to **680.014 MiB** (-6.897%), passing both
full-control memory allowances. Initial BF16 parameter gradients match eager
exactly; all 9,099,648 final weights and the 20-step repeated-batch history also
match in this run. All 243 tests pass. These local equivalences are not a
universal bitwise guarantee. No validation targets are scored; a balanced
language-quality screen is now earned, with actual screen memory still checked.

## Held-out language-quality screen: rejected

The completed [H056 screen](../../../../overcomplete_screen_results.md) uses
three rates .0003/.0006/.0012, each with seed 17 and 200 steps on pinned WikiText-2.
NLL is **6.303587 / 6.134152 / 6.043891**. The selected final rate loses full
SwiGLU by 2.305%, full GELU by 2.800% and calibrated narrow by 2.223%; all quality
gates fail. Selected square headwise is also 0.375% better. The candidate beats
that headwise control at the two lower rates, without winning recipe selection.

Actual allocated training peak **676.931 MiB** passes both full-control limits;
70.3125% fewer FFN weights and 42.170% fewer total weights remain. All three
trials and final Adam states are finite; 245 tests pass. This screen uses the
existing factor fan-in LR policy (input 8/8, output 8/(32/3), private tensors 1)
and .1 parameter decay, differing from H055's uniform-rate execution check.

Initialization RMS matching does not preserve trained scale. On the fixed
validation-input diagnostic, layer-0 FFN RMS grows from .012844 to 13.7203,
versus full SwiGLU 3.8765. Later FFN parameter-gradient norms are smaller but
finite and nonzero. These are descriptive, parameterization-dependent measures;
they do not prove a causal normalization problem or vanishing gradients.

The exact quadratic witness remains useful. It does not guarantee easy learning
of that function or lower language NLL. No longer run or added activation is
earned by this failed recipe. Keep the module for reproducibility and scoped
mechanistic analysis; the full research goal and verified novelty remain open.

## Checkpoint geometry: no causal repair established

[H057](../../../../residual_geometry_results.md) completes 14 CPU FP32 and six
native BF16 fixed training-prefix probes, without optimizer updates or held-out
scoring. Every local RMSNorm VJP matches the derived finite-epsilon formula.
The candidate's later native local gain is .7071 times full SwiGLU and .7742
times narrow, but full GELU's still smaller gain accompanies better language
quality. Lower-rate candidate checkpoints have larger gains and better matrix
conditioning yet worse NLL. These counterexamples prevent a sufficient causal
interpretation or an automatic normalization/optimizer repair.

The exact value/down gauge rescaling changes associated parameter-gradient
norms by 1/8 and 8 with zero measured logit, loss or predicted-gradient error in
the selected trained copy. Original weights are restored exactly. Raw gradient
magnitudes therefore do not establish lost functional sensitivity. The composed
B maps remain numerical rank 384 while their condition numbers reach 129-3978;
these are linear-factor spectra, not whole-FFN Jacobians. All 248 tests pass.
The diagnosis is finished; H056's failed quality gates remain authoritative.
