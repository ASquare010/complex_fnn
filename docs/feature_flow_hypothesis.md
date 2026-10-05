# Hypothesis: nonlinear feature mixing inside a compact FFN

Registered 2026-10-04 before implementation or candidate measurements. The
basis-readout campaign is terminal and retired. Its readout added no nonlinear
depth between B and Q and failed both dense quality margins. The completed
[activation-map diagnostic](ffn_map_result.md), using training activations only,
finds 8192-pair affine held-out R2 of 0.4485–0.6175 on TinyStories and
0.3206–0.5259 on WikiText. Rank-128 restriction of those affine fits loses only
0.0042–0.0275 R2. These sampled fits do not support a nearly affine FFN map.
They are not population ceilings, semantic-information measures, or proof that
all unexplained variation requires this new computation. Sample-size sensitivity
and train/held-out gaps are retained; no architecture was tuned on test data.

Question: can a small learned matrix repeatedly mix nonlinear responses inside
the FFN, allowing each response to depend on other evolving features, instead
of adding more correlated basis responses that only meet at a linear readout?
The global input/output matrices run once. Repeating the cheap internal operation
is different from repeatedly running the full input/gate/output FFN and its
backbone normalization, as the repository's earlier Looped BlockShuffle did.

## Fixed computation

Width 512, four independent FFNs. Group size 32, 16 groups. No cross-token state,
attention changes, routing or dynamic halting. For each token:

```text
z = P x                                    # full 512-by-512 matrix
phi(h) = scale * (h * SiLU(h) - mean)        # scalar reaction, fixed shape
psi(h) = tanh(phi(h))                       # bounded reaction used in updates
h[0] = z
for t = 0,1,2:
    h[t+1] = h[t] + blockdiag(M) * psi(h[t]) / 3
FFN(x) = Q * phi(h[3])                      # full 512-by-512 matrix
```

M contains 16 learned 32-by-32 matrices, shared across the three internal steps
within that layer, independently learned across layers. Phi after the last
mix is nonlinear: M and Q cannot simply merge into one linear basis readout.
Repeated updates allow feature interactions through a learned small graph.
P sees all residual features and Q reads all final features, but groups still
limit internal interactions. There is no known semantic locality, information
retention guarantee, or guarantee that repetition learns useful compositions.

Initialize M to exactly zero, P std .02 and Q std .02/sqrt(2*layers), using common
named up/down generators. All recurrent controls then start at the same function
Q*phi(Px). Compute Gaussian mean/variance in CPU FP64 using 128-node quadrature,
verify against 256 nodes at variance 512*.02^2. Mean is exactly variance/2 for
h*SiLU(h) under a centered Gaussian. Match expected full SwiGLU 1376 residual
variance through scale, not all correlations/distributions/logits. No pretrained
weights, fitted diagnostic maps or SVD bases enter training. Bounded psi avoids
quadratic growth inside recurrence; it does not bound M or guarantee stability.

Primary FFN weights 4*(2*512^2 + 16*32^2) = 2,162,688, 74.42% fewer than full
SwiGLU. Forward projection FLOPs/token 4,587,520 across all four FFNs: P/Q once,
M three times. Ordinary training projection work 13,762,560 plus 393,216 from
reconstructing the three M updates, total 14,155,776. Include all scalar SiLU,
tanh, reconstruction and reduction work separately; FLOPs are not speed evidence.
M is FP32, so its physical cost differs from BF16 P/Q projection work.

## Controls and precision

* One step: same M/weights, update h=z+M*psi(z), output Q*phi(h). Total nominal
  update time remains one. It tests added nonlinear depth against one larger
  update; the primary must beat it on both corpora.
* Diagonal three steps: actual learned 512 diagonal values, assembled into the
  same grouped matrix operation. 2,099,200 FFN weights, honestly smaller. It tests
  feature exchange versus local nonlinear evolution; primary must beat it on both.
* No updates: no M parameter, Q*phi(Px), 2,097,152 weights. Scalar-function control;
  its actual simpler execution/size is reported, not a candidate substitute.
* Cached three steps: same function/weights/precision with ordinary autograd.
  No NLL superiority over the same mathematical cached function is required.

All four start at the same initialized function as the primary; independent
controls learn from scratch. Compare full kernel SwiGLU 1376, native SwiGLU 1376,
GELU 2064. Compact kernel SwiGLU 352 and GELU 528 both exactly match primary weights.
Also register a variance-calibrated SwiGLU 352 control with down-weight std
multiplied by sqrt(1376/352), keeping its counts unchanged. This controls for the
candidate's matched initial residual variance. Select the strongest ordinary
compact across all three; an initialization improvement alone cannot establish
the proposed nonlinear feature-mixing advantage. All controls get the same
training windows/budget/rate and are retained.

P/Q use usual BF16 AMP. Initial z is BF16, convert once to FP32 inside the recurrent
operation; hidden updates, phi/psi, and grouped M GEMMs use FP32, autocast disabled.
Cast final phi once to BF16 before Q. Cached uses identical layouts/casts. TF32
stays disabled, with actual settings recorded. CPU FP64 references use FP64 inside.

## Exact gradient and memory execution

The custom discrete operation saves original z and M references, not all steps.
Backward reconstructs h0..h3 in FP32, then computes exact discrete backpropagation
in reverse. This is not a continuous ODE adjoint or approximate gradient.
With incoming final derivative g, begin with g*phi'(h3). At each t in reverse:

```text
dM += outer(incoming_state_gradient, psi(h[t])) / 3   # within each group, summed tokens
next_gradient = incoming_state_gradient
                + (M.T * incoming_state_gradient) * psi'(h[t]) / 3
```

Use phi'=scale*(SiLU(h)+h*SiLU'(h)) and psi'=phi'*(1-tanh(phi)^2). Gradients of
the same M sum over all steps. Return FP32 master M gradients and original-dtype
z gradients, only one final input cast. Reconstruct one layer's states at a time,
record real peak memory and time including temporary states. No higher-order
gradient claim. Torch-owned memory, existing Torch context/stream, validated
FP32/BF16 shapes; no unsafe raw launches or approximate differentiation.

Before profiling: independent CPU FP64 scalar/group equations, every z/M gradient,
finite differences, Gaussian quadrature, all five complete-model causality/locality/
counts/save-load/exact CPU optimizer recovery; initialization equality across
controls. CUDA FP32/BF16 all adjoints at complete/incomplete token/group shapes,
rounding and frozen interventions against ordinary autograd. Verify compact
control's variance calibration, not a claim of full distribution equivalence.

## Screens and decisions

After CPU/CUDA checks, eight hardware variants: three full, primary, one-step,
diagonal, no-updates and cached. Three alternating-order rounds on each corpus,
20 warmup/100 timed updates, maximum allocated peak/median throughput; record
cold compilation and reserved memory. Primary must pass against every full on
both corpora: >=20% memory reduction OR >=1.2x throughput, <=5% deterioration in
the other, reserved below 6 GiB. Retire a resource failure before language.
Controls cannot be substituted as post-hoc candidates.

If primary passes, integrate one readable complete Transformer and verify exact
prototype/production equivalence and shared-Trainer CPU checkpoint recovery.
Then eleven variants per corpus (full three, compact three, primary and four
mechanism/execution controls). Width 512, layers four, heads 16, context 256,
vocabulary 4096, batch eight, BF16, threads four, seed 101, 2000 updates /
4,096,000 targets, rate .0006, warmup 200, decay .1. Same backbone/prepared data/
sampled windows/full eligible validation as prior screens. Test loss unopened.
Primary needs <=1% cost against strongest full and >=1% gain against strongest
ordinary compact on BOTH corpora, plus better NLL than one-step and diagonal.
No choosing a different architecture per corpus or tuning a failed recipe.

At frozen checkpoints: zero M, remove M's off-diagonal values retaining its trained
diagonal, use one step while retaining original calibration, zero even hidden
coordinates in each internal psi response (not the residual h identity), and
training-mean FFN replacement (32 training batches, seed 2027). Repeat on controls
where meaningful. Changed statistics do not identify semantic roles; retrained
controls are required. Record all results and retirement decisions in result.md.
Survivors need equally tuned dense references, repeated paired seeds and fresh
independent longer-budget/resource/test confirmation under the unchanged goal.

## Prior work and limits of originality

[GLUs](https://arxiv.org/abs/2002.05202) already use multiplicative gates;
h*SiLU(h) ties the two inputs rather than inventing a new scalar primitive.
[Neural ODEs](https://arxiv.org/abs/1806.07366),
[Universal Transformers](https://arxiv.org/abs/1807.03819) and
[ODE-based Transformer analysis](https://arxiv.org/abs/1906.02762) establish hidden
evolution, recurrence, weight sharing and differential-equation views.
[StructuredFFN](https://arxiv.org/abs/2406.16450) establishes block-diagonal matrices.
The existing repository's Looped BlockShuffle already repeats FFN computation.
This tests a particular cheap internal grouped recurrence with P/Q only once;
that distinction does not establish novelty. It uses fixed discrete steps and
exact backpropagation, not an adaptive ODE solver, continuous adjoint or a claim
of a physical law. Verify closest implementations before any originality claim.
