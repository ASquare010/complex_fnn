# Paired readout: directed responses with one learned projection bank

Registered before implementation or measurement, after readout-reuse v5 passed
numerical checks but failed all six resource comparisons. Its fixed tied control
was cheaper than its learned grouped readout, motivating a different mathematical
allocation. This is a new experiment, not promotion of a previous control.

## Mechanism and limits

For each layer independently, P has shape [1152,512]. Compute z=P*x and the
same calibrated scalar response r=a*(z*SiLU(z)-mu) as the readout-reuse study.
For each pair, u_even=-r_odd and u_odd=r_even, then y=alpha*P.T*u,
alpha=1/sqrt(2*layers). Thus each detector writes through its partner's direction.
P is the only learned FFN weight bank; pairing and signs are fixed buffers.
No learned group matrices, biases, token mixing, recurrence or backbone changes.

Writing the fixed pairing as J, the Jacobian is alpha*P.T*J*diag(phi')*P.
Unlike signed diagonal tying, it is generally nonsymmetric. J is skew-symmetric,
but the nonlinear FFN's Jacobian need not be skew-symmetric. No Hamiltonian,
energy-conservation, stability or gradient-flow guarantee is claimed.
This changes which learned output directions features use; it adds no nonlinear
interaction depth after the scalar response. P must learn both useful detectors
and useful partner directions, which could impair quality.

The research hypothesis is that this directed pairing retains useful readout
freedom without group-matrix overhead, and its wider feature bank outperforms
compact ordinary dense FFNs. More features alone do not prove that hypothesis.

Initialization: named P normal(.0,.02); unchanged backbone initialization.
q=512*.02^2, mu=q/2, and 128-node quadrature chooses a so response variance is
(1376/512)*(512/1152)*q*E[SiLU(z)^2]. Verify 256 nodes. Pairing preserves response
norm, but scalar calibration does not guarantee complete-model variance matching.

Across four layers: 2,359,296 FFN weights, 72.09% fewer than 8,454,144 full
SwiGLU weights. Forward projection work: 9,437,184 FLOPs/token. Count both
uses of P and both master-gradient contributions. With saved projected z and
mixed response u, projection training work is 28,311,552 FLOPs/token, plus
scalar/index operations. Every extra saved tensor counts toward training VRAM.

## Execution and controls

Primary execution saves native projected z and BF16 u inside an explicit
autograd function. Compute scalar response and paired scatter in one FP32 CUDA
kernel, using Torch's existing context/current stream and native casts. Backward
uses native matrix products, partner gathering and native SiLU backward in the
operation order validated by readout-reuse v5. No custom BF16 casts or relaxed
precision. CPU/FP64 uses ordinary autograd. All modes provide an ordinary compute
path as the reference. Unsupported raw layouts fall back explicitly.

Five prototype recipes are fixed now:

- Primary `pair_skew`: the signed paired readout above, hidden 1152.
- `pair_symmetric`: swap partners without signs, same width/count/execution.
- `pair_diagonal`: signed diagonal tying, same width/count/execution.
- `pair_recompute`: primary mathematics/weights, save z but reconstruct u in
  backward. This tests the execution tradeoff, not a required quality improvement.
- `pair_untied`: independent P/Q, hidden 576, same calibrated response and signed
  pairing before Q. Exactly 2,359,296 weights; an ordinary dense comparator.

Other language comparators: full native/fused-projection SwiGLU 1376, kernel
SwiGLU 1376, GELU 2064; compact kernel SwiGLU 384, GELU 576, and calibrated
kernel SwiGLU 384 whose down initialization gains sqrt(1376/384). All four
compact references, including pair_untied, enter strongest-compact selection.

## Gates registered before measurements

First CPU FP64 independent pair-sum equations, all input/weight gradients,
joint finite differences, counts, calibration, unchanged backbone initialization,
token locality/causality and exact AdamW save/load recovery. Check the Jacobian
formula and its nonsymmetry on a fixed random example; this is algebra, not
language evidence. GPU FP32/BF16 compares every mode with ordinary execution at
small, incomplete-row and [8,256,512] production shapes, including even-feature
and pairing removals. Check all parameter/input gradients and complete models.
Tolerances stay FP32 1e-5/3e-4; BF16 output .004/.02, gradients .0001/.05.
Numerical failure retires this execution before resource/language measurements.

After numerical success, freeze sources and profile eight variants (the five
prototype recipes and three full baselines) on both corpora, three alternating
rounds, 20 warmup/100 measured updates: 48 profiles. Same d512/L4/heads16/V4096,
context256/batch8/BF16/threads4, AdamW .0006, weight decay .1, clipping1, seed101.
Use median throughput and maximum allocated memory. Primary must pass every
comparison: >=1.2x throughput with memory <=1.05x OR >=20% lower allocated
memory with update time <=1.05x. No concurrent GPU work or live source changes.

Only a resource survivor is integrated and checked against the prototype with
shared-Trainer exact recovery. Then all 11 recipes train on both prepared corpora:
seed101, 2000 updates/4,096,000 targets, lr.0006/warmup200/wd.1, identical train
windows and full last-checkpoint validation. Tests unopened. Primary must cost
<=1% NLL against strongest full and gain >=1% against strongest of four compact
controls on both corpora, and beat symmetric and diagonal tying on both.
Finish all controls/removals before retiring a failed recipe; do not promote a
control post hoc. No tuning starts on a failed screen.

Frozen-checkpoint removals: disable partner exchange while keeping signs; remove
the signs while keeping exchange; mask even responses before exchange; replace
the FFN by its mean from 32 train batches, seed2027. Preserve calibration and
report honest no-ops. Compare separately trained symmetric/diagonal controls with
these interventions to distinguish training effects from inference dependence.

Only a survivor receives equal validation-only lr tuning .0003/.0006/.0012 for
primary and all full/compact baselines, followed by paired seeds211/307/401 and
independent seed509 at 8000 updates/warmup800. Repeat mechanism controls with
primary settings. Require both quality margins on both corpora across repeated
seeds and on independent validation/test, opening tests once after freezing.
Sustained resource confirmation uses 100 warmup/1000 measured updates, three
alternating rounds and the same gates. Freeze detailed confirmation before tuning.

## Prior work and provenance

Transpose tying is covered in [the existing prior-art notes](ffn_readout_prior_art.md).
[Saremi's gradient-network analysis](https://arxiv.org/abs/1910.12744) studies
weight tying and symmetric-Jacobian constraints. [Hamiltonian DNNs](https://arxiv.org/abs/2105.13205)
already investigate structured dynamical architectures. Their guarantees do not
transfer to this FFN. Fixed signed permutations and paired matrices are established
tools. This initial search does not establish originality of the exact allocation;
no new primitive is claimed. Any positive mechanism claim requires a closer
implementation comparison. Save every screen's result.md and refresh the leaderboard.
