# H154: learn directions inside a narrow reversible coupling core

Previous turn: H153 completed and audited a negative tangent diagnostic (progress).
Its local failure changes the next action: train nonlinear projection directions,
rather than add another shape coefficient to fixed dense mixers.

## Mechanism and limitations

Permute x to (a,b), each d/2 wide. For each of four blocks use
(a,b) -> (a + V GELU(U b + c)/sqrt(4) + e/sqrt(4), b), then undo the
permutation. Hidden width h=d/4. Add a learned full output matrix and bias.
The inverse subtracts the same update evaluated on unchanged b. The triangular
Jacobian has determinant one. If the update Jacobian has operator norm k, the
block singular values lie between sqrt(1+k^2/4)-k/2 and
sqrt(1+k^2/4)+k/2. Across blocks multiply bounds; unconstrained learned U,V
provide no uniform training-time bound or guarantee against vanishing gradients.
The readout need not be invertible. Setting all down weights/biases to zero and
readout W=M represents any linear target M exactly. This removes H151's fixed
linear-tail obstruction, without proving nonlinear performance.

[NICE](https://arxiv.org/abs/1410.8516) and
[RevNets](https://arxiv.org/abs/1707.04585) establish additive coupling and
activation reconstruction. This is an established mechanism at an unexplored
budget in this repository, not a claim of a novel architecture. Unlike H087's
fixed-center scalar-controlled twist, projection directions here are learned.
Unlike H143, weights are not shared between residual uses.

## Frozen bounded screen

d32, four blocks, h8; coupling 2176 trainable parameters (includes readout),
full GELU 8448, narrow GELU h8 2208. Fixed-hidden coupling is an ablation, as is
same-parameter affine coupling. Use the six conventional H152 controls: ReLU,
LeakyReLU, PReLU, GELU, SiLU, SwiGLU. Ten arms total.

Tasks: H152 fixed random GELU teacher and cyclic-product teacher, new data/model
seeds 347/359/373. 4096 train, 2048 validation, 2048 reporting; only train target
mean/RMS normalization. 600 updates, batch128, AdamW(.9,.95), decay0, clip1,
FP32, TF32 off, LR .001/.003 equally searched. Every arm sees identical saved
batch indices per fixture. Native PyTorch autograd for all arms; input gradients
not requested during fitting. 120 fits / 72000 updates maximum. Rotate arm order.
No early stopping, altered recipes, or reporting-based selection.

Choose one LR per task/arm by mean validation loss over all three seeds.
Each task is qualified only if both full GELU and SwiGLU reduce reporting MSE
at least20% vs zero on every seed. Candidate must then be within1% of BOTH full
baselines and budget GELU on every seed and beat fixed-hidden and affine coupling
by at least5% on every seed. Both tasks must qualify and pass to promote. A
positive-control failure makes that task inconclusive, not evidence of universal
candidate failure. Gate does not establish real-dataset performance or novelty.

Before training, FP64 inverse roundtrip, finite-difference gradcheck on small
nonzero weights, exact linear witness, and GPU/CPU FP64 input/parameter gradients
must pass (relative1e-4 for GPU, inverse atol1e-10 at tested inputs). Save protocol
hashes before numerical checks. Abort on failed preflight or nonfinite gradients.
Record all losses/global preclip gradient norms every50 steps, trained states,
optimizer moments/counters, training peak allocated/reserved memory, independent
inference peak and fit wall time. Small-model memory includes resident training
data and is descriptive only: no optimized reconstruction/checkpoint comparison.
If quality fails, do not spend on a custom reconstruction kernel. If quality
passes, next test actual reconstruction vs equally checkpointed controls at d384.

Audit every saved checkpoint using explicit CPU FP64 formulas (different batch
size257), regenerate datasets and fixed buffers, verify optimizer counters and
trainable/buffer counts. Report all-arm mean/median/sample variance and paired
seed ratios, no lucky best run. Preserve any failed attempt and its provenance.
