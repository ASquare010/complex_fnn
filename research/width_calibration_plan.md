# H021: calibrate the narrow dense reference - frozen ablation

Both paired-feature candidates failed their frozen LM gates. Even the duplicate
feature control beat them despite its redundant output coefficients. That
observation makes initialization and optimizer geometry a necessary control
before attributing a compressed model's gain to nonlinear expressivity.

The original protocol keeps the down-projection standard deviation and learning
rate fixed while reducing its fan-in from 512 to 152. The first choice reduces
its expected initial output variance, independently of function class.

## Exact initialization calculation

For zero-mean independent output weights D_ij with variance sigma_D^2 and h
exchangeable hidden features phi_j of second moment q, independence gives
E[(sum_j D_ij phi_j)^2] = h * sigma_D^2 * q. Cross-feature correlations do not
contribute because independent D coefficients have zero cross expectation.
The up/gate projections keep the same distribution when only hidden width
changes. Multiplying D's initial standard deviation by sqrt(h_ref/h) therefore
matches the full FFN's ensemble expected output variance at initialization.
This does not match individual realizations or constrain trained activations.

Use h_ref=512 for SwiGLU at d=192 (and 4d for ungated dense FFNs). A separate
learning-rate hypothesis multiplies only the down-projection rate by h_ref/h.
For h=152, the gain is sqrt(64/19) and the LR multiplier is 64/19. This is a
local fan-in heuristic inspired by width-aware parameterization, not a claimed
complete muP derivation or an optimality theorem. See
[Tensor Programs V](https://arxiv.org/abs/2203.03466) and the
[structured-layer study](https://arxiv.org/abs/2406.06248).

Whenever LR is multiplied by kappa, set that matrix's AdamW decay to q/kappa.
Then its pure decay factor remains exactly 1-eta*q, matching the original
reference. All other parameters retain their original rate, decay and initial
values. This prevents a hidden regularization change. No new learned weights,
forward operations, checkpoints or caches are introduced.

## Three fixed screens

At hidden width 152, seed 17 and 200 steps, compare the original narrow SwiGLU
with initialization-only calibration, LR-only calibration, and both. Use the
same common data, token budget, clipping, base optimizer and BF16 configuration.
The full SwiGLU reference is an exact fixed point of both rules (ratio 1).
Verify this and unchanged non-FFN tensors before training.

These are baseline ablations, not a new architectural claim. Promote at most
one recipe, the lowest NLL, to 800 steps only if it improves over the original
narrow control by at least 1% relative NLL and remains finite within the memory
gate. Compare the longer run with the existing 800-step original narrow, full
and structured references, disclosing the changed recipe. Improvement over an
untuned baseline alone cannot establish global superiority. A competitive
result must earn three seeds and equally applied execution/tuning comparisons.

## 800-step outcome and locked replication
The selected initialization-plus-LR recipe reaches seed-17 NLL 3.126104 versus
3.149475 for the original narrow reference. It remains worse than BlockShuffle
3.098956 and full SwiGLU 3.065045. The improvement is smaller than at 200 steps,
so the early relative gain cannot be extrapolated to a longer budget.

This is a useful stronger conventional control at the same 350,208 FFN weights,
with no new inference operations. Freeze two additional 800-step runs, seeds
29 and 43, without changing the selected recipe. Compare all three paired
seeds against the existing full, original narrow and BlockShuffle runs. Also
measure the seed-17 checkpoint with equally applied eager and CUDA graph
serving, using the established rotating timing procedure and fixed validation.
These replications evaluate the conventional reference; they do not promote
it as a novel primitive or declare the original target achieved.
