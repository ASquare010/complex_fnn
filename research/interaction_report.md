# Cross-group interactions: proofs and three-seed results

The mechanism has a proven representational advantage over an additive grouped
FFN on specific cross-group functions. The implemented candidates fail the LM
promotion gate. These are separate findings, both retained.

## What is proved
At the FFN input, the original grouped sandwich is additive across input groups;
its cross-group mixed second derivatives are zero. Fixed coupling can exactly
represent a*(a+.5*d). An explicit SiLU construction and the additive population
normalized-MSE lower bound 5/21 are in the
[fixed-coupling derivation](cross_group_hypothesis.md).

The adaptive projected-value variant contains every original grouped FFN at
zero coupling and can exactly represent a*d. Its value-mixing matrix has
singular values bounded between .2 and 1.75 for every allowed coefficient.
See [proof and limitations](adaptive_coupling_plan.md). These are elementary
structural results, not novelty certification or whole-network gradient guarantees.
CPU finite-difference tests and an exact BF16 initialization/backward check pass.

## What training found
The same train/test inputs are verified by identical saved hashes. Each cell
is mean +/- sample SD over seeds 17,29,43, with 1000 FP32 AdamW steps. Targets
are normalized by training-set statistics; the population lower bound need not
exactly equal finite-holdout error. Parentheses list all trainable toy parameters.

| Model | Constructed polynomial | Additive control | Pure product |
|---|---:|---:|---:|
| swiglu (193) | 4.32e-05 +/- 1.6e-05 | 0.000216 +/- 8.6e-05 | 3.82e-05 +/- 1.4e-05 |
| swiglu_narrow (49) | 0.000214 +/- 0.00019 | 0.00387 +/- 0.0057 | 0.000177 +/- 0.00016 |
| structured_swiglu (49) | 0.226 +/- 0.00097 | 0.00012 +/- 8.6e-05 | 0.994 +/- 0.0011 |
| structured_swiglu_coupled (49) | 2.08e-05 +/- 1.4e-05 | 0.224 +/- 0.00094 | 0.746 +/- 0.0014 |
| structured_swiglu_adaptive (65) | 0.000221 +/- 0.00026 | 0.000179 +/- 6.6e-05 | 0.0157 +/- 0.027 |
| swiglu_budget (61) | 0.000346 +/- 0.00027 | 0.00479 +/- 0.0012 | 0.000201 +/- 8.9e-05 |

Fixed mixing realizes the constructed task but fails the additive control and
pure product. Adaptive mixing repairs the additive failure. On the pure product,
its seed-43 error is .0470 versus .0000174 and .0000507 in the other seeds.
The explicit exact representation therefore does not imply reliable finite-step
optimization. Narrow dense controls are substantially more reliable on that task.
The 65-parameter adaptive variant also has four more parameters than the closest
smaller 61-parameter budget control, and 16 more than the original grouped model.

The function tasks were selected to diagnose the mathematical mechanism. They
are not a representative sample of complex functions. A win on the constructed
target cannot establish general approximation, sample-efficiency or LM superiority.
[Standalone figure](../results/plots/interactions.png) shows every seed and min/max.
[Machine-readable summary](../results/interaction_summary.json) includes learned
couplings; raw checkpoints, optimizer states, histories, data and source archives
remain in results/interactions_v1 and results/interactions_adaptive_v1.

## LM decision
At 200 steps, seed 17: original grouped SwiGLU 4.64820 NLL, fixed coupling
4.63555, adaptive coupling 4.63076. Narrow GELU is 4.24112, narrow SwiGLU 4.41907.
Adaptive reduction is 74.8264% of FFN weights, but forward throughput falls to
60597 tokens/s and allocated peak memory reaches 272.06 MiB. Neither candidate
earns longer LM training. Learned mixing coefficients and response slices are
saved in the adaptive LM run, with every coefficient and predetermined channels.

## Consequence for the next round
Stop adding gates to this weak single-factor grouped control. A stronger next
control is two-factor BlockShuffle/Monarch or Block Tensor-Train, where feature
mixing happens before the nonlinearity. Its initialization and optimization
must be derived and compared fairly; the current grouped results cannot reject
that family. Sharing remains a separate useful control, still short of the gold
quality target. No branch has established the requested research breakthrough.
