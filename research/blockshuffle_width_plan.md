# H014: use the remaining structured parameter allowance

Frozen after h832 reached3.115322 NLL and the product-decay ablation reached
3.118254. The decay correction has no positive signal; retain parameter decay
for the next width test. This is a default-preserving choice, not proof that
the two decay methods differ statistically.

Test blockshuffle_swiglu h1024,G8 with factor-aware LR and parameter decay,
800 steps, seed17. At d192,L4, its unique FFN count is350208, exactly matching
narrow SwiGLU h152 and shared SwiGLU h608. It retains70.3125% FFN reduction,
with1728192 model weights. FFN matrix FLOPs retain the same70.3125% reduction
versus full SwiGLU h512. Matrix work is18.75% higher than h832; report this.

This consumes the 70% acceptance allowance without relaxing it. If the NLL
improves by at least.01 over h832, lock h1024 for seeds29,43; otherwise lock
h832. Then repeat the corresponding parameter-matched narrow control and
full SwiGLU at800 steps on the same seeds. Include every seed and the same
fixed validation set. No further width increase past the cap in this round.

Passing the single-seed1% gate would remain preliminary. The learned primitive
is established BlockShuffle plus an explicit optimizer recipe; novelty, equal
hyperparameter-search budgets, broader data and larger models remain separate
requirements. Inference caching must report added storage and its own loss.


## Recorded outcome
Hidden width 1,024 improved seed-17 NLL from 3.115322 to 3.098982, exceeding
the predeclared 0.01 improvement threshold. Native gate recomputation resolved
the training-memory gate before three-seed replication. The locked recipe
reaches mean NLL 3.108014; the [final comparison](blockshuffle_results.md)
includes every seed and the remaining quality and runtime failures.
