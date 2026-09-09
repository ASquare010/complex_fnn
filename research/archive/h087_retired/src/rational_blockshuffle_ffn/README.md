# Rational BlockShuffle SwiGLU

**Secondary candidate: promising one-seed quality, memory gate failed.** This is
plain BlockShuffle with a small learned residual added to its SiLU gate. Only the
native implementation remains active.

`phi(z) = SiLU(z) + 0.25*(a0 + a1*z + a2*z^2 + a3*z^3)/(1 + beta*z^2)`

Coefficients are bounded by tanh and shared within each channel group. Each layer
learns its own shape. At eight groups and eight layers, the activation adds just
320 weights: 2,801,984 FFN weights and 9,099,968 total, a 70.3091% FFN reduction.

The selected one-seed 200-step WikiText trial reaches NLL 5.898003, 1.216% below
its selected plain control and 0.244% below calibrated narrow. Its training peak
is **883.17 MiB**, failing memory. These are short-screen results, not a replicated
advantage. Resetting the final learned shape changes NLL by only 0.0056%:
[all screen outcomes](../../research/learnable_activation_results.md).

Use [the retained recipe](../../configs/wikitext2_blockshuffle_rational_screen.json)
with `--learning-rate 0.0012` for the selected screen rate. The file retains its
original 0.0006 template default. Checkpoint gate recomputation is supported.
The rejected compiler backends and affine/shifted activation branches are
[archived](../../research/archive/README.md). Plain packed/fused SwiGLU adapters
reject this model because they do not evaluate its learned activation.

Read [the mathematical notes](model.md) and the
[activation domain guide](../../research/learnable_activation_domain.md).

The [native recomputation audit](../../research/native_recompute_results.md)
preserves the measured initial gradients, 20-update losses/norms and final
weights/moments exactly. Whole-block recomputation saves 46.17% of this synthetic
training allocation, but remains about 40% above equally checkpointed full
controls and costs 25.61% more time per update. The memory limitation remains;
no full language repeat is earned by that execution change.

The [allocation diagnosis](../../research/rational_memory_results.md) observes
128 MiB in eight live allocations from FP32 activation evaluation during native
recomputation, at a 460.332 MiB synthetic peak. Tracing preserves every measured
gradient and allocator total. Other origins remain unresolved on Windows; the
128 MiB is an observed live cost, not demonstrated removable memory. The native
candidate remains secondary and no longer training is earned.

The subsequent [fixed staged repair](../../research/staged_resource_results.md)
preserves full-model measured numerical behavior but saves only 32 MiB (6.95%)
at 8.85% more update time. Its 428.333 MiB synthetic peak still fails both dense
memory caps. This execution prototype remains outside active source, and no
language repeat or alternate partition is earned.
