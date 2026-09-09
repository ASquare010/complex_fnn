# H053: Overcomplete representation and numerical qualification

**The frozen qualification passes. Language-model superiority remains untested.**
The new [standalone module and mathematical notes](archive/retired/src/multihead_ffn/overcomplete.md)
use the same 350,208 weights/layer as the existing compressed models. The
[plan](overcomplete_qualification_plan.md) allowed no language training or scoring.

| Required check | Measured outcome |
|---|---|
| Explicit constrained-factor representation of Q | PASS; 384 coordinates plus a global sum, 770 active signed units |
| FP64 witness forward, 31 inputs | Relative L2 1.718e-16 |
| Three Hessians, eight directions each | Maximum relative error 3.740e-14 |
| Independent dense-factor/head-loop derivatives | All input and parameter derivatives pass; input gradcheck passes |
| Input/output isometric initialization | Maximum errors 1.333e-15 / 6.939e-17 |
| Three-seed Gaussian output RMS / full SwiGLU | 1.01356 / 1.01821 / 1.01643 |
| Native BF16 forward / maximum gradient error | 0.007889 / 0.008171 |
| Ten fixed synthetic AdamW updates | All finite; every parameter receives a nonzero initial gradient |
| Full regression suite | 238 tests pass |

This proves that the specific quadratic excluded by all single square-mixer
additive headwise layers is representable by these overcomplete structured
factors. It does not prove class containment, novelty, fast learning, general
quality or a whole-network gradient guarantee. The isometry and covariance
claims concern initialization. Private nonlinearities remain ordinary SwiGLU.

The **memory limitation is material**: isolated eager forward/backward allocates
69.366 MiB versus full SwiGLU's 57.626 MiB, **20.37% higher**. These measurements
exclude optimizer states and do not replace the complete Transformer's gate.
A separately frozen integration/memory check is required before any LM screen.
No router, new activation shape, gate recomputation or custom kernel was added.

The [CPU result](../results/overcomplete_qualification_v1/cpu.json),
[CUDA result](../results/overcomplete_qualification_v1/cuda.json),
[protocol](../results/overcomplete_qualification_v1/protocol.json), witness weights
and source archive retain the complete evidence. All 31 registered models,
trainer, optimizer and data code remain unchanged; there are still 165 retained
LM/profile runs. This module is not a registered Transformer variant.

The [238-test record](../results/verification/overcomplete_tests_v1.json) retains
the tested source. A post-qualification local lambda-to-function style cleanup
has an identical expression and otherwise identical AST, recorded in the
[style equivalence check](../results/overcomplete_qualification_v1/style_equivalence.json).
It changes no model, witness, calibration calculation or recorded GPU result.
Original qualified source is archived. All qualification phases completed on
first attempt; no numerical or process failure occurred.
