# Rational activation: isolated execution audit

The frozen execution decision is **PASS**. This tests compiler fusion of the learned rational activation/product on the selected checkpoint. It does not retrain or replace the twelve-trial architecture screen.

Read the [frozen protocol](activation_execution_plan.md), [architecture result](learnable_activation_results.md) and [raw audit](../results/activation_execution_v1/protocol.json).

## Fresh isolated training-memory comparison

| Case | Allocated peak MiB | Reserved peak MiB | Serial training tokens/s |
|---|---:|---:|---:|
| full gelu | 658.23 | 690.00 | 13346 |
| full swiglu | 700.61 | 734.00 | 10763 |
| rational native | 884.35 | 924.00 | 4413 |
| rational compiled | 713.81 | 764.00 | 7122 |

Compiled/native rational allocated-memory ratio is 0.8072. The 10% full-reference limits use these fresh measurements, not historical peaks.

| Gate | Decision |
|---|---|
| memory within ten percent full gelu | PASS |
| memory within ten percent full swiglu | PASS |

## Numerical and gradient fidelity

| Check | Measured |
|---|---:|
| logits max abs | 0.0 |
| logits relative l2 | 0.0 |
| all gradients relative l2 | 0.000784290506311299 |
| activation gradients relative l2 | 0.00016879276423865594 |
| finite | True |

Original full-validation NLL is 5.898002606; compiled-product NLL is 5.898002606, a +0.0000000% change over 322,688 targets. Original checkpoint hashes and learned parameter objects are preserved.

## What was measured

Each worker loads its selected seed-17, 200-step checkpoint and AdamW state. Correctness checks and original-checkpoint validation precede profiling. Model and optimizer state are restored; all workers use the same thirty sampled minibatches. Ten warmup updates precede twenty measured updates. The updates exercise real BF16 forward/backward, clipping and AdamW allocation, but are transient: no resulting weights are saved or scored as a new model.

Only the rational activation and its product are compiled. Projections, attention and optimizer remain native; gate checkpointing remains enabled. Static fullgraph Inductor uses low-precision cast emulation and no CUDA graphs. Compilation and CPU reference buffers are excluded from steady-state GPU peaks. PyTorch allocated memory does not include all driver allocation. The native full controls and serial timings do not support an equal-compiler full-model speed claim.

A passing memory/derivative audit is still not a compiled training-trajectory replication. The original architecture screen's memory failure remains a historical result. A bounded compiled training repeat, independent seeds, longer training and corpus transfer are required before stronger claims. No new activation primitive or full-network stability theorem is established.


Follow-up: the [full 200-step compiled training repeat](activation_training_repeat_results.md) fails quality despite passing memory. The isolated PASS above does not establish training-trajectory equivalence.
