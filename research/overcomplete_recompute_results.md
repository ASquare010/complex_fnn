# H055: Native gate recomputation clears the measured memory gate

**The recomputed candidate passes the frozen parameter, numerical and memory
qualification and earns a separately frozen language-quality screen.** It has
not yet demonstrated language-model quality or converged superiority.

| Complete Transformer execution | Allocated training peak MiB | Change vs eager candidate |
|---|---:|---:|
| Full SwiGLU, retained H054 | 706.733 | - |
| Full GELU, retained H054 | 660.108 | - |
| Overcomplete headwise, eager H054 | 730.389 | - |
| Overcomplete headwise, native recomputation | **680.014** | **-6.897%** |

The reduction is **50.375 MiB**. Native recomputation is 3.781% below full
SwiGLU and 3.016% above full GELU, passing the <=10% allocation allowance against
both. Counts stay at **2,801,664 FFN weights** (70.3125% below full) and
**9,099,648 total weights**. This is training allocation in the specified
pipeline, not inference throughput, generation speed or a larger-model result.

## Controlled change

The [frozen plan](overcomplete_recompute_plan.md) changes only
`recompute_gate=true` and `gate_recompute_method=native` from the H054 candidate.
It reuses the established RecomputedSwiGLU function already used by BlockShuffle.
The ordinary graph retains the gate input, value input and SiLU activation;
the custom backward retains the two inputs and recomputes SiLU. Projections,
activation formula, parameters, initialization, uniform LR multipliers, .1
parameter decay, global LR .0012, batch16/context128 and AdamW settings stay equal.
There is no new router, activation family or inference implementation.

A shared `configure_gate_recomputation` helper applies the same setup in the
real trainer and qualification worker, preserving the prior BlockShuffle and
learnable-activation rules. The new module defaults to eager execution. All
**32 default model signatures** remain exact after the change. H054's worker
is reused with explicit frozen configuration/output arguments and a pretraining
arithmetic check; its exact original implementation remains archived.

## Numerical and trajectory evidence

FP64 eager/native forward values match exactly, every input/parameter derivative
passes an independent comparison, and input gradcheck passes. On the full native
BF16 Transformer, initial eager/native loss matches within 1e-7 and **every
parameter gradient matches exactly** in this check (maximum relative L2 error 0).
BF16/FP32 initial logit relative error remains .008384.

One new cell repeats H054's exact first CUDA WikiText training batch for 20
constant-rate updates: **40,960 training-token exposures**, 2,048 unique targets,
zero additional sampler batches in the arithmetic check and **zero validation
or test targets scored**. Every weight, gradient, Adam moment and layer diagnostic
is finite. The checkpoint round trip restores logits exactly without recalibration.

A separate [trajectory comparison](../results/overcomplete_recompute_v1/trajectory_equivalence.json)
finds all **9,099,648 final weight elements** and the entire 20-step history
bitwise equal to H054's eager trajectory. This is an observed equality for these
runs, not a general promise of bitwise equality at arbitrary precision/duration.
The final repeated-batch loss 5.413896 and 65% clipping fraction are diagnostic;
they are not held-out quality results or selection criteria.

## Retained evidence and next boundary

All **243 tests pass**, with [exact tested source](../results/verification/overcomplete_recompute_tests_v1.json).
The [raw decision](../results/overcomplete_recompute_v1/result.json),
[preflight](../results/overcomplete_recompute_v1/preflight.json),
[protocol](../results/overcomplete_recompute_v1/protocol.json), source archive,
checkpoint, every gradient comparison and all logs remain available. All phases
complete on first attempt with no numerical or process failure. H054's eager
memory failure remains recorded.

This earns only a separately frozen balanced short quality screen. That screen
must check actual memory under its own training configuration, complete held-out
NLL, calibrated narrow and both full controls; independent seeds and longer
training remain later requirements. No novelty or breakthrough follows from
reusing an existing activation-recomputation method.
