# H141: maintained opt-in training-memory helper

**Integration verification passed.** The qualified classifier and final-block
input offload are available through explicit APIs in
[src/core/training_memory.py](../src/core/training_memory.py).
[Usage and limitations](training_memory_usage.md) include a complete invocation.
No existing maintained source or training default changed.

## Maintained versus frozen GPU replay

| Corpus | Source seed | Frozen peak MiB | Maintained peak MiB | Raw gradient relative L2 | Worst tensor gradient relative L2 | Parameter relative L2 | Moment relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext2 | 101 | 330.430 | 330.430 | 5.358e-08 | 1.472e-07 | 1.027e-09 | 3.060e-08 |
| wikitext2 | 113 | 330.430 | 330.430 | 5.153e-08 | 1.338e-07 | 9.240e-10 | 2.980e-08 |
| wikitext2 | 127 | 330.430 | 330.430 | 3.203e-09 | 2.482e-08 | 5.488e-11 | 1.996e-09 |
| tinystories | 101 | 324.251 | 324.251 | 8.277e-08 | 1.735e-07 | 1.487e-09 | 4.613e-08 |
| tinystories | 113 | 324.251 | 324.251 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| tinystories | 127 | 324.251 | 324.251 | 9.144e-08 | 1.752e-07 | 1.510e-09 | 4.978e-08 |

All six H140 source model/Adam/sampler states were replayed once under each path,
with alternating arm order: 12 complete updates/backwards and 49,152 targets.
The two paths start from identical states and batches. Correct suffix wrapping,
parameter object identity, state paths, finite states, Adam step 801 and complete
wrapper restoration are checked. Thirteen zero allocated/reserved boundaries
verify GPU cleanup. Saved raw/clipped gradients, parameters and moments were
independently compared on CPU with NumPy FP64. Every prospective numerical and
memory limit passed. Per-pair bitwise results remain in the raw result and are
not a gate under ordinary nondeterministic attention.

The classifier's forward arithmetic has identical AST to H127's qualified
native-buffer forward; backward arithmetic has identical AST to H128's layout
repair. Public argument validation happens before that unchanged core. The CUDA
adapter uses the same native save_on_cpu hook around whole-block checkpointing.
The maintained context additionally preflights the selected blocks, rejects
unsupported overrides/nesting and bypasses transfer on CPU. It restores methods
on exceptions. It does not change recomputation settings implicitly.

## Test coverage and environment recovery

Sixteen new tests passed: native loss/input/weight derivatives, column-major and
strided layouts, masked labels, all-ignored semantics, finite-difference checking,
caller tensor ownership, tied model gradients, zero/invalid suffix sizes, atomic
preflight and restoration after nesting/errors.

The full suite initially had **126 passes and six failures**. All six were in
existing optional Triton inference tests and failed compiler discovery before
kernel execution. The UV interpreter's sysconfig directory differs from the
repository package directory. A bounded rerun set process-local CC to the already
installed Triton TinyCC executable: **all six passed**. The original failure log,
compiler path/hash, recovery plan and focused rerun remain recorded. Thus all
132 collected tests have passing coverage across those runs; this was not a
single green full-suite invocation. No compiler was installed and no old test
or source was changed to suppress a failure.

## Scope and next experiment

These checks establish integration equivalence at the previously qualified scale.
They do not provide new validation scores or meaningful throughput measurements:
one-step peaks include gradient/state diagnostics and cannot replace whole-job
training benchmarks. The earlier H140 short-segment savings remain supporting
prior evidence; H138's long-run failure remains unchanged. Parameters are unchanged.
This is an experimental opt-in helper, not a new activation, SOTA result or proof
of the broad research goal.

Next test a different workload scale with the maintained API, ordinary controls,
complete memory/compute accounting and explicit quality checks. Keep the separate
parameter-efficient FFN research objective open; do not mistake memory-implementation
integration for novel neuron geometry or proof of sustained deployment throughput.

[Prospective plan](training_memory_integration_plan.md),
[GPU replay](../results/training_memory_integration_v1/result.json),
[independent CPU comparison](../results/training_memory_integration_v1/independent_check.json),
[evidence receipt](../results/training_memory_integration_v1/receipt.json).
