# Optional compiler backend: compatibility investigation

The current eager/graph results show a concrete runtime issue: the factorized
FFN uses six grouped GEMMs plus channel layout transformations per block.
The [Monarch paper](https://arxiv.org/abs/2204.00595) already motivates grouped
matrix products for hardware use; factorization itself is established prior art.
Compiler fusion is an execution experiment, not a new mathematical primitive.

The local environment has torch2.14.0+cu132 on Windows/CPython3.12, with no Triton
module installed. The existing uv.lock resolves Triton3.8.0 for Linux only.
A read-only [PyPI metadata snapshot](../results/environment/triton_windows_catalog.json)
finds triton-windows3.8.0.post28 with a CPython3.12 Windows wheel, published
2026-08-29. The snapshot includes the wheel SHA256. This matches the upstream
minor-version requirement; actual runtime compatibility is still untested.
Primary context: [Windows fork](https://github.com/triton-lang/triton-windows),
[release cadence](https://github.com/triton-lang/triton-windows/blob/readme/RELEASE.md),
and [package metadata](https://pypi.org/pypi/triton-windows/json).

No dependency or active training environment was changed during this inspection.
Complete the frozen scale training and baseline serving first. If runtime is
still inadequate, an optional pinned UV compiler extra can support a bounded
fullgraph/static-shape compiler probe. Apply the same compiler mode to full
SwiGLU and factorized BlockShuffle, measure full fixed validation and fresh
inputs, preserve failures and compile time, and use rotating paired timing.
Do not count compilation as steady-state inference, silently fall back to eager,
assume bitwise equality, or compare compiled candidates against eager controls.
No kernel implementation or compiler performance improvement is established yet.

## Installation and first runtime evidence

After the scale training and ordinary serving audits completed, UV installed
only triton-windows3.8.0.post28 as an optional Windows extra. Torch remained
2.14.0+cu132 and reported Triton3.8.0. A native full-worker abort was retained;
a subsequent tiny standalone compiler test passed. See the [frozen compiler
probe and failure history](compiler_serving_plan.md). The metadata snapshot
above describes the earlier read-only inspection, before installation.


## Completed runtime outcome

The identical-mode retry completed both isolated workers and six paired timing
rounds. Both models pass the frozen BF16 checks. Compiled candidate/full speed
is .632, failing the .8 gate, despite general compilation speedups for both
models. Four operator profiles are complete. This demonstrates usable runtime
compatibility for the stated workload, not universal compatibility or a serving
solution. See the [completed report](compiler_serving_results.md). The earlier
inspection and first failure above remain historical evidence.
