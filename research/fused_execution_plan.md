# Four-kernel BlockShuffle inference probe

Frozen before custom-kernel execution. The previous default compiled candidate
has .632 relative throughput versus equally compiled full SwiGLU. Profiles show
48 FFN grouped matrix calls per eight-layer forward versus 24 dense calls, plus
layout/pointwise work. This motivates an execution experiment on the existing
trained weights. No architectural novelty or training improvement is proposed.

## Fixed implementation and arithmetic

Replace each gated FFN with four Triton launches: (1) both square input factors,
(2) both rectangular expansion factors plus rounded SiLU/product, (3) the
rectangular down factor, (4) its square factor. Each factor's output permutation
is written directly to the layout read by the next stage. Keep all original
FP32 factors, casting values to BF16 in the kernels. Store factor outputs and
activation/product in BF16, accumulate matrix products in FP32. No dense product
cache, no new learned weights, and no persistent packed-weight buffer.

This is the same factorized equation in real arithmetic. BF16 roundings remain
explicit at the same factor/gate boundaries; approximate exponentials and
accumulation order require numerical checks. It is inference-only and rejects
grad-enabled/training use. The original training architecture remains intact.

Use a fixed initial tile: 32 token rows, 32 reduction columns, power-of-two
output tile covering a local square block, and 64 output columns for the
rectangular expansion. Four warps and two pipeline stages; no autotuning.
A second token tile of 16 may be examined only if correctness passes and the
first microbenchmark identifies a concrete occupancy/latency reason. Do not
sweep more choices under this plan. Preserve failed compilation or validation.

Report logical factorized matrix FLOPs separately from scalar FMA work implied
by executed padded tile dimensions. At d384/G8, local square width48 pads to64;
rectangular reductions also pad to64. With full token tiles and h2048 this is
983040 padded scalar matrix FLOPs/token/layer versus 700416 logical factorized
and 2359296 full SwiGLU. These are arithmetic counts, not measured hardware
instruction counters. Pointwise work and data movement remain additional.

## Cheap correctness and promotion

First test individual kernels and full FFNs against native PyTorch at small,
non-tile-aligned token counts, both width192 and width384, and on the trained
larger checkpoint. Verify fresh inputs, permutations, parameter identity,
finite outputs, and inference-only guards. Require trained FFN output RMS error
<=.005 times reference RMS and max absolute error <=.05 on three inputs.

Then apply to the original width384/layer8 seed17 800-step checkpoint, with no
post-training condition-floor edit. Evaluate all 32768 fixed validation targets
under batch16/context128 CUDA BF16. Require absolute NLL change <=.001 versus
its own native checkpoint, fresh-input RMS logit error <=.01, and max error
<=.15 on three inputs. A failed eager numerical gate cannot proceed to compiled
serving. These are BF16 tolerances, not bitwise-equivalence claims.

Use a cheap isolated-FFN microbenchmark after numerical checks to determine
whether full-model serving is justified. Apply default fullgraph/static-shape
compilation equally to native BlockShuffle, fused BlockShuffle and full SwiGLU.
Four warmups and six rotating rounds of 30 forwards, all at 2048 tokens. Retain
micro results if fused FFN is at least 10% faster than compiled native
BlockShuffle; a failed micro gate is not a full-model speed measurement.

If earned, perform full-model validation and serving with the identical
compiler/numerical/memory protocol from [the compiler probe](compiler_serving_plan.md).
Include equally compiled full SwiGLU and native BlockShuffle in rotating timing;
use separate fresh workers for memory. The actual compiled candidate NLL is
authoritative. Require <=1% NLL degradation versus compiled full SwiGLU and
>=.8 median throughput ratio. Preserve compile time, failures, cache bytes,
source archives, checkpoint/data hashes and every timing round. Fixed-shape
full-sequence inference is not autoregressive generation or training speed.

## Sources and novelty scope

The implementation follows the documented tiled-dot execution model in the
[Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
and the [dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
[PyTorch's user-defined Triton guide](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
describes compilation integration. Kernel fusion and structured projections
are established techniques; a useful implementation result alone is not a
new FFN primitive or proof of the broader research objective.


## Full-model audit earned

The tile32 probe passed trained eager checks, reproduced NLL2.772705555, and
achieved a 2.597 median isolated-FFN speedup versus compiled native BlockShuffle.
It earns the full audit; no token-tile16 alternative is needed. Before full
execution, fix three compiled cases (full SwiGLU, native BlockShuffle, fused
BlockShuffle), six rotating rounds, four warmups and 15 forwards per case.
Workers retain native eager logits on CPU before replacing FFNs; those fixed
fresh-input references permit numerical comparison while retaining only one
set of model weights on the GPU for memory measurement. CPU reference storage
is disclosed separately and is not counted as GPU allocation. Each worker has
a 300-second deadline; no hidden fallback or retry on numerical rejection.


## Post-result implementation check

The first full-model audit passed all gates, with fused/full median speed1.206
and every round above1.0. Collect one compiled fused profile with ten warmups
and ten measured forwards to verify the four FFN kernels per layer. Profile
instrumentation is not a new serving measurement. After the first audit ended,
the ambiguous kernel argument name O was changed to OUTPUT_WIDTH for lint and
readability; expressions and operations are unchanged. Preserve the first
audit source archive, and repeat the optional kernel tests on the current
source. No tile or arithmetic treatment changes are made.
