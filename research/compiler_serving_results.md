# Default compiler serving result

Compilation passed the frozen numerical checks but failed the relative-speed gate. This completed post-training probe uses the width384/layer8 seed17 800-step checkpoints, batch16/context128, CUDA BF16, and the same default fullgraph/static-shape Inductor treatment for both models. Read the [frozen plan](compiler_serving_plan.md) and [environment audit](compiler_backend_audit.md).

| Model | Eager NLL | Compiled NLL | First compile + forward seconds | Compiled peak MiB | Compiled/eager speedup |
|---|---:|---:|---:|---:|---:|
| Full SwiGLU | 2.775852 | 2.775915 | 51.81 | 169.08 | 3.593 |
| BlockShuffle | 2.772706 | 2.772725 | 117.72 | 144.15 | 3.494 |

The paired compiled candidate/full throughput ratio is **0.632**, range 0.575-0.755 over six rotating rounds. Every round is below the .8 floor. Comparing a compiled candidate with an eager dense reference would hide this failure.

Both models have maximum absolute logit difference .0625 versus their own eager checkpoints; maximum RMS differences are .007718 for full SwiGLU and .008653 for BlockShuffle. Fresh-input checks before/after warmup and after timing pass. Compiled NLL uses the actual compiled forward callable on all 32768 fixed targets. This establishes the stated BF16 tolerance, not bitwise equivalence.

Memory comes from a fresh worker for each model after compilation. The parent holds both models for paired timing only. Compilation time is excluded from steady-state speed; no optimizer or training memory is included. Workspace disk caches are separate from learned weights.

## Retained failure and diagnostic profiles

The first reference worker aborted with native status 3221226505 before a result JSON. Its failure and log remain in results/compile_scale384_default_v1. A tiny CUDA compiler smoke test passed, then one unchanged-mode retry completed under results/compile_scale384_default_v2. The native abort's cause remains unknown.

Four separate profiles retain ten eager/compiled forwards for each model. Instrumented operator times are diagnostic and are not serving throughput. Host operator and CUDA kernel rows overlap; summing all rows would double count.

| Compiled model | Matrix operator | Calls per forward | Self CPU ms/forward | Self device ms/forward |
|---|---|---:|---:|---:|
| Full SwiGLU | aten::mm | 41 | 3.770 | 2.242 |
| BlockShuffle | aten::bmm | 48 | 4.596 | 1.183 |
| BlockShuffle | aten::mm | 17 | 1.512 | 0.878 |

The compiled traces contain 17 common matrix calls per forward in both models, plus 24 dense FFN calls for SwiGLU or 48 grouped FFN calls for BlockShuffle. The factorized model also retains layout/pointwise kernels. This motivates testing reduced launch/layout overhead, but these traces neither establish a host-bound fraction nor predict a custom kernel's speed. Autocast can add nested eager dispatch rows; eager operator counts are not physical GEMM counts.

No architecture, training trajectory, training speed or autoregressive-generation improvement follows from this compiler probe. The backend is usable for this fixed workload, but is eliminated as the solution to the relative serving-speed requirement.

Reproduce this report with `uv run python -m src.core.compiler_report`. [Raw results and every timing round](../results/compile_scale384_default_v2/result.json), worker logs and source archives remain in the experiment directory; the [first failure](../results/compile_scale384_default_v1/failure.json) is preserved.
