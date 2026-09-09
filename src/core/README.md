# Shared experiment code

Only the FFN factory changes the architecture. Shared decoder, data, training,
evaluation, optimizer setup and reproducibility code remain here.

| Files | Responsibility |
|---|---|
| `config.py`, `transformer.py` | Six active variants, exact parameter accounting and shared decoder |
| `trainer.py`, `optimization.py` | One training loop, explicit calibration/decay groups, gate recomputation |
| `data.py`, `wikitext_data.py` | Frozen token caches, verified manifests, deterministic sampling and target coverage |
| `benchmark.py`, `diagnostics.py`, `model_audit.py` | Loss evaluation, timing, gradients and scoped projection diagnostics |
| `structured_linear.py` | Reusable grouped linear map and channel shuffle; not a standalone candidate |
| `reproducibility.py`, `frozen_train_worker.py` | Source/config snapshots and qualified immutable cell execution |
| `cli.py`, `report.py`, `analysis.py` | Run commands and cohort-aware historical summaries |
| `serving.py`, `graph_serving.py`, `compile_serving.py`, `fused_serving.py`, `conditioning.py` | Retained post-training execution controls and numerical tools |

Execution tools contain explicit scope checks. Plain packed/fused adapters do not
support learned activations. Dense caches consume extra buffers. These tools do
not establish new training speed or quality findings simply by being available.

Completed one-off experiment drivers live in the [archive](../../research/archive/README.md).
Reproduce their frozen source rather than adding discarded variants back to the
active factory. See [current state](../../research/CURRENT_STATE.md) for allocation.

## Qualified native recomputation

`Transformer.set_recompute_scope("ffn")` or `"block"` opts into non-reentrant
native checkpointing during training with gradients enabled. `"none"` is the
default. Evaluation bypasses this setting, and parameter names/objects are
unchanged. Record the setting separately: ordinary model state dictionaries do
not save it, and a freshly constructed model defaults to none after loading.
The shared CLI training recipes have not changed their execution defaults.

[native_recompute_audit.py](native_recompute_audit.py) records the bounded
five-recipe comparison and synthetic checkpoints. Both scopes pass the measured
fidelity checks, but fail the rational memory-promotion gates against equivalent
full controls. Read [the result](../../research/native_recompute_results.md)
before allocating further training; this tool does not establish a quality gain.

## Allocation diagnosis

[allocation_trace.py](allocation_trace.py) replays default native CUDA allocator
blocks, including requested-versus-allocated padding, splitting, delayed frees and
reuse. [the archived rational memory driver](../../research/archive/h087_retired/src/core/rational_memory_audit.py) preserves four bounded
zero-update workers and exact untraced controls. The completed
[H064 report](../../research/rational_memory_results.md) gives partial source
attribution and exact accounting; it does not qualify a memory repair. Completed
output paths refuse overwrite. Do not rerun this experiment as routine testing.
