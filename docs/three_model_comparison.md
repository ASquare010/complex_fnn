# Three-model comparison

## Historical experiment — hierarchy retired 2026-10-12

Hierarchy completed 20,000 updates: validation CE 4.200900109, token accuracy
24.643939%, versus token-memory CE 3.013115671 and 41.297348% at 20,000 updates.
Both scores use 211,200 development target tokens. Hierarchy improved but did
not close the quality gap under this budget; this is not proof that all
hierarchical architectures fail. Its checkpoints and active source were removed
at the user's request. Logs/results and source archives remain in dump.
Archived source: dump/retired-hierarchy-source-20261012.zip. Deletion manifest:
dump/hierarchy-cleanup-20261012.json. The hierarchical commands below are
historical and no longer runnable. Raw baseline and token-memory remain active.

Existing token-memory run completed 20,000 updates: validation CE3.01311567,
token accuracy41.29735%,211,200 validation tokens (100 fixed documents).
Final checkpoint reload replay passed in its saved result.json. Existing sources,
weights and results are preserved; the two new models start from scratch.

| Model | Trainable parameters | Previous context | Current token prediction |
|---|---:|---|---|
| token_memory (completed) | 23,209,216 | Frozen encoder vectors | Large Branch Sigmoid transformer, up to512 token positions |
| raw_transformer | 23,077,376 (-0.57%) | Up to2048 raw tokens | Standard causal transformer with GELU; up to512 positions |
| hierarchical_memory | 23,244,032 (+0.15%) | Frozen encoder vectors and already completed64-token blocks | Large global transformer produces conditioning state; small causal decoder generates64 tokens |

Baseline: width512,8layers,8heads,GELU FFN1536,tied embedding/output,RMSNorm,RoPE.
Raw history + active block capacity2560. Same source documents as the memory arms.
The dataset documents have2112 tokens, so the last block has64 targets; capacity
2560 does not imply training on2560 real tokens in one example. The original run
used activation checkpointing for concurrent FP32 jobs. On2026-10-11 the user
requested disabling it for speed: the baseline now retains activations instead
of recomputing them. Raw per-worker memory guard is10GiB (hierarchy stays5GiB).
Do not assume the old concurrent micro8 memory measurements apply to this mode.
Reviewed source-hash migration permits existing checkpoints to resume with the
same weights, optimizer, targets and schedule; only execution/memory behavior
changes. Existing live workers require a saved stop and restart to use it.

Hierarchy: global width512,7layers,8heads,Branch Sigmoid. Decoder width256,
3layers,4heads,GELU FFN1216 (chosen to match the total trainable budget). A learned
start state handles empty history. The decoder receives both a projected global
conditioning state and cross-attention access to causally available global states.
No target vector matching, MSE, frozen reconstruction decoder or parallel64-token
output. Both trainable networks receive gradients from next-token CE.
The frozen encoder has9,426,432 extra parameters in each memory arm; baseline has
no encoder. Total system parameters and compute therefore are not identical.

## Causality and matched data

Reuse dump/data/token-memory directly; no download, repacking or corpus encoding.
Each2112-token document gives the same five target segments512/512/512/512/64.
All three use batch32, seed47, the same shuffled segment order, all target tokens,
20,000 updates, FP32, AdamW decay0.01, gradient clip1, LR3e-4 to3e-5 with warmup200,
same 4096-token tokenizer and the same100 development documents every1000 updates.
Loss is summed token CE divided by all non-padding tokens in the optimizer batch.
Changing microbatch only changes accumulation grouping and floating-point rounding.

Within a hierarchical512-token target segment, eight64-token predictions are
evaluated together during training. Their encoder inputs are independent64-token
chunks. Causal global attention and prefix-restricted decoder cross-attention
ensure each prediction can see only completed earlier chunks, never its own
target encoding. Decoder inputs are BOS+shifted true tokens within each64-token
block. Every target token is scored exactly once. At inference, generated tokens
are fed back into the decoder; each completed64-token block is encoded before the
next global prediction. Global states are reused during the64 token decisions.

This changes the representation boundary from512 to64 for the hierarchy. It is
an architectural comparison, not a pure test of one-vector conditioning alone.
GELU vs Branch Sigmoid and the hierarchy are additional differences. Equal steps
and target counts do not mean equal FLOPs or wall-clock time. Simultaneous jobs
compete for the GPU, so their runtime is not a clean isolated speed benchmark.
One seed and reused development data do not establish a general benchmark win.

## Run in two PowerShell terminals

The two models have separate worker locks; duplicate workers of the same arm are
refused. Fresh names refuse overwriting existing output. CPU14/15 are available
on this PC; each command keeps the existing single-CPU pinning mitigation.

Baseline:
```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\src\models\raw_transformer\training.py --name raw-transformer-2560-20k --steps 20000 --batch-size 32 --microbatch 8 --cpu 14
```

Hierarchical:
```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\src\models\hierarchical_memory\training.py --name hierarchical-memory-64-20k --steps 20000 --batch-size 32 --microbatch 8 --cpu 15
```

Ctrl+C once requests a saved stop after the optimizer update; wait for PAUSED and
the shell prompt. Resume using the same model's training.py, a new --name, and
--resume pointing to the old run directory. --steps is a total, not additional.
Microbatch may change; effective batch, architecture, sources and data must match.
Optimizer, RNG, sample position and LR schedule restore. Progress25/save250/
validation1000, retained1000-step milestones, source archive, checkpoint hashes
and final reload replay are retained. No research training was launched by Codex.

Compare losses at common saved steps:
```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\src\models\comparison\compare.py
```

## Verification

CPU tests cover target identity, future-token and future-encoder-vector leakage,
padding invariance, gradients into the global model and conditioning projection,
frozen encoder gradients, raw activation recomputation, separate arm locks,
wrong-arm resume rejection, exact tiny synthetic resume, matched schedules and
global-state reuse during generation. Synthetic lifecycle tests are two updates
on tiny fake data, not research training.

GPU forward/backward checks used micro8 and mixed long real segments (raw padded
length2560): baseline peak reserved1.73GiB, hierarchy1.47GiB. Neither check used
an optimizer or changed weights. Adam moments add about0.17GiB per model, plus
runtime/allocator overhead. These independent checks support trying both at8,
but are not a guarantee of all concurrent training peaks. Per-worker guard5GiB.
Receipt: dump/comparison-memory-check.json. Raw needs more computation per update
because it attends to the uncompressed history and recomputes saved activations.
