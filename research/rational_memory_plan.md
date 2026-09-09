# H064: locate the rational activation memory peak

Frozen before instrumentation or measurement. H063 is completed progress: native
recomputation preserves its measured computations but both boundaries fail the
rational memory limit against equally checkpointed full controls. This is a
bounded diagnosis, not another checkpoint search or a language-training allocation.

## Question and mechanism

Where does the remaining allocation occur under whole-block recomputation? Compare
native rational and plain BlockShuffle at the same shape and execution policy.
Distinguish persistent model/optimizer/input allocations, forward or recomputed
pointwise tensors, matrix/attention work, and allocations only attributable to
native autograd. Do not force unattributed backward buffers into the rational bin.

The homogeneous rational formula evaluates full-width FP32 work arrays. At
batch 16, context 128 and hidden width 2048, each FP32 array contains 4,194,304
values (16 MiB), versus 8 MiB in BF16. This predicts a potentially material
transient cost but does not establish which arrays coexist at the observed peak.
No tensor dtype, function, parameter, reduction, optimizer or batch size changes.

Use the installed PyTorch allocator history, with a 50,000-event cap, allocation
stack contexts including C++ where available, and baseline/forward-end/backward-end
snapshots. [Official memory documentation](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html)
describes allocator events and snapshot blocks. This measures PyTorch-managed CUDA
allocation, not all driver memory. Tracing is not used to claim runtime speed.

## Four fixed workers and budget

Pin H063's input metadata and selected native seed-17 checkpoints for rational
and plain BlockShuffle, with the loaded optimizer moments. Keep whole-block
recomputation and each recipe's native inner product checkpoint policy. All
model/source/configuration files from H063 remain byte-identical; add only shared
diagnostic code/tests. Validate the replay parser on synthetic allocator events
and run the full active suite before dispatch.

Run these fresh CUDA workers in order: plain untraced, plain traced, rational
untraced, rational traced. Each uses d384/L8/V4096/B16/T128/native BF16, four CPU
threads, and H063's first synthetic batch from generator seed 60017. One warmup
forward/backward precedes one measured forward/backward. Clear gradients between
them, but make ZERO optimizer updates. Preserve weights and optimizer moments.
Total: eight forward/backward evaluations, 16,384 synthetic target exposures,
zero corpus training/validation/test scoring and zero new LM cells.

For each worker record original weights/moments, model configuration, source,
environment, token/RNG hashes, measured loss/logits/every gradient, and allocator
baseline/forward/end/peak bytes. Hash gradients only after measurement. Save raw
allocator snapshots in traced workers. Traces do not retain live GPU tensors.
Keep all original artifacts and failures; no automatic worker retries.

## Accounting and decision

Replay allocated memory from the baseline live blocks, adding alloc events and
subtracting free_requested once. free_completed changes reuse eligibility, not
allocated tensor bytes. Preserve delayed-free distinctions and reject unknown
frees, duplicate live addresses, truncated rings, or inconsistent terminal state.
Compare replayed start/end/peak bytes with CUDA allocator counters and reconcile
each phase boundary with its snapshot. Test pointer reuse and delayed frees in
the parser before relying on its attribution.

Require exact traced/untraced loss, logits and all gradients, unchanged weights,
optimizer moments and RNG, and equal measured allocation peaks within each recipe.
If instrumentation changes these, report it and do not attribute an uninstrumented
peak using the altered trace. This is a diagnostic failure, not model failure.

For valid traces, report live allocations at each recipe's peak by stack origin,
source line and size, their phase and largest allocation sites. Cross-model peaks
may occur at different times; do not equate a bucket's size with the causal memory
saving from removing it. Distinguish allocated from reserved bytes and fully
explained from unresolved origins. The one-batch, zero-update state differs from
H063's maximum over twenty updates; keep those results separate.

A supported pointwise attribution may motivate one separately specified execution
hypothesis with a concrete memory target and exact numerical gates. It does not
itself earn a repair implementation, compiler retry, reduced batch, offload, new
activation or longer training. If the gap remains unattributed, close this diagnosis
and report the limitation instead of expanding the tracing budget automatically.
No new architecture, novelty, quality, or general gradient claim follows.
