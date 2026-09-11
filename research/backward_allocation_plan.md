# H122: identify the chunked backward peak

Previous goal turn: progress. H121 measured only about 5% peak savings from
offloading 48 MiB of checkpoint inputs with chunked classifier loss. Locate
the remaining peak before proposing another change.

Use the two H121 chunked-loss fixtures, native/offloaded storage each, in
their recorded order. Reuse the unchanged model, optimizer-state loading,
data, loss and offload helpers. Two warmup backwards and one allocator-traced
backward per case: **12 backwards, zero updates, 49,152 diagnostic targets**.
No new quality score or timing claim. One GPU process, existing UV runtime.

Record Python allocation stacks with the installed PyTorch memory-history API,
limited to 100,000 entries. Preserve initial/final allocator snapshots and
record boundaries at forward completion and each block recomputation entrance/
exit. Wrappers call the original block method without changing tensor operations.
Capture phase peaks before resetting counters. Save four final gradient maps.

Require: intact H121 source/input hashes; unchanged weights/moments/sampler;
gradient symmetric relative L2 <=1e-5 and max tensor <=1e-4 against H121's
independently audited saved gradients; relative loss error <=1e-6; GPU allocator
zero between cases. Reconstruct allocated bytes by allocation/free-request
events from the initial live set. Reconstructed final allocation and traced
peak must exactly match allocator counters; reject attribution if history
truncates or accounting disagrees. Preserve unknown stack attribution.

Report live allocation groups at the peak, trigger stack, and whether the peak
occurs in classifier backward, block recomputation, or another interval. Stack
sites identify allocations, not necessarily semantic tensor ownership. Compare
with H121 peaks but do not interpret this instrumented trace as throughput.
No candidate promotion, training extension, or broad-goal completion follows
from this diagnostic alone. Preserve failures; do not repeat completed cases.
