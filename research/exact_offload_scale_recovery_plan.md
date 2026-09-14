# H161 bounded cleanup recovery

The original driver terminated with exit1 after case00 recorded all30 updates
and saved initial gradients/final weights. Its zero-GPU boundary assertion failed
before result.json was written. Preserve that folder and original sources.
Resource summaries from that attempt are unavailable; do not reconstruct them
from model weights. The diagnosis script separately tests cuBLAS workspace
retention with one matrix product and zero training updates.

Repair only the post-run cleanup: clear autocast and cuBLAS workspace caches
before empty_cache, as the repository's established clear_boundary already does.
Do not clear them within measured training or validation. Apply the same cleanup
to native scoring audit. No recipe, model, data, source helper or gate changes.
Run the six-case schedule in a separate recovery folder; the first attempt's
30 updates and one initial backward remain excluded but counted as spent work.
Accepted budget180 updates/186 backwards; total210 updates/217 backwards.
The separate diagnostic adds one matrix product, zero backwards.

The original plan mentioned process RSS, but the implementation did not collect
it. Explicitly report RSS unavailable; pinned-host allocator counters are not
process RAM measurements. This instrumentation omission remains a limitation
and is not silently converted into measured evidence. All original numerical,
GPU memory, pinned allocation, timing and short-NLL gates stay fixed. Freeze
this adapter and the preserved failed evidence before starting recovery.
