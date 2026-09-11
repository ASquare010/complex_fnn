# H112 execution recovery: one dependency import, separate fresh models

The first H112 dispatch is terminally interrupted before qualification or
training. Its qualification child exits3221225477 during Python source import
within Torch. Neither the CPU-preload completion message nor any scientific
record was written. The original frozen source, protocol, logs, exception and
empty corpus output roots remain preserved. No model update is repeated.

H111 ran72 evaluations and the independent24-state audit successfully in
long-lived processes. The bounded recovery therefore reduces repeated imports:
launch one process, load the same dependencies once, run the frozen eight-case
qualification, then create and fully discard each original H112 model/optimizer/
data instance sequentially. Keep the original corpus, seeds, policy order,
evaluator, updates, optimizer, dtype, instrumentation and quality/resource gates.

The explicit difference from the first H112 plan is process isolation. Models
are independently initialized, but trials share the Python/CUDA process. Before
each trial and after complete garbage collection, require zero allocated CUDA
tensor bytes; record that boundary and fail if any model storage remains. Cached
allocator memory is cleared at boundaries. The CUDA context and external library
state can persist, so this cannot establish fresh-process timing/driver-memory
equivalence. The already-frozen20-update timing warmup and balanced policy order
remain. The reported resource is allocated tensor memory, with process/driver
overhead still excluded. No numerical gate or budget is relaxed.

Use a new `results/whole_job_memory_continuation_v1` root. Freeze all original
failed evidence and recovery adapters before execution. Reuse the unchanged
original H110 scientific worker and frozen H112 evaluator/data adapter. Preserve
the added step200 checkpoint and independent native rescoring requirements.
One start attempt only; stop on the first runtime, source, state, data or
nonfinite failure. No silent rerun, automatic retry or replacement of original
logs. Root cause of the native failures remains unknown.

All twelve trials are still prospective and never-trained: eight hundred
updates each, two corpora, three seeds. If completion is interrupted, report only
verified completed trials and retained partial histories. An import/workaround
success is not a scientific success. H110 stays stopped, and the broad research
goal and architectural parameter target remain open.
