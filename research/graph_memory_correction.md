# Correction: CUDA graph memory accounting

The first graph audits reused one process but created a fresh warmup stream for
each mode. Deleting model tensors and emptying the allocator cache did not free
that stream's cuBLAS workspace. Consequently, later modes included context
memory retained from earlier modes. Their reported peaks are real process
allocations, but the original single-model memory interpretation is invalid.

A direct [probe](../results/verification/graph_workspace_probe.json) deletes
all models, graphs and outputs, synchronizes and empties the cache. Retained
allocation grows by exactly 8,519,680 bytes per new warmup stream. This matches
the documented default workspace size. PyTorch keeps a workspace for each
handle/stream pair; see [the official documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cublas-workspaces).

The corrected protocol reuses one explicit warmup stream and primes the
ordinary stream before every capture/memory measurement. A regression test
checks that repeated captures leave the same retained allocation. Every mode
is rerun; no estimated workspace subtraction is applied to old numbers.

Original artifacts remain intact:

- results/serving_blockshuffle_graph.json
- results/serving_width_calibrated_graph.json

Corrected artifacts use the suffix _v2.json and record memory_protocol as
shared_warmup_stream_v2. The correction concerns graph-audit memory accounting.
Training peaks and eager audits used the ordinary stream and are unaffected.
The old graph NLL/correctness observations remain valid, and all graph timings
are repeated alongside the corrected memory measurements. Use the corrected
records for current comparisons. They still measure allocated process memory,
including live model/output storage and context workspaces, rather than every
physical driver allocation or autoregressive serving memory.