# Within-update batch layout analysis

18 October2026. Existing jobs, model sources, checkpoints, data and caches are
unchanged. This is an unadopted efficiency candidate for later training.

The CPU analysis checks all48,762 original updates over two original epochs.
Each update retains its same32 examples (including the last partial batch),
labels and update boundary; only the order of microbatches of8 changes. All
1,560,372 example exposures and361,357,104 LM labels are preserved.

| Layout | Padded positions ratio | Dense attention pair ratio | Local reader scalar-slot ratio |
|---|---:|---:|---:|
| Existing |1.000|1.000|1.000|
| Length then memory |.8251|.7302|.7780|
| Memory then length |.9362|.8903|.6253|
| Raw length times memory |.8850|.8154|.7596|

For each microbatch, n is maximum raw length, m is maximum code count and b is
batch size. Proxies sum b(n+m), b(n+m)^2 and b*n*m*64*9 respectively. These omit
vocabulary work, optimizer, I/O and other costs and are NOT wall-time speedups.

A separate actual two-layer STOP-model CPU fixture uses heterogeneous raw and
memory lengths, the existing LM/query/source objective and unchanged global
denominators. Every parameter's clipped gradient agrees across original,
length-first and memory-first layouts, maximum absolute difference2.98e-8.
An eight-update deterministic layout trajectory matches an exact serialized
Adam resume after update4, including optimizer tensors. Twelve fixture optimizer
calls were executed (8 uninterrupted,4 resumed); production updates remain0.
The original fixture initialization stays unchanged. Executed source is saved
as probe_cpu.txt alongside the receipts.

Adoption still requires full-size GPU gradient/optimizer/capacity/timing checks
and explicit layout provenance in the new trainer/recipe. Within-update floating
reduction order changes. Existing original-corpus orders and schedules must not
be overwritten, nor may layouts cross effective-update boundaries.

Reports: dump/native256-batch-layout-analysis-20261018/report.json and
cpu-gradient-resume-report.json; fixture-update-accounting.json clarifies counts.
