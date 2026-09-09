# H086 - Fixed 2:4 sparse projection: native pass, backend qualification incomplete

The compact sparse reference passes all16 native checks. Its first CUTLASS
inference call fails with `sparse_semi_structured_mad_op : CUTLASS not supported`
in the installed PyTorch build. Qualification stops as planned: one attempted
hardware case,11 unexecuted cases, no training and no candidate promotion.
cuSPARSELt support and speed remain unmeasured; this is not a claim that every
sparse backend is unavailable on this hardware.

The fixed exception classifier did not recognize that exact error wording.
The failure is preserved rather than widening the classifier and rerunning the
grid. All28 cases were collected,16 passed, and one failed in10.48 seconds.
The qualification coordinator ran September9,19:54:14.989 to19:54:29.624 UTC.

## What was established

- Both compact FFNs contain350,208 learned values and350,208 bytes of uint8
  support metadata. GELU uses hidden912 and SwiGLU hidden608 at width384.
  Eight FFNs would have70.3125% fewer learned weights than the full controls.
- Native execution materializes dense matrices. Aggregate FP32 dense temporaries
  and temporary int64 indices each total2,801,664 bytes per FFN across projections.
  These are sizes of intermediates, not a measured simultaneous peak.
- Independent auditing verifies15 tensor archives,48 output/gradient pairs and
  36 exact ordinary/checkpoint pairs. CPU FP64/FP32 and CUDA FP32/BF16 cases pass.
- The coordinate isometry and one admissible full-rank matrix witness pass.
  This square sparse witness uses twice the parameters of square BlockShuffle;
  it does not prove equal-budget FFN superiority.

The [plan](compact_sparse_operator_plan.md) and [theory](compact_sparse_operator_theory.md)
scope the claims and attribute established sparse methods. There are zero
optimizer updates, training examples, language targets or valid timing results.
The broad goal remains unmet. This implementation has no earned training budget.

## Evidence and execution failure

The independent analysis initially failed while importing PyTorch with
`MemoryError: bad array new length`, before reading tensor evidence. Its cause
is unknown. One explicit fresh-process execution of the unchanged analysis
succeeds; qualification and training are not repeated. The import event and
successful analysis log remain local with the other raw artifacts.

[Audit receipt](../results/verification/compact_sparse_operator_analysis_v1.json),
[qualification process](../results/compact_sparse_operator_v1/qualification_process.json),
[source/protocol manifest](../results/compact_sparse_operator_v1/protocol.json).
All139 frozen scientific sources,75 plans and prior H085 recovery anchors match
at audit time. Later intentional source retirement must use its recorded source
snapshot for historical reproduction.
