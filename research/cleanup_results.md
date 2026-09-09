# Active-model cleanup

**The requested cleanup is complete; the scientific target remains unmet.** The
repository keeps the strongest available compressed reference, one conditional
learned-activation candidate and the dense controls needed for fair comparisons.
No new quality or novelty claim is made.

| Active inventory | Before | After |
|---|---:|---:|
| Model folders | 11 | 3 |
| Registered variants | 32 | 6 |
| Core Python files | 85 | 21 |
| Test files | 50 | 16 |
| JSON recipe files | 47 | 9 |

Active model folders are `dense_ffn`, `blockshuffle_ffn` and
`rational_blockshuffle_ffn`. Dense variants are full/narrow GELU and SwiGLU.
The [archive guide](archive/README.md) records why each removed branch was retired.
Nine old folders and 138 obsolete individual files moved out of active directories.
The shared `GroupedLinear` and shuffle functions were preserved in the core;
retained parameter names, initialization, equations and optimizer behavior match.

## Verification

- The completed pre-cleanup audit passed 248 tests; its historical result remains
  [recorded](../results/verification/residual_geometry_final_v1.json).
- A verified 526-file source ZIP preserves the complete pre-cleanup environment.
  Every archived file was hash-checked before moving its active counterpart.
- Twelve exact before/after cases cover all six retained variants on CPU FP32 and
  CUDA BF16, including nonzero learned shape coefficients, initial weights,
  logits, loss, gradients, diagnostics, optimizer groups and two clipped AdamW
  updates. [Before](../results/verification/cleanup_before_signatures_v1.json) and
  [after](../results/verification/cleanup_after_signatures_v1.json) are identical.
- The [complete active suite](../results/verification/cleanup_tests_v1.json) passes
  **89 tests in 23.59 seconds**. It includes independent derivatives, actual
  counts, memorization, GPU kernels, optimizer controls, data/evaluation guards,
  and synthetic training/checkpoint reloads. All assertions and GPU checks stayed
  enabled; active source was unchanged during the run.
- Plain packed/fused inference adapters now reject the rational subclass before
  replacing any layer. Previously those adapters could omit its activation.
  Regression tests exercise both direct construction and partial-conversion risks.
- CLI/training validation rejects retired variants and failed activation compiler
  backends clearly. Historical reporting can still read retired run records.

The verification uses fixed synthetic inputs for short updates. It performs no
new corpus training or corpus validation scoring. All 168 existing language-model/
profile runs remain, and the 2,178 protected result/data files are unchanged in
size and modification time. The final audit checks source/config hashes, archive
integrity, protected metadata, active imports and documentation links.

Two preliminary probe launch errors occurred before mutation: a PowerShell quoting
error while writing the helper and a direct-script import-path error. The helper
was then written literally and invoked as a module. Both CPU/GPU probe runs and
the full test suite completed on their first executable attempts. No training
trial was retried, hidden or overwritten.

## Next allocation

The [additive block/low-rank proposal](additive_block_lowrank_proposal.md) remains
one unimplemented hypothesis with an analytical matched budget. It must earn
numerical, memory and balanced quality qualification. The discarded overcomplete
and affine branches receive no automatic continuation. Rational activation remains
conditional on a faithful memory improvement; previous compiler attempts are closed.
