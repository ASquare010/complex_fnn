# Original test inputs independently verified

18 October 2026. All input-preparation workers completed successfully. This is
an input/source audit, with zero student predictions and zero optimizer updates.
It makes no claim about model quality or codec losslessness.

The original test split contains 7,646 segments, 1,757,421 assistant labels and
48,459 completed 64-token prefix blocks. The separate frozen encoder has
9,426,432 parameters. Preparation stored its FP32 semantic codes in a new cache
without changing the original corpus, original caches, native256 or any student
checkpoint. Literal-ID packing remains a later RAM-only operation.

The independent CPU audit derived every document prefix directly from original
token/mask/segment bytes and re-encoded every block using groups of 127 rather
than preparation's 256. All codes matched exactly: maximum absolute error 0.
All segment label totals, source/input/cache hashes and encoder weights/buffers
were unchanged. Session 66169 finished with exit code 0.

The independent GPU input check covered 1,165 blocks selected across sources,
boundaries and the full block range. FP32 with TF32 disabled gave maximum
CPU-cache versus GPU absolute error 2.288819e-5, within the specified tolerance.
Encoder weights/buffers remained exact. This also completed successfully.

Authoritative receipts are in
`dump/native256-original-test-inputs-20261018`:

- `manifest.json` binds preparation sources, inputs and cache.
- `independent-cpu-verification.json` records the all-block CPU audit.
- `gpu-input-verification.json` records GPU agreement and its helper source.

Student scoring on this original test split remains reserved for actual
step 48,762 checkpoints with verified two-original-epoch exposure to
361,357,104 assistant labels. That final evaluator and same-format novel-literal
test are still outstanding. No pilot test score has been used to select the next
architecture or training objective.
