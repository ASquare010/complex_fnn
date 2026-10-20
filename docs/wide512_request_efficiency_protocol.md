# Final wide512 request efficiency protocol

This is a separate, read-only benchmark. It does not modify the182 frozen model,
training or evaluation sources or any checkpoint. GPU execution is prohibited
until BOTH fresh arms pass48,762-update/361,357,104-label exposure gates and the
single-worker lock is available. No reserved student TEST inputs are used.

Prepare12 immutable workloads from original development documents: old prefixes
of128/512/2048 tokens, raw phases0/63, batch1/4, and128 original continuation token
advances per row. Select the first sufficiently long original documents, with no
decoder-success or model-output filter. Hash source/code/tests/protocol, input
files, cases and the original training manifest. Reverify exact token slices
before execution. Timing inputs are forced equal work, including possible EOS;
they are not factual predictions, QA labels or additional training examples.

One warmup and six measured rounds per case; alternate arm order each round.
Only the active reader is resident on GPU; the frozen encoder is resident only
for compressed64. Checkpoint/model transfers occur outside request timing.
FP32, TF32 disabled, same70% quota. Time CPU wall clock with CUDA synchronization,
including input transfer, original frozen encoder, literal packing, cache
allocation, global memory prefill, raw suffix prefill, all128 token advances and
all rollover encoding. Exclude tokenization, case disk I/O, checkpoint loading
and moving model weights to GPU. Report phase times and total request medians,
all samples, allocated/reserved memory peaks and resident weight memory.
Token selection and text decoding are excluded: this is complete fixed-token
model-request work, not a whole serving-stack or variable-EOS chat latency claim.
Capture memory peaks before the separate payload audit allocations.

Use the existing evaluator's max-new-token cache reserve policy for both arms.
After timing, audit both read layers' complete payload IDs and raw suffix against
the full original request sequence. Check exact memory/raw counts and unchanged
cache field schema. Report actual total KV/cache bytes including raw64, validity
and token metadata and reserved capacity.64:1 concerns old-token global positions;
total cache or request compute need not improve64-fold. Local64-word dictionary
work and encoder overhead remain real costs. The added literal payload is not a
claim of learned lossless or byte compression.

CPU tests use the actual two-layer model and public forward, a deterministic fake
encoder ONLY for test mechanics, and both1/64-word modes. Cover ordinary/near-full
raw phases, one/two rollovers, exact payload recovery, encoder call counts, empty
old history and public/cached equality. Real GPU execution always uses the original
frozen encoder and audited final trained weights. No optimizer is instantiated.
Require five passed CPU tests and a source-bound receipt, no initialized CUDA
context, unchanged test weights/RNG and no gradients. Test that an unfinished
training gate rejects before constructing models/encoder or initializing GPU.
Test the independent cache-byte/sample arithmetic verifier against actual cache
allocations and corrupt sample/encoder/latency reports. After the GPU benchmark,
CPU `verify` independently recomputes all sample timing sums, medians, ratios,
encoder/rollover counts and exact cache bytes including reserve/raw metadata.

Prepare command (CPU, safe while training):

```powershell
& .\.venv-cuda64\Scripts\python.exe .\tools\benchmark_wide512_requests.py prepare
```

After BOTH full exposure audits and the running worker's terminal completion:

```powershell
& .\.venv-cuda64\Scripts\python.exe .\tools\benchmark_wide512_requests.py benchmark
& .\.venv-cuda64\Scripts\python.exe .\tools\benchmark_wide512_requests.py verify
```

The benchmark itself narrowly allows PyTorch's version string metadata for
weights-only loading. Results are separate from factual/language quality and
cannot select or extend a checkpoint. Evaluate meaningful uncompressed quality
as well; two equally failed readers do not establish successful compression.
