# Fresh lexical64 versus conventional1:1: training in progress

Latest18October2026: compressed1,000 evaluation and both independent verifiers
are complete. Language44.136780%, but positives21/285 and corrected-prefix/full
notebook all-four0/64. Raw completed/audited1,000; its full evaluation is running,
with language stage33.631227%/CE3.694165 and factual/results audits pending.
A separate untrained nine-slot/33-word pooled-key reader passes full CPU/GPU and
independent gates; its corrected fresh original-corpus v2 run is prepared and
all-data audit passed. It adds about18% batch8 body time/703MiB temporary peak with unchanged
KV. See [verified intermediate results](native256_lexical_1000_results.md) and
[pooled-key costs](native256_lexical_span_results.md).
The dated preparation details below precede these completed checks.

18 October2026. Goal ACTIVE/UNMET. No broad quality or adoption claim.

The user confirmed23M reader parameters, with the encoder separate. Both new
readers have exactly23,209,216 trainable weights and fresh seed47 initialization.
The compressed arm keeps one global attention/KV position per64 old tokens and
raw64; the conventional reference gives every old token its own position. The
compressed variant adds literal token IDs in low FP32 bits, rather than proving
lossless recovery from the original semantic encoder. Local reading adds compute.

The new active run is `dump/native256-lexical-full-unified-v2`. Prepared v1 stays
zero-update and frozen. All original/native runs, sources and caches are retained.
The recipe uses exactly the original unified-v4 segments, assistant labels,
orders, AdamW/LR schedule and2 original passes:48,762 updates and361,357,104
assistant labels per arm. There are no extra QA training examples or losses.

Verified before continuation:

- Reference and optimized full-size GPU/math/cache gates, independent CPU audits.
- All780,186 train/7,831 dev segments and4,987,062 source-code blocks audited;
  all48,762 original batch orders/target counts/LRs match.
- Exact fresh initial states and empty Adam states checked independently.
- Selected microbatches8 compressed/2 raw pass full-size gradient/capacity gates.
  These zero-update body samples exclude Adam, data IO, evaluation and saving.
- Equivalent updater passes exact8-step Adam/serialized-resume comparison and
  explicit NaN/Inf rejection before any optimizer update.
- Both actual first updates paused at1 using the same32 examples/7,035 labels.
  Both independent first-update audits passed, including all Adam steps and
  all2,097 protected earlier files.

The compressed arm completed and paused at exactly1,000 on the original
48,762-step schedule. Its independent training audit passed:all Adam states1,000,
7,416,257 labels, exact original orders/targets/LRs/cursor and protected earlier
files unchanged. Its last-batch CE3.20155 is not held-out quality evidence.
Raw completed and paused at250. Its intermediate audit verifies all Adam states
at250,1,831,425 labels and every batch matching the audited compressed first250.
Compressed full evaluation and its independent result audits are complete;
raw resumed from250 and is continuing to1,000 for the matched comparison.
These are intermediate cursors, not resets or extra epochs.
Both retain the full original schedule. Native256 remains untouched, and only
one GPU worker runs at a time. The full factual evaluation and independent
result audits for raw remain pending.

The preliminary full-development stage at1,000 updates is complete:

| Reader at1,000 original updates | Token accuracy | CE |
|---|---:|---:|
| Old plain64 |35.105476%|3.511001|
| Old native256 |33.170807%|3.733725|
| New lexical64 |44.136780%|3.065599|

New lexical64 correctly predicts820,674 of1,859,388 assistant labels, an observed
+9.031305 percentage-point gain at the same update count. Saved document totals
reaggregate to the printed score. This compares the complete changed architecture
and representation, rather than isolating one component. The old1,000 weights
were not retained, so no paired document/seed claim is made for this comparison.
The matched raw1,000 checkpoint and its evaluation remain pending. Compressed
free factual outputs and both independent audits are complete:21/285 positive
full replies,1/64 changed-value pairs,0/64 changed-query pairs and0/64 all-four
notebook units. Teacher-forced accuracy is not free fact recovery,
and1,000 updates is not two completed original epochs. The receipt is
`compressed64/evaluation-1000/early-development-comparison.json` under the run.

The independent supplemental verifier now checks source/edit/query/targets,
strict EOS paired scoring, all-four units and source-selected public-forward CPU
generation independently of cached decoding. Its two CPU controls pass; actual
reconstruction of all540 natural evaluation cases passes. It also verified the
trained outputs:60 independently selected public CPU replays match cached GPU
generation.32 additional notebook public CPU replays match, with all outputs and
blocked invariance audited. Full-budget matched comparison remains outstanding.

A supplemental evaluation-only notebook control is now prepared and fully
audited:64 four-way units,512 full-reply/completion cases and128 blocked controls,
selected from67 structurally eligible original-dev dialogues. Locations swap
with identical source wordbags, statement/value coordinates and raw query tails;
both queried people must succeed against both memories. Conditional prefixes
stop at actual BPE boundaries before value content. Three CPU controls pass,
including rejection of unequal value coordinates and failure of constant,
last-record and truncated answers. No new training data or model change.
See [prospective binding check](native256_lexical_binding_check_protocol.md).

Additional input-only audits confirmed every conditional prefix is exactly the
queried name plus fixed relation and is identical across swapped memories.
Visibility varies:the requested name remains fully raw in168/256 full-reply
cases and248/256 completions.248/256 full answers and all256 completions fit
before rollover. Entire queries are raw in52/256 full cases and44/256 completions.
Compressed all-four remains zero in the prospectively defined raw-name/no-rollover
strata(40 full/62 completion units) and whole-query-raw/no-rollover strata(11 each).
The matched raw comparison will report the same strata. Cases
were not filtered or selected using model predictions. Failures with compressed
query text cannot uniquely diagnose fact binding.

The same-step comparison tool also has passing paired-bootstrap/all-four CPU
controls. It requires both independent result audits, checks actual sampler/LR/
label/cursor logs and identical evaluation prompts, and reports fixed-checkpoint
document/unit intervals. It retains the stronger old61.013% mixed512 reference
separately; parity with a weak new raw reference alone is insufficient. No actual
comparison result has been produced yet.

After the live raw worker stops at1,000, audit and evaluate that checkpoint:

```powershell
& .\.venv-cuda64\Scripts\python.exe .\tools\run_native256_lexical_full.py verify --audit training --arm raw1
& .\.venv-cuda64\Scripts\python.exe .\tools\evaluate_native256_lexical_v2.py --arm raw1 --step 1000
```

Do not automatically extend compressed1,000 or duplicate the live raw worker.
The new pooled-key gate is prepared but must wait for the GPU to become free;
it tests unchanged parameter/KV budgets with explicit mean-embedding CPU
equations and isolated body-memory measurements before any new training.
These commands do not resume or change native256.
See the [full protocol](native256_lexical_full_protocol.md),
[equivalent updater](native256_lexical_update_protocol.md), and
[verified efficiency evidence and limits](native256_lexical_gpu_results.md).
