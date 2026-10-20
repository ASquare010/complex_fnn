# Wide512 full-budget training: verified early result

18 October2026. Goal ACTIVE/UNMET. The fresh compressed64 arm is still training;
the fresh matched raw1 arm is queued sequentially behind it. Native256, original
models/runs, encoder, corpus and caches remain unchanged. No completed pilot was
extended, no source migration occurred and no reserved student TEST was scored.

Reader budget:23,209,216 trainable parameters; frozen encoder separate. One global
memory position represents64 old tokens; raw context64. This experiment retains
the disclosed exact literal-ID payload in code mantissa bits. It is not evidence
of a lossless learned encoder or64-fold byte/total-compute compression. Two local
query readers inspect that payload without adding global attention/KV positions.

## Full original development result at2,000 updates

All7,831 development segments/1,859,388 assistant labels. Historical references
are the saved step2,000 scores on the same original development labels; they are
not substitutes for the new matched raw1 control.

| Reader | Development CE | Token accuracy |
| --- | ---: | ---: |
| New wide512 compressed64 |2.509291|50.570188%|
| Historical original raw512 plus older compressed memory |2.840777|45.199657%|
| Historical plain64 |3.191832|39.054840%|

New-minus-original accuracy:+5.370531 percentage points; new-minus-plain64:
+11.515348 points. All17 source groups have higher accuracy and lower CE than
both historical references. This is higher same-step language accuracy for the
new architecture/training bundle, not a causal attribution to any single change,
a final two-epoch comparison, or proof of factual/reasoning recovery. In particular,
teacher-forced memory-source accuracy must not be called free-answer retrieval.

## Independent verification

The actual step2,000 checkpoint was copied immutably while the original GPU worker
continued. The frozen training auditor verified all2,000 original example orders,
14,841,890 assistant-label exposures, learning rates, cursors, physical microbatch
orders and auxiliary denominators/objectives. All optimizer counters are2,000;
parameter/moment shapes, FP32 moments, finite weights/moments and2,770 protected
file hashes passed. Auxiliary exposures:3,181 query segments/6,362 terms;
2,522 source segments/5,044 terms, including2,522 name and2,522 value terms.

Every document/source development total was reaggregated. Public CPU forward
replayed one complete document from each of the17 sources and matched GPU correct
counts/label counts, with mean CE difference below1e-4. Thirteen source documents
fit one segment; the remaining four required multiple segments. These replays
check the saved checkpoint/evaluation path and do not add training updates.

Authoritative artifacts under `dump/token-memory-64-wide512-unified-v1`:

- `development-step2000-audit.json`: checkpoint/training audit, all-document/source
  reaggregation,13 source CPU replays and historical reference comparisons.
- `development-step2000-additional-replay.json`: the four remaining source
  documents, proving combined17-source replay coverage.
- `historical-step2000-references.json`: original saved scores and source hashes.
- `checkpoint-monitoring/`: immutable checkpoint copies and captured log prefixes.
- `compressed64/dev-documents-2000.json` and `validation.jsonl`: original new-run
  development totals.

## Full original development result at4,000 updates

| Reader | Development CE | Token accuracy |
| --- | ---: | ---: |
| New wide512 compressed64 |2.290836|53.482221%|
| Historical original raw512 plus older compressed memory |2.469274|50.791174%|
| Historical plain64 |2.903172|43.041635%|

New-minus-original accuracy:+2.691047 percentage points; new-minus-plain64:
+10.440586 points. Versus its own step2,000 result, new accuracy improves2.912033
points and CE improves0.218456. This remains an interim teacher-forced language
result, not a full-budget/matched-raw1 or free-factual result.

Against the historical original512 at this same step,14/17 sources improve in
both accuracy and CE. Three trail it: everyday(-2.226525 accuracy points),
stories(-0.865434), reasoning_tools(-1.072706). All17 improve over historical
plain64. The new model's memory-source teacher accuracy declines from97.072013%
at2,000 to96.008089% at4,000, and its memory CE worsens from0.166144 to0.269053.
Thus the global language improvement does not establish uniformly improving
memory behavior. Teacher-forced memory tokens are not free-answer fact recovery.

`development-step4000-audit.json` records independent reaggregation of every
document and all17 sources; exact4,000 baseline sampler/target/LR comparisons;
29,660,180 original assistant-label exposures; actual checkpoint format/recipe/
source/cursor, all Adam counters4,000 and finite saved model tensors. This is a
metadata/reaggregation audit, not a new public CPU replay or full moment/protected
file audit; the stronger saved-checkpoint/public-forward path was verified at2,000.
The running job and recipe were unchanged.

## Full original development result at6,000 updates

| Reader | Development CE | Token accuracy |
| --- | ---: | ---: |
| New wide512 compressed64 |2.183718|55.052792%|
| Historical original raw512 plus older compressed memory |2.335474|52.798017%|
| Historical plain64 |2.767915|44.965333%|

New-minus-original accuracy:+2.254774 percentage points; new-minus-plain64:
+10.087459 points. All17 sources improve accuracy and CE versus the new model's
step4,000 result, with global accuracy+1.570571 points and CE-0.107118.
Memory-source teacher accuracy recovers to96.649960%/CE0.212211, still below its
step2,000 accuracy. These are teacher-forced language measurements, not factual
free-answer or reasoning success.

Against same-step original512,14/17 sources improve both metrics. Everyday still
trails by1.581155 accuracy points, stories by0.780750 and reasoning_tools by1.907032.
All17 improve over historical plain64. The historical references include different
reader architectures/objectives and are not the new matched raw1 comparison.

`development-step6000-audit.json` verifies all document/source reaggregations,
44,448,180 original assistant-label exposures and all6,000 original sampler orders,
targets, learning rates and cursors. Actual checkpoint format/recipe/source/cursor,
all Adam counters6,000, finite saved model tensors and checkpoint hash were checked.
As at4,000, this is a metadata/reaggregation audit; public CPU replay was performed
at2,000 and the source remains unchanged. No optimizer updates were used for audit.

## Remaining requirements

Both arms must finish48,762 original updates/two epochs/361,357,104 assistant-label
exposures EACH, with full training/optimizer/source audits. Then require the
matched language, natural free-answer, changed-value/query, missing/blocked/
shuffled, four-way binding, historical and reasoning evaluations and independent
replay. A failed raw1 control cannot establish successful compressed retrieval
by parity. Reserved novel-literal student TEST stays closed until BOTH full
exposure gates pass. Any efficiency claim must distinguish global position/KV
compression from local dictionary work/raw-window overhead and include encoder,
prefill and rollover for end-to-end timing.

The current single GPU job is already running. Its strict weights-only loader
needs the standard `torch.torch_version.TorchVersion` class allowlisted because
PyTorch serialized its version metadata as that string subclass. The external
compatibility bootstrap is recorded in `loader-compatibility.json`; no model,
trainer, source snapshot or checkpoint was rewritten to fix metadata loading.
