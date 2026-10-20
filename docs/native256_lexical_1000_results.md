# Lexical64 at1,000 updates: language gain, binding still unresolved

18October2026. Training, full development/factual evaluation, independent
aggregate/public-source verification and controlled notebook verification have
completed for compressed64 at1,000. **Goal ACTIVE/UNMET. This is an intermediate
checkpoint on the48,762-update/two-original-epoch schedule, not final parity.**

Run:`dump/native256-lexical-full-unified-v2`. Reader23,209,216 trainable parameters,
fresh seed47, encoder separate/frozen. All1,000 original orders/LRs/targets/cursor
and7,416,257 assistant labels verified; all Adam states1,000. All2,097 protected
earlier files unchanged.64 old tokens retain one global attention/KV position.
Literal token IDs are deliberately added in existing FP32 low bits; this does
not prove the original semantic encoder is lossless. No extra QA training loss,
data or original epochs. Current native256 and completed experiments stay intact.

## Language at the same original update count

| Reader | Full-dev accuracy | CE |
|---|---:|---:|
| Old plain64 |35.105476%|3.511001|
| Old native256 |33.170807%|3.733725|
| New lexical64 |44.136780%|3.065599|

New lexical64 has820,674 correct of1,859,388 labels:+9.031305pp over plain64 and
+10.965974pp over native256. Native's actual first1,000 orders/labels/LRs/cursors
were independently compared and match. These compare changed architectures and
representations; they do not isolate one component. Old1,000 weights were not
retained, so no paired-document/seed claim is made against those logs. Matched
raw1,000 and full-budget comparison remain pending. Teacher-forced accuracy is
not free factual recovery. The stronger old61.013% mixed512 reference remains a
separate full-budget reference, not a conventional memory-free baseline.

## Free answers and counterfactuals

| Original corpus check | Strict complete answer plus EOS |
|---|---:|
| Positive originals |21/285|
| Missing originals |0/63|
| Changed value |3/64|
| Changed queried person |3/64|
| Both original/changed-value answers correct |1/64 pairs|
| Both original/changed-query answers correct |0/64 pairs|
| Fact vectors blocked |0/64|

All285 positive originals ended in EOS. Exploratory whole-source-value analysis
finds the requested name correct in95/285, requested value correct in60/285 and
both correct in21/285.68 use the right name with another record's whole value;
39 use another name with the requested value;42 give another complete source
record;39 combine another name with a different source record's value.20 formed
replies contain a value not present as a whole source record;56 have another
format. These descriptions are not independent causal evidence or successes
from partial words. The model accesses source words but fails robust selection
and binding to the question.

Historical notebook exact:8/103 positives,3/32 changed,1/25 missing; compressed
colors0/8. All seven retained legacy access groups have0/512 exact/all-four.
Those access/value-only formats were not current training examples; their zeros
alone cannot diagnose literal-value capacity. The limited numeric probe is0/53,
with3/53 EOS; it does not establish general reasoning parity.

## Controlled notebook binding

Both names are queried against both swapped memories. Locations swap with the
same wordbag, statement/value positions, length and raw tail.64 units give256
full replies and256 conditional completions. The completion prefix consists
only of the queried name/fixed relation, at actual BPE boundaries before values.
No model-output or decoder-success filtering.

| Format | Exact plus EOS | Both swapped memories correct | All four correct |
|---|---:|---:|---:|
| Full reply |22/256|1/128 pairs|0/64 units|
| Token-boundary completion |78/256|1/128 pairs|0/64 units|

Blocked first16 units:0/64 exact in each format. All64 matched blocked swapped
pairs have identical generated tokens. Independent public CPU replay matches
32 enabled/blocked cases; all saved text/EOS/metrics reaggregate. All-four remains
zero in the prospectively defined raw-name/no-rollover strata(40 full,62
completion units), and whole-query-raw/no-rollover strata(11 each). Therefore
compressed question text or rollover cannot explain all the binding failures.

The frozen main evaluator also retains a different completion check:0/477.
That prefix ends in standalone whitespace; inspected outputs often immediately
produce a period. The new controlled test uses token-boundary prefixes. These
sets/conditions differ, so their counts are not a clean before/after attribution
to tokenization. Robust binding fails even with the corrected boundary.

## Concrete local-key limitation and next model

An input-only audit finds the current nine-token key excludes EVERY owner-name
token at the first value token distinguishing two records:570 correlated
comparisons across285 original positive cases. A representative key decodes to
` keeps a notebook in the black`. Including the complete owner name requires
11..20 tokens. Main semantic memory/query hidden states still exist; this is a
local feature limitation, not whole-model impossibility or proof of a sole cause.

A separate untrained `token_memory_64_lexical_span` package keeps the same nine
learned key slots but pools offsets0,1,2,3,4,5..8,9..12,13..20,21..32. It exposes
the complete owner name in these inputs without extra global positions or KV
arrays. Parameters, fresh tensors, RNG and buffers are identical to lexical64;
all23,209,216 reader weights remain trainable. Six CPU tests pass independent
explicit-embedding-mean/all-parameter/input gradients, cached/public/actual64
rollover, bytes/array inventory, future isolation, empty/blocked behavior,
microbatches and serialized two-epoch8-step Adam resume.

The sixth check runs a separate full-model implementation that constructs and
projects actual mean embedding vectors, comparing all parameter gradients to
the scalar prefix kernel. New zero-update GPU/independent-audit entrypoints are
prepared; they include conventional raw1 and original lexical64 loaded latency,
actual rollover and isolated batch4/8 body peaks, each starting with zero live
GPU allocation. They have not run while raw training owns the GPU.

A further full21-layer CPU check on genuine original cached source words passed
with zero loss difference and maximum all144-parameter-tensor gradient error
3.576279e-6 against the explicit mean-embedding reference. Fresh weights/RNG
match exactly and no update occurred. Its report is under
`dump/native256-lexical-span-cpu-preflight-20261018`; this is equation evidence,
not trained quality or GPU efficiency evidence.

The change adds scalar prefix sums/range gathers. GPU numerical/cache/body gates
and independent CPU verification have now passed. Batch8 body adds about18% time
and703MiB temporary peak, with no extra KV storage. No trained quality result
exists. See [verified cost and limits](native256_lexical_span_results.md).
See [prospective protocol](native256_lexical_span_protocol.md). Raw1 is unaffected
and completed/audited1,000 on its original schedule. Its full quality evaluation
is running; the language stage reports33.631227%/CE3.694165 and remaining factual
and independent result audits are pending. Any later adoption still
requires full two-epoch matched factual/language/reasoning evidence, same-format
novel values and final original-test evaluation. Conditional completion alone
cannot replace full question answering.

## Evidence

Under`compressed64/evaluation-1000`:report.json,early-development-comparison.json,
independent-aggregate-verification.json,independent-public-verification.json,
notebook-binding/independent-verification.json,name-value-error-analysis.json,
key-span-diagnosis.json and all saved cases/outputs.60 main-suite public CPU
replays and32 notebook replays match GPU outputs. Checkpoint SHA:
`f660aaeff5fecd71aaf3df5768d87948331bc939942f63041666fdfad7d7aaee`.
