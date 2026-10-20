# Query-conditioned literal reader: 1,000-update pilot

18 October 2026. The full quality goal remains **ACTIVE / UNMET**. This is a
bounded early pilot, not the completed two-original-epoch comparison. Training,
aggregate scoring, public CPU replay, notebook scoring/replay and the selector
diagnosis passed. Final matched-comparison audit89820 also passed, including
2,589 protected/input/source/output guards, matched exposures and all prompts.

## Budget and intervention

New run: `dump/native256-query-entity-unified-v1`, model
`src/models/token_memory_64_query_entity_v2`. The reader has exactly23,209,216
trainable parameters; the frozen9,426,432-parameter encoder is separate, matching
the user's confirmed budget. Raw context is64 tokens. Each64 old tokens occupy
one global attention/KV position; no KV entries were added.

The model selects entity and topic words from the latest causal user question,
matches literal word keys in old memory, and biases the existing word-copy path
toward the matching sentence. Two reader layers perform this operation. Derived
TRAIN-only supervision teaches query selection and critical source discrimination;
ordinary assistant-token CE continues to train the language model. Six FFN hidden
units fund3,072 new query-reader weights, keeping the exact parameter total.

This uses **added literal token-ID metadata** in the FP32 code payload. It is not
evidence that the old semantic encoder alone is lossless. Global position count
is preserved, but query selection and local literal scoring cost extra work.
There is no claim of64x wall-clock speed or64x byte compression of raw token IDs.

## Matched training audit

Both this run and the ordinary pooled-key Span control start fresh with seed47
and use the same original batches, assistant labels, learning rates and cursors.
The new architecture's initial tensors were independently verified; they are
not identical to the control because the parameter allocation differs.

Each completed1,000 updates:32,000 original segments,7,416,257 assistant labels,
all Adam counters1,000. The new auxiliary exposures are1,551 query segments/
3,102 slot labels and1,241 source segments/2,482 critical terms, half name and
half value. Missing cases receive query supervision without a forced source-NULL
target. The original48,762-update/361,357,104-label/two-epoch horizon is retained.
No new language examples or student TEST predictions were added.

Checkpoint SHA256:
`d9fd0c7b4743f1d542639ce46ab2ce848c382825455330da84fc5dac2a6684bc`.
The checkpoint is immutable; no further updates were performed.

## Verified output results

All factual figures below require the correct complete answer and EOS. Both arms
see identical prompts. The control is the completed1,000-update ordinary Span run,
not the much longer trained original512 reference.

| Measure | Ordinary Span | Query reader |
|---|---:|---:|
| Full-dev token accuracy |44.164101%|44.213472%|
| Full-dev CE |3.062043972|3.057949714|
| Natural notebook positive replies |23/285|225/285|
| Natural missing-information replies |13/63|30/63|
| Changed-value replies |3/64|49/64|
| Changed-question replies |1/64|50/64|
| Both original and changed-value correct |2/64|49/64|
| Both original and changed-question correct |0/64|39/64|
| Natural positives with memory blocked |0/64|0/64|
| Controlled notebook full replies |15/256|197/256|
| Full-reply both-swap pairs |2/128|98/128|
| Full-reply all-four sets |0/64|37/64|
| Proper name/relation-prefix completions |77/256|256/256|
| Completion both-swap pairs |11/128|128/128|
| Completion all-four sets |0/64|64/64|
| Controlled blocked full / completion |0/64 /0/64|0/64 /0/64|

The four-way check asks both questions against both same-wordbag swapped memories.
The completion prefix supplies only the name and relation, not the answer value;
its data, prefix independence and visibility were separately audited. This is a
strong notebook-specific binding improvement. The gap between full replies and
proper completions identifies remaining answer-generation errors on this set;
it does not establish universal reading or reasoning ability.

The win persists in the independently audited whole-query-raw/no-rollover stratum:
38/44 full answers and8/11 all-four sets versus3/44 and0/11 control; proper
completion44/44 and11/11 all-four versus20/44 and0/11. This small stratum rules
out question rollover as the sole explanation of this comparison's gain.

The small full-dev accuracy difference is+0.049371 percentage points, paired
document95% interval[+0.013191,+0.086192],5,000 bootstrap draws/4,852 documents.
This measures uncertainty for two fixed checkpoints, not training-seed uncertainty.
Excluding the memory source, accuracy is43.877079% versus43.850131% control.

All seven access-phrase groups remain0/512 exact each, including familiar, values,
names, wording, positions, missing and composed. The numeric reasoning probe is
0/53. Historical notebook positives reach10/103 and changed facts2/32; blocked
positives remain0/103. Broad transfer and the final quality target are unmet.
The older convenience completion probe has a known token-prefix limitation and
is not substituted for the separately audited proper notebook completion above.

## Selector and cost evidence

The learned input selector chooses both expected query words on348/348 original
DEV questions and1,551/1,551 actual TRAIN exposures. The correct old entity signal
is present for285/285 DEV positives and absent for all63 missing cases. This
diagnoses the input heads; it is not a free-answer accuracy metric.

Original V2 trained at about0.884 updates/s in this bounded run. Its earlier
fresh equal-budget benchmark was slower than ordinary Span:248.793 versus
202.301ms forward/backward body and77.562 versus46.546ms cached batch1 token.
These are component timings, not end-to-end training or deployment estimates.

A separate reversible metadata vectorization experiment passed CPU invariants
and trained GPU integer-field/gradient/cache checks. On alternating same-weight
samples, body time fell267.485 to251.868ms; cached batch1 fell79.780 to62.685ms,
batch8 80.861 to60.249ms. KV bytes are unchanged, with224 bytes of fixed role-table
storage. Full free-generation parity was not tested, and quality evaluation used
the original V2 path. Do not adopt this optimization from component timing alone.

## Interpretation and remaining work

The combined literal query reader and derived supervision substantially improve
this trained notebook task without expanding global attention positions or KV.
This experiment does not isolate architecture from auxiliary supervision, and
does not solve broad retrieval, the61.013194% long-trained mixed-memory reference,
or final fresh two-epoch parity. Keep the pilot paused. Investigate failed
cross-task query selection/output behavior before selecting a full-budget run.
Native256, prior runs, original data/caches and reserved TEST inputs remain intact.

Artifacts: `dump/native256-query-entity-unified-v1/compressed64/`
`training-verification-1000.json`, `evaluation-1000/`, `selector-diagnosis-1000/`
and `vectorized-metadata-1000/`. The passed matched comparison and its fixed-weight
paired intervals are recorded under `comparison-1000/report.json`. All workers
are terminal. Reserved student TEST/novel-literal inputs remain unscored.
