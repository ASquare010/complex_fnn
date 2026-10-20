# Matched original-corpus1,000: language gain, no robust binding

18October2026. Raw1 and old lexical64 trained from fresh47 for the SAME first
1,000 updates of the original48,762-update/two-epoch schedule,32,000 original
segments and7,416,257 assistant labels. Actual indices/labels/LR/cursors match.
Both23,209,216 trainable reader parameters; separate frozen encoder for compressed
mode. These are early checkpoints, not two completed original epochs.

Both complete evaluations and independent training/aggregate/public CPU/notebook
audits passed.60 uncached public CPU replays and32 notebook replays per model
match saved GPU outputs. All prompts/targets match; blocked swaps are invariant.
All guarded earlier source/cache/model files remain unchanged, including native256.

| Same-step outcome | Conventional raw1 | Old lexical64 |
|---|---:|---:|
| Full-dev token accuracy |33.631227%|44.136780%|
| Full-dev CE |3.694165|3.065599|
| Positive complete fact answers |0/285|21/285|
| Missing-information answers |59/63|0/63|
| Changed-value answers |0/64|3/64|
| Changed-query answers |0/64|3/64|
| Both original/value-swap correct |0/64|1/64|
| Both original/query-change correct |0/64|0/64|
| Full notebook all-four |0/64|0/64|
| Proper-name-prefix notebook all-four |0/64|0/64|
| Proper-name-prefix complete values |0/256|78/256|
| Historical positive facts |0/103|8/103|
| Historical changed facts |0/32|3/32|
| Limited numeric final-answer probe |0/53|0/53|

Raw returns the missing-information template on280/285 positive cases. Its
missing-answer success is therefore abstention, not broad factual reading.
Old lexical64 often supplies a source value, but fails reliable question/owner
binding and missing information. Conditional prefixes include the known name;
their individual value successes cannot replace full question answering.
All seven legacy access groups have0/512 exact and zero all-four in both models.
Those formats were not current extra training examples; zeros do not establish
a universal capacity limit. Numeric EOS counts are1 raw/3 lexical, cap256;
this small probe does not establish general reasoning parity.

The lexical-minus-raw language difference is+10.505553 percentage points, with
paired fixed-checkpoint document95% interval[10.202668,10.811576] across4,852
documents/5,000 bootstrap draws. This measures document uncertainty for these
fixed models, not seed uncertainty. The comparison changes architecture and
representation: literal token IDs are added in low FP32 bits, and a dedicated
reader replaces some FFN capacity. It does not isolate one cause or prove the
original semantic encoder lossless. Global OLD attention/KV64:1 remains; local
reading adds work. Whole-cache savings include raw/reserve and are smaller.

Both models remain zero all-four in the prospectively specified raw-name/no-roll
and whole-query-raw/no-roll strata. Thus query compression/rollover alone cannot
explain this failure. Matching the weak raw early checkpoint is insufficient for
adoption. The stronger old mixed512 reference61.013194% remains a separate final
reference; it includes older compressed memory and is not memory-free or a
matched early-step control. Full two-original-epoch/broad/test goal stays unmet.

The separate pooled-key reader is now trained and training-audited at1,000 on the
same original exposures. Its evaluation and independent audits are complete. Language
44.164101%/CE3.062044, positives23/285,missing13/63,changedvalue3/64/query1/64,
bothvalue2/64/query0/64. Full/known-name notebook all-four remains0/64, with the
same value for both queried people in121/128 known-name comparisons. Paired
language interval includes zero; no robust binding gain or adoption result.
Do not extend/adopt automatically. See [span cost/protocol](native256_lexical_span_results.md).

Authoritative matched export: `dump/native256-lexical-full-unified-v2/comparison-
1000/report.json`, with all fixed checkpoint/source/input/audit hashes. Current
span snapshot SHA`a9d853aa45c3d9e3831d6a5e2b0b61f6da907317f04a11180db793c2d00244a8`.
