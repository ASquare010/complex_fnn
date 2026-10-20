# Source-occurrence1,000 pilot: verified failure, no adoption

18 October 2026. Training, full evaluation, source/aggregate audits,60 independent
public CPU replays,32 notebook CPU replays, blocked-swap controls and paired
comparison all completed successfully. All workers are terminal. No native256,
original corpus/cache/model or completed control checkpoint was modified.

Both arms are fresh47 on the same original first1,000 updates:32,000 segments,
7,416,257 assistant labels, identical orders/LR/cursors, all Adam counters1,000.
The new arm added16,834 derived source targets across1,553 original segments
(1,243 positive,310 missing). It uses the same23,209,216-parameter pooled-key
reader, frozen9,426,432-parameter encoder,64:1 old global positions/raw64 and
unchanged inference/KV. This is not two original epochs done.

| Verified outcome | Ordinary pooled-key control | Source-alignment arm |
|---|---:|---:|
| Full-dev accuracy |44.164101%|42.651722%|
| Full-dev CE |3.062044|3.178111|
| Complete positive facts |23/285|20/285|
| Complete missing answers |13/63|0/63|
| Changed-value both-correct pairs |2/64|3/64|
| Changed-query both-correct pairs |0/64|0/64|
| Controlled full replies |15/256|15/256|
| Controlled known-name completions |77/256|76/256|
| Full / completion all-four |0/64 /0/64|0/64 /0/64|

The language difference is−1.512379pp, fixed-checkpoint paired-document95% interval
[-1.559751,−1.464757]. Positive-fact difference−1.052632pp has source interval
[-4.912281,2.807018]. Missing-answer difference−20.634921pp has interval
[-30.158730,−11.111111]. These are fixed-weight cluster intervals, not seed
uncertainty. There is a clear language cost and no robust retrieval gain.

All-four remains zero in the raw-name/no-roll and whole-query-raw/no-roll strata.
In the proper completion test,119/128 same-memory question changes still return
the same value(control121/128); both questions are correct in0/128 memories
(control1/128). One-question swap pairs10/128 versus11/128 do not demonstrate
binding. Full replies have1/128 swap pairs versus2/128. All64 matched blocked
question/format pairs have identical swapped outputs; blocked strict hits are0.

All seven legacy groups remain0/512. Historical positive replies improve3/103
to7/103 and changed replies0/32 to2/32, while historical missing8/25 falls to0.
These small isolated gains do not offset broad failures. Numeric probe remains
0/53,EOS2 versus4 at the identical256-token cap; not general reasoning evidence.
Recent-color2/8 has zero EOS and is not strict complete recovery. The older
natural-completion1/477 result retains a known trailing-space prefix confound;
the proper notebook completion above is the relevant controlled measure.

The independent conditional diagnosis explains why lower auxiliary loss was
misleading. First-name-piece LM accuracy falls81/285 to63/285. Correct source
top1 falls29.47% to25.96%, with source NULL probability rising0.0000208 to0.08760.
First-color LM accuracy rises115/285 to131/285, but the correct SOURCE occurrence
is top1 only34.74% with three records in each case. Later value tokens have
gold name/color prefixes: their word accuracy92.21% to92.66% and source CE0.5480
to0.4770 are conditional continuation, not free question binding.

Output copy mass is already high:0.873 at first name and0.996 at first color.
Weak copy gating is not supported as the main bottleneck. Missing-first-token
NULL CE improves13.9425 to2.6635, yet LM accuracy falls13/63 to0/63. Correct
"I" copy contribution falls0.21108 to0.00340; generator contribution only reaches
0.00207. Generic word-copy NULL is not equivalent to absence of a queried fact:
copying "I" from an acknowledgement can be legitimate. This auxiliary constraint
suppresses a valid output path without establishing a correct missing response.

The exact coefficient audit finds779 updates with auxiliary labels. Layer-averaged
loss coefficient totals are4.770684 for1,241 first-name exposures,15.366667 for310
NULL exposures and57.762649 for later positive pieces. NULL/first-name ratio is
3.221061, compared with0.248 in ordinary LM CE. Two positive tail-only segments
contain no first-name term. These are coefficients, not gradient norms or causal
proof; nevertheless per-piece means underweight the binding decision relative to
teacher-prefix continuation and NULL. All DEV first name/I pieces are at body
offset0, so an initially considered whitespace-phase explanation was rejected.

Do not adopt or automatically extend this completed pilot. Preserve its checkpoint
and records. A possible next reader is described in the
[query-entity hypothesis](native256_query_entity_reader_hypothesis.md): expose
causal query words to a small selector, use exact lexical identity to bias old
record sentences, preserve EOS/word-copy NULL, and confine extra gradients to the
new path. It is unimplemented and unproven, with explicit scope/alias limitations.

Authoritative evidence:

- `dump/native256-source-occurrence-unified-v2/comparison-1000/report.json`
- `compressed64/evaluation-1000/independent-{aggregate,public}-verification.json`
- `compressed64/evaluation-1000/notebook-binding/independent-verification.json`
- `dump/native256-source-occurrence-teacher-diagnosis-1000/{report,independent-verification}.json`
- `dump/native256-source-occurrence-weight-audit-1000/report.json`

The original full TEST input cache and reserved1,664-case novel-literal inputs
are independently prepared/CPU/GPU verified, with zero student test predictions.
Final fresh two-epoch training, original TEST/novel-literal scoring and broad
quality/efficiency parity remain outstanding. Goal ACTIVE / UNMET.
