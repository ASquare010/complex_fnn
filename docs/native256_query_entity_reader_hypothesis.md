# Next hypothesis: expose query identity to a small lexical reader

Current status: the proposal below now has a separate repaired V2 prototype,
audited original-data labels and zero-update checks. It remains untrained on the
corpus and unproven for quality, with material timing costs. See
[verified preparation](native256_query_entity_preparation_results.md). The text
below preserves the prospective proposal rather than claiming an empirical win.

18 October 2026. PROPOSED ONLY; no new model, training data or optimization.
Finish the current source-occurrence comparison and keep its checkpoint immutable.
This is not a demonstrated fix or a declaration that23M cannot learn binding.

The verified conditional diagnosis does not support a weak output-copy gate:
its mean is already0.873 at the first name piece and0.996 at the first color.
First-name accuracy worsens and first-color source selection is near chance,
despite strong continuation after gold answer tokens. Original development
names are held out from the memory training names. Generic source-word NULL
also has a different meaning from "no matching fact": a missing response may
legitimately copy "I" from an acknowledgement. Forcing NULL removed that useful
copy path. The actual auxiliary coefficient on NULL was3.221x the coefficient
on first-name pieces, versus a0.248x ratio in ordinary LM CE. These observations
motivate a query-identity route; they do not prove one sufficient cause.

A focused new reader can retain the current generic word reader, and add a small
query selector and exact lexical match bias. Recover only the recent128 causal
word IDs from existing memory-V metadata plus the existing active raw-token array.
These are temporary IDs for the reader, not additional global attention/KV slots.
Keep the LM's raw64 window. Identify the latest User body using its existing
serialized role headers; exclude assistant answer tokens from query selection.
If a complete User scope is unavailable, use the unchanged generic reader.

Two small convolutional selectors score whole-word starts in that User scope:
entity and topic/object. Inputs use normalized word embeddings plus fixed tokenizer
boundary/case/punctuation features. Four causal predecessor slots suffice for
role cues at a word start. A fixed TRAIN-token-frequency bias can favor content
words; its metadata is global and must be counted in memory measurements.
Supervise actual query-name and notebook-word positions from ORIGINAL TRAIN
metadata, both positive and missing cases. No gold name/topic is an inference
input. Measure held-out selector accuracy separately from complete fact recovery.

Select the two word keys, with confidence, and match their complete ordered BPE
sequences against visible old input. An exact key of up to8 BPE IDs can be stored
collision-free as two int64 fields plus length; longer words must fall back to
the generic reader, with that limitation reported. Require complete visible word
boundaries. Exclude the current User question's own occurrences from match evidence.
Use punctuation/EOS sentence groups: a source sentence containing the entity
receives a confidence-weighted bias, strengthened by a matching topic word.
Keep an entity-only fallback so a topic synonym does not hard-disable the cue.
This is a lexical cooccurrence cue,
not proof that every semantic relation in that sentence answers the question.

At each existing read layer, add8*tanh(g(h))*match to non-EOS word scores. Preserve
the original total EOS and internal NULL probability exactly; normalize ONLY
within the original non-EOS mass. This avoids forcing endless copying or confusing
record absence with generic word-copy suppression. A small zero-initialized
presence vector can expose whether such an old sentence exists to the decoder;
gate it by selector confidence and valid query scope. Relation aliases, multiple
entities per sentence, long words/queries and wrong selector choices remain risks.

Budget proposal: two selectors, each4x256 weights=2,048 parameters; two256-wide
scalar gates=512; two256-wide presence vectors=512. Total3,072 new reader weights.
Remove six FFN hidden units in total (512 weights each), changing31374 to31368,
for exactly23,209,216 trainable reader parameters. Encoder9,426,432 stays separate
and frozen. Only fixed vocabulary/role metadata buffers are added; preserve every
KV array, shape, byte count and one global position per64 old tokens. Local work
is linear query-context lookup and query-to-old-word matching, not old-word n^2
self-attention. Benchmark actual total compute/temporary memory before training.

Training proposal: ordinary global assistant CE, query-slot CE on input positions,
and a modest POSITIVE critical-source loss. Critical positions are first differing
name/value pieces against distractor records, so a gold differentiating prefix
cannot trivialize the decision. Do not force generic NULL for missing responses.
For auxiliary source loss, detach old pointer probabilities, incoming core states
and word embeddings: update only the new gates/selectors. Ordinary CE still updates
all weights and the presence vectors. These choices aim to limit language
interference but do not guarantee preservation. Exact coefficients/curriculum
must be fixed prospectively before a new run.

Before optimization require independent input/query-slot/critical-source audits,
explicit-loop versus vectorized gradients, exact disabled-bias behavior on shared
core shapes, EOS/NULL conservation, causal/future/padding isolation, cache/public/
real-rollover consistency, budget/accounting and serialized Adam resume. Fresh47,
original data/orders/labels/two-epoch schedule; no current run changes. Evaluate
all original language/fact/missing/historical/legacy/notebook groups and selectors.
Full original TEST and reserved novel literals remain gated to actual48,762 steps.
Broad final parity and efficiency, not a small structured reading success, remain
the objective. No implementation or new run has started for this hypothesis.
