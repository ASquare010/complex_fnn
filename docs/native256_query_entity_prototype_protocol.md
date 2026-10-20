# Causal query-identity prototype preflight

18 October2026, written before prototype execution or optimization. This is a
new namespace `src/models/token_memory_64_query_entity`; every earlier model,
checkpoint, source-occurrence experiment, native256 run and original cache is
immutable. No new training is authorized by a passing unit test alone.

The previous goal turn was progress: a completed, independently verified failed
source-occurrence pilot and reserved final TEST inputs. The whole goal remains
unmet. This prototype tests whether explicit causal query identity helps select
the correct OLD sentence. Literal cooccurrence is not general relation reasoning.

Keep raw64, one global attention/KV position per64 OLD BPE tokens,21 layers of
width256/four heads, and two local read units at layers7/14. Existing word-bit
metadata carries literal IDs; it is additional information, not proof that the
old semantic encoder alone is lossless. Two4x256 selectors, two256 scalar gates
and two256 presence vectors add3,072 trainable weights. Remove six FFN units:
31,374->31,368 total hidden units, exactly23,209,216 reader parameters. The frozen
9,426,432-parameter encoder is separate. Fixed vocabulary boundary/capital/word/
sentence buffers and serialized role patterns are model metadata; count their
memory explicitly. No extra persistent history/KV/query bank is permitted.

Query selection uses only the last128 visible literal IDs recovered transiently
from existing V metadata plus existing active raw IDs. Parse existing public
newline role headers, take the latest complete User body, stop at the next role
header, and exclude assistant answer text. No gold names, corpus spans, source
coordinates, target tokens or lookup-table answers are inference inputs. Missing
or truncated User scope falls back to the generic Span reader. Literal keys are
collision-free ordered sequences of up to eight12-bit BPE IDs in two int64
fields. Only complete ASCII alphanumeric words are keyed; other text falls back.
Require the word's terminating boundary also to be causally visible. Match old
sentences outside the current User body. Entity-only fallback remains when the
topic differs. Tokenizer role-like text, aliases, non-ASCII words, long words and
queries, multi-entity sentences and wrong query selection remain limitations.

Bias equals8*tanh(new_gate(h))*entity_sentence_confidence, strengthened by topic
agreement. Redistribute only the old pointer's non-EOS mass. Preserve EOS and
generic word-copy NULL mass exactly. Presence vectors start at zero. No fixed
frequency prior is used in this first prototype; capitalization is a fixed input
cue and selectors are trainable. This detail supersedes the optional frequency
prior suggestion, not the underlying hypothesis.

Before any run: independent surface/key/scope reference; bounded literal-match
reference; disabled-bias shared-core forward/all-gradient comparison; probability
conservation/finite empty controls; future/padding/current-answer isolation;
public versus cached/chunked/real64 rollover consistency; global microbatch
normalization/actual serialized optimizer resume; exact parameter and persistent
KV/model-buffer accounting; CPU/GPU agreement and actual maximum-shape training
capacity/body/decode timing including selector and matching work. Benchmark against
ordinary Span on identical inputs/shapes, without claiming speed from body-only
measurements. A failed preflight must be fixed and rechecked before training.

Future training, NOT started: fresh seed47 on identical original sampler,
48,762-update/two-original-epoch schedule and361,357,104 assistant labels, no extra
dialogs/passes. Ordinary global assistant CE plus0.05 query-selection CE averaged
over the two slots and unique original query exposures per segment, plus0.02
positive critical-source CE averaged by field/segment/read-layer. Critical source
targets are first differing name/value pieces against distractors; exclude later
gold-prefix continuation and never force generic NULL on missing cases. Auxiliary
old pointer, incoming core states and embeddings are detached; only the new query
selectors/gates receive auxiliary gradients. The all-case derivation, independent
label audit and actual objective tests must precede export. These coefficients
are prospective; any later change requires a separate recorded experiment.

Evaluate full original DEV, free facts/missing/swaps, all-four notebook binding,
legacy/historical/numeric probes and selector accuracy, with blocked controls.
Reserved full original TEST and novel literals remain unscored until the final
48,762-update/exposure gate. A narrow pilot win cannot complete the broad goal.
