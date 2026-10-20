# Possible next component: copy from the causal raw prefix too

18October2026. UNIMPLEMENTED; finish the current pooled-key1,000-update comparison
first. This is a conditional follow-up, not a change to the running model.

Observed old lexical1,000 outputs: requested name95/285, requested whole value
60/285, both21/285. Controlled proper-name prefix gives78/256 complete values,
but0/64 all-four units; full replies22/256. Raw1,000 gives0/285 positive and
280 false unknowns; its proper-prefix notebook currently0/256. Independent raw
audits are pending. These counts show separate name/decision/value problems;
they do not prove one mechanism causes them.

Current code exposes only OLD memory IDs to the dedicated copy distribution:
lexical_lm.Model.hidden/feed_tokens builds reader.features from memory V payloads.
Current raw tokens enter the ordinary core/generator, but the copy branch cannot
directly select a name appearing only in the recent question. Existing cache
already keeps those64 raw input IDs. Wider33-token source keys address owner-name
coverage at value candidates; they do not explicitly add this separate channel.

If pooled keys improve conditional binding while full replies still fail name
selection, test ONE further change: include the current CAUSAL raw prefix in the
local reader's candidate bank. Keep same nine weights/23M/same global positions,
same existing raw-token array and no extra persistent history/KV bank. Additional
local scalar candidates are at most64 per query window; no expanded transformer
memory positions. It may help copying without solving binding or abstention.

Critical implementation requirements: teacher input x[t] is previous content,
so candidates may include x[0:t+1], NEVER x[t+1] or y[t]. Packed512 training has
eight separate64-token windows; each query may use only its own causal raw window
and already-visible OLD codes, even when batch tensors contain future codes and
raw IDs. Prefix key context across a boundary must read permitted old content,
not the future code of the current window. Exclude BOS/PAD, retain role/word
ordering, blocked-code controls and exact physical count/positions. Raw candidates
remaining during a code ablation must not leak any old fact value. Test this with
independent equations and all-parameter/future-input gradients, cached/public/
actual rollover, real original batches and serialized Adam resume before training.

Fresh initialization/state/RNG/labels/order/LR must match current span, with
unchanged total original two-epoch budget. Measure added temporary memory/compute;
candidate count alone is not a speed proof. Require full name/value/EOS, missing,
both swapped memories/both questions/all four and original language preservation.
Correctly copying the queried name alone is not memory retrieval. Query-only
missing names could increase hallucination, so abstention must be tested directly.
No adoption or next-run commitment follows from this untested idea.
