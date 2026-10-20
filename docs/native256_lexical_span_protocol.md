# Nine-slot source-key pooling: one change, not a new compression ratio

Prospective18October2026. Current lexical64 step1,000 improves full-dev token
accuracy to44.136780%, but original positive exact is21/285 and the properly
token-boundary-prefixed notebook test has0/64 all-four binding units. Input-only
audit of570 discriminating source-value comparisons found NO owner-name token
in any existing nine-token key. Including the complete name requires11..20
tokens in these inputs. Semantic codes/query hidden states still exist, so this
is a local feature limitation, not whole-model impossibility or proof of cause.
Independent current-results verification passed after the initial draft;
matched raw1,000 remains pending. This does not change the feature hypothesis.

Test ONE change: keep nine learned256-wide key slots, but use causal ranges
0,1,2,3,4,5..8,9..12,13..20,21..32 rather than nine consecutive words. Each range
contributes its mean valid word embedding. Project query-to-vocabulary exactly
as before, then pool scalar word scores with masked prefix sums/gathers. No
dense expanded old-word key/value bank or reconstructed raw-history input.
Future/inaccessible scores are ZERO BEFORE cumulative sums, as well as masked
at candidate selection. Empty ranges contribute zero; NULL remains finite.

All original model weights/buffers, fresh seed47 initial tensors/RNG,23,209,216
trainable parameters, raw64, two read units, FP32 word-bit representation,
frozen separate encoder, one global position per64 old words and existing cache
arrays must stay identical. Incoming source/cache/data and current native/v2
checkpoints stay immutable. New source package is separate. This does not add
training examples, QA losses, decoder features or epochs. A fresh bounded
comparison, if launched, uses the same original schedule/orders/assistant CE
as lexical64 v2; final acceptance still requires full two-original-epoch quality.
Conventional raw1 is unaffected and can remain the same audited baseline run.

Compute is not free: the nine vocabulary projections remain, with additional
scalar masking/cumulative sums/range gathers over[B,Q,9,old_words]. Global
transformer attention shape and persistent KV bytes do not increase. Temporary
memory and latency may increase. Measure full-size gradients/cache/actual
rollover, all-parameter microbatch/future isolation, exact fresh states, and
complete loaded-input latency/training-body peaks BEFORE training. Compare to
the unchanged v2 kernel on identical sources; do not claim an unmeasured speed
gain. Independent explicit embedding-range equations must agree with the
optimized scalar implementation. Include actual serialized Adam/cursor resume.

Then judge original full dev/free facts/missing/changed-query+value pairs,
proper-prefix/full notebook all-four+blocked invariance, held literal values,
positions/names/wording/historical/reasoning and matched raw performance. Correct
individual words or better teacher loss alone is insufficient. Keep source/input
visibility strata and the eventual full original-test requirement. This is a
feature hypothesis; no quality/adoption result exists for this model yet.

Preflight implementation freezes sources into a new span-preflight export.
Compare span64/unchanged lexical64/conventional raw1 at batch1/4 and1/39 old
blocks: full forward, prefill, cached next token, actual64-word rollover (with
encoder for compressed only), three warmups then six rotating/reversing rounds.
Inference peaks include all three resident models. For training bodies isolate
one model at a time, require ZERO allocated GPU bytes before construction, and
profile batch4/8 on the same genuine original512-token/39-code example. Allocate
two moment-sized buffers; exclude actual Adam update/scratch and data I/O.
Never describe these as end-to-end update throughput. Fullsize CPU independent
reference constructs actual mean embedding vectors and projects them, with no
unique-vocabulary scalar prefix kernel. Audit public/cached/rollover logits and
all saved parameter gradients, every cache byte/array and measured median.
