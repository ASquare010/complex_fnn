# Fresh selective LM: bounded question-to-address pilot

This follows the separate integrated-cache preflight, not a continuation of any
completed512-update arm. Reader budget is23,209,216 INCLUDING frozen word decoder
and controller; frozen encoder is separate, as confirmed by the user. One FP32
code per64 source tokens, one attention/KV position/code, raw64. No gold addresses
enter model inputs. The new core/controller start fresh seed47, not native8750.
The word decoder is copied from the pretrained codec and stays frozen. No native,
completed component, original dataset, cache or checkpoint is modified.

Two matched arms each run512updates: generation-only versus selective word-copy
output. Both retain the same complete23M architecture, original-code last-layer V,
address supervision, initialization, language batches, reading batches and LR.
This isolates use of the selective output route; it is not the old native model
control or a comparison of trainable packing. Existing frozen-packing comparisons
are complete and remain immutable. Encoder/packing is frozen in both new arms.

Each update:16 original language segments plus16 reading examples, micro8; loss
.75 language CE +.25 answer CE +.25 address CE. Address targets are independently
derivable source token coordinates, and NULL for EOS/absent answers. They are
training labels only. FP32 AdamW(.01), clip1, warm30, LR3e-4 cosine to3e-5 over512.
This is two passes over a NEW4096-example reading set, not two original language
corpus epochs or a final quality comparison. No decoder reconstruction loss.

1024 four-case units each ask both entity questions against both value-swapped
two-record memories.128 single-token answer words, equal positive exposures.
Facts stay inside one completed64 block; prefix noise varies position0..25 and
record order. Names have equal BPE lengths; raw question phase is matched.
128units contain one absent requested entity (6.25% of examples), preserving
balanced positive-value counts. Training includes NULL/EOS without unknown-only
supervision. No decoder-success filtering; genuine frozen encoder codes.

Fresh held-out groups: known words/blocks, disjoint128 answer IDs, names, question
wording, unseen source offsets, missing and composed; stratify both record orders.
All fact blocks are disjoint across train/eval groups. Every positive answer word
is one token; unknown has two BPE tokens, so complete answer+EOS is required.
Held answer IDs are held from reading supervision only, not from codec pretraining
or ordinary language replay. Report this limitation.

Before training: dataset audit from source serialization/regex/token offsets,
balanced values/orders, packed/public/causal/cache tests, actual mixed objective
microbatch gradients and exact serialized Adam resume, fullsize zero-update
objective preflight. Pause after the first actual update and audit before resume.
Same initial state and fresh Adam in each arm. Unique checkpoint format prevents
resuming through native/plain trainers. Frozen sources guarded while live.

Evaluate full value+EOS, first-token and source-address accuracy separately,
both swapped memories/both questions/all-four units, unseen IDs/names/positions,
missing, blocked codes with count preserved and shuffled donor codes. Include
actual training fit, full original language dev and historical regressions.
Verify saved outputs/labels/orders/optimizer states independently. A512-step
fresh-model diagnostic cannot establish full-budget language/reasoning parity;
do not adopt or automatically extend either arm on narrow gains.
