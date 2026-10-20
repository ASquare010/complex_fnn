# Retrieval-only capacity comparison, 16 October 2026

This bounded diagnostic asks whether the current completed23M plain64 model can
learn direct retrieval from fresh compressed fact blocks with answer-only loss.
It is not the requested final fresh two-epoch language comparison. The full goal
still requires broad factual reliability, language quality beyond the roughly61%
raw512-plus-memory reference, and retained compression efficiency.

## Fixed design, before training

Both arms start from identical completed plain64 step48762 tensors, fresh Adam.
The `code` arm uses the unchanged plain64 model, legacy positions, frozen original
codec, one memory position per64-token block and no new KV/reader parameters.
The `raw` arm sees the exact same serialized source histories and questions as
raw tokens, with no compressed codes, using the original up-to512-token forward.
The raw arm is a diagnostic control with higher attention cost. It is not a
proposed replacement for raw64 or an equal-inference-cost architecture ablation.
Both retain23,209,216 trainable parameters; all original files stay unchanged.

Data:16384 paired training histories/32768 examples, four256-example development
groups and six512-example test groups. One or two facts about colors, multiword
access words or notebook locations. Two-record positive pairs swap values while
preserving their word bag;20% of pairs replace the queried entity on one side,
giving approximately10% missing examples. Output is only the value or `unknown`,
including EOS, predicted over the full vocabulary. Entire question and the first
differing-answer prefix are identical within each compressed pair.

Train has16 names,8 colors,32 access phrases and64 locations. Training never
contains held key/value bindings, including distractor/missing replacement facts.
Names co-occur within their binding-partition group; this is a synthetic-task
limitation. New history labels, filler mixtures, order and values create fresh
blocks. All split/group prompts and fact-bearing blocks are disjoint. Filler-only
blocks may repeat. There are54437 distinct completed blocks, not a128-code bank.

Test factors are separated: familiar distribution/new blocks, held bindings,
held names, held question wording, held values, and names+values+wording together.
Dev names and wording differ from corresponding test names and wording. The
values remain from finite prespecified lists; arbitrary strings, many records,
corrections and unrestricted language are not established by these tests.

The entire question stays in raw64, all facts are earlier compressed content,
and all teacher targets fit before the next64 rollover. Production generation
must still use actual token rollover if wrong outputs continue. Raw prompts are
at most308 tokens across these groups, so a24-token generation cap needs no raw
control rollover; every gold answer fits that cap. Both arms have exactly the
same source information and targets, although their positions/cost differ.

Training:2048 updates/arm, two passes over the new data, effective32/micro16,
pair order seed4701650+epoch, FP32/TF32off, AdamW.01, clip1, LR3e-5 cosine3e-6,
warmup50. Full-vocabulary CE on answer tokens only, no language replay, no extra
contrastive loss, no decoder features. Each arm sees65536examples/354284 labels.
This explicitly tests specialization; language regression is a risk to measure,
not evidence of final success. Original checkpoints/caches are read-only.

## Checks completed before launch

All-case independent audit verifies source serialization, answer semantics from
sentence regexes, target shift, matching pair queries/positions, held bindings,
block provenance, disjoint fact blocks and exact code tensor dimensions. An
initial audit used a substring check that confused `red` inside `stored`; only
the auditor was corrected to whole-word matching. Data was unchanged.

Original frozen decoder diagnostic: all fact-body tokens and bodies in every
dev/test group reconstruct exactly. Train has684291/684318 correct fact tokens
and32754/32768 exact fact bodies; these errors are retained, not filtered. The
decoder is used only for this audit. All124 protected original files match.

Four CPU tests pass: raw/code target and causal-input alignment, original public
forward/full-parameter gradient parity, future/padding isolation, actual objective
microbatch equivalence and serialized two-epoch tiny Adam resume. Fullsize actual
data zero-update GPU preflight passes, original weights unchanged, gradient error
under1e-5. Heavy capacity with Adam/scratch equivalents: raw2.307GiB allocated/
2.920GiB reserved; code.872/.939GiB. Three update-body samples exclude Adam,
encoding, I/O, validation and saving, so are not end-to-end speed claims.

## Evaluation and decision rules

Check each trained arm and its parent on held-out answer-only teacher CE,
first-differing-token full-vocabulary accuracy, free complete answers with EOS,
paired correctness, missing answers and false abstention. Use wrong/swapped and
blocked code controls at unchanged query positions. Measure training-fit examples
separately. Report all groups, not just familiar ones. Decoder agreement does not
substitute for generated answer correctness.

Measure the trained weights on all7831 original development segments using the
unchanged plain64 production inference path, including the raw-trained control
weights. Label this transfer evaluation clearly. Reuse verified parent metrics
only with unchanged checkpoint/data provenance. Check historical factual cases
for broader transfer. No checkpoint is selected on test results; fixed2048 is the
prespecified endpoint. No automatic extension based on lower training CE.

Raw success/code failure would locate a difficulty accessing compressed contents
under this recipe, not mathematically prove insufficient capacity. Success only
on training/familiar cases would not establish general retrieval. Even perfect
synthetic scores cannot complete the broad user goal or hide language damage.

Data: `dump/token-memory-64-retrieval-only-data-20261016`.
Run: `dump/token-memory-64-retrieval-only-pilot-20261016`.
