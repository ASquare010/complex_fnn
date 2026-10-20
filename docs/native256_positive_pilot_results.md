# Positive single-fact packing pilot

## Status

Both arms finished exactly 512 updates. Training, evaluation and independent
results verification passed. **Positive-only warmup removed the `unknown`
shortcut, but replaced it with a frequent-value shortcut. Trainable packing did
not establish a useful retrieval improvement.**

The run is `dump/token-memory-64-positive-pack-pilot-20261017`. Every worker from
this pilot has finished. Do not restart, extend or adopt either checkpoint. The
native256 source/run/checkpoints/logs and completed mixed-fact pilot are unchanged;
all 346 protected hashes match. The broader factual-reading goal remains unmet.

## Matched setup

Both arms start from the identical copied native256 step-8,750 state used by the
previous packing pilot, with fresh Adam. This is an additional diagnostic
continuation from a partial reader, not fresh two-epoch language training.

The reader has 23,209,216 parameters. The packing arm also updates the encoder's
existing 2,080,000 packing weights; the other arm keeps them frozen. The encoder
prefix and reconstruction decoder stay frozen. Both retain raw64 and
**64 old tokens → one 256-dimensional code → one attention position**. No
production layers or KV entries are added. Decoder reconstruction is a training
auxiliary and does not expand reader attention.

All 512 updates use positive single-fact examples. There are 1,024 new four-case
units: two different values in matching old histories, each asked using two
question paraphrases. Two shuffled passes give 8,192 exposures to 4,096 unique
examples, without any `unknown` reading target. Original language replay can
still contain missing-information answers. Every update has 16 language segments
and 16 reading examples, microbatch 8. The original 8,192 language segments and
their order exactly match the previous pilot.

Loss is .75 language CE + .25 full-answer CE + .05 reconstruction CE. FP32 AdamW,
weight decay .01, joint clipping at1; reader LR3e-5 cosine to3e-6, 30 warmup,
packing LR one tenth. See the [prospective protocol](native256_positive_pilot_protocol.md).

## Data and checks

Fresh held-out single-fact groups separate ordinary histories, held combinations,
names, question wording, new answer words and composed changes. Each has 256
positive cases; a separate 256-case group contains 52 missing cases. Actual
training fit uses 128 positive cases from the start and end of the training set.
The previous two-record groups are retained as regressions, not a blind test.

All fresh fact-bearing blocks are disjoint from old cases and other fresh splits.
Facts occur entirely in compressed memory; the whole question and answer/EOS fit
raw64. There is no decoder-success filtering. The old frozen-prefix cache is
reused read-only; only 3,072 fresh blocks add 192 MiB of features. Both arms
recompute final codes through their current packing weights.

Six CPU tests pass, including actual objective/microbatch gradients, frozen-pack
isolation, exact serialized Adam resume, positive targets and combined-cache
indexing. The independent data audit checks all cases, text-derived answers,
serialized blocks, two passes and matched language order. Full-size zero-update
GPU preflight passes: live/cached packing-gradient error1.43e-6; microbatch
gradient error1.12e-8; weights unchanged; peak3.490 GiB allocated/3.814 reserved
including Adam-equivalent buffers. Its two contended timing samples exclude Adam
and are not an end-to-end speed comparison. The actual first update was audited
before resuming the same checkpoint.

Each arm received 1,872,347 language labels, 39,512 reading labels and 131,072
reconstruction labels. All Adam states reached512; all batches, cursors and
learning rates match. Frozen packing weights are exactly unchanged; trainable
packing weights changed. All346 protected files, including the original159,
match. Observed sequential training rates were1.67/1.77 updates/s, with cache
warming and timing confounds; this is not a controlled speed comparison.

## Evaluation and interpretation

Evaluation covers complete value plus EOS, both changed memories and
both paraphrases, blocked/shuffled codes, missing-case hallucinations, two-record
regressions, full original development language, historical312 and original
decoder fact reconstruction/code drift. Independent verification rederived the
saved metrics and checked all protected hashes. A supplementary audit rederived
all historical score keys, the word-prior baseline, changed-memory pairs and
recipe/initialization hashes.

| Measure | Partial parent | Frozen packing | Trainable packing |
|---|---:|---:|---:|
| Full-dev token accuracy | 45.601133% | 46.618027% | 46.619694% |
| Full-dev CE | 2.726427284 | 2.671200706 | 2.671136603 |
| Familiar single-fact exact | 0/256 | 2/256 | 2/256 |
| Held combinations | 0/256 | 0/256 | 0/256 |
| New names | 0/256 | 4/256 | 4/256 |
| New question wording | 0/256 | 8/256 | 8/256 |
| New answer words | 0/256 | 0/256 | 0/256 |
| Composed transfer | 0/256 | 0/256 | 0/256 |
| Six single-fact test groups total | 0/1,536 | 14/1,536 | 14/1,536 |
| Both changed values correct, same question | 0/768 | 0/768 | 0/768 |
| All four correct, both questions/memories | 0/384 | 0/384 | 0/384 |
| Actual positive training fit | 0/128 | 2/128 | 2/128 |
| Training-fit changed-value pairs | 0/64 | 0/64 | 0/64 |
| Fresh single-record missing facts | 0/52 | 0/52 | 0/52 |
| Six two-record test groups, positive exact | 0/1,380 | 4/1,380 | 5/1,380 |
| Two-record changed-value question pairs | 0/690 | 0/690 | 0/690 |
| Two-record test missing facts | 0/156 | 0/156 | 0/156 |
| Historical notebook facts | 0/103 | 0/103 | 0/103 |
| Historical changed notebook facts | 0/32 | 0/32 | 0/32 |

Every development/test/training-fit group has zero all-four successes. The
separate single-fact development group scores 0/256, and the mixed single-record
group's positive subset scores 2/204 for both trained arms. Two-record development
scores 2/230 positive; the extra packing-arm test hit is on new names (4→5/230).
Historical compressed colors remain 0/8, and historical missing answers remain
21/25 for parent and both arms.

Packing minus frozen language accuracy is only **+0.001667 percentage points**;
the paired-document 95% interval is **[-0.001184,+0.004485]pp**. These are fixed
checkpoints on one seed, not training-seed uncertainty. Both arms received extra
language training; their gain over the partial parent cannot be attributed to
positive reading supervision without a matched language-only continuation.

## What changed and what failed

On the exact same familiar two-record cases used by the previous mixed pilot,
false `unknown` responses fall from 228/230 to 0/230 for both arms. Positive exact
accuracy remains 0/230. Correct missing answers fall from 24/26 to 0/26. On the new
single-record missing suite, every one of 52 missing cases gets a non-`unknown`
response. The positive curriculum changes the shortcut rather than establishing
when an answer is supported.

The most common training value is **`narrow candle`**, only 94/4,096 examples.
Nevertheless, it appears in 219/256 familiar outputs for frozen packing and
218/256 for trainable packing. Always answering that value would score 12/1,536
on the six positive test groups, compared with 14/1,536 for either model. Those
same 12 model hits keep the identical response under the changed fact. The two
remaining hits are isolated new-wording cases and fail their matched changed
value. No question gets both changed values right across the 768 test pairs.

Both training-fit hits are also the constant value; the swapped fact leaves the
response unchanged. Training-fit outputs are `narrow candle` 119/128 frozen and
117/128 packing. Thus failure is already visible on actual seen examples, not
only held-out generalization. Complete-value substring checks do not reveal
additional positive successes hidden by answer formatting.

There is some sensitivity to memory, which should not be called correct reading.
For the first 64 new-wording cases, both models score 4 with genuine memory, 1 with
the fact vector blocked and 2 with shuffled memory; all-four remains 0. This small
control does not establish robust factual access.

## Compression and content preservation

The original reconstruction decoder recovers **39,947/39,948 fact tokens and
1,919/1,920 fact bodies** from the evaluated codes for all three arms. It is
nearly exact on this sample, not globally lossless. Whole-block recovery is
122,873/122,880 tokens for parent/frozen and 122,872/122,880 for packing; exact
blocks are 1,913/1,920 for every arm. Packing median relative code drift is
0.787352%. Packing gradients were active and weights changed while sampled fact
content stayed recoverable by the original decoder.

This does not prove the reader can access that representation, that packing can
never help, or that 23M parameters are universally insufficient. It shows that
this 512-step recipe did not establish reliable access, even after removing
missing-answer targets. The 64:1 attention-position compression, production
operations and KV-entry count remain unchanged.

Lower answer CE or fewer `unknown` outputs alone is not factual reading. A common
value can replace the old abstention shortcut. Compared with the previous pilot,
this curriculum jointly changes positivity, record count and repeated exposure;
the new frozen-versus-packing comparison itself is matched. One-seed, partial
parent and synthetic-task limits apply. No automatic extension or adoption.

A next separate learnability test could use a balanced small vocabulary of
single-token values and fixed fact placement, require correct changed-value
pairs above constant/blocked/shuffled controls, then add position variation,
longer values, absent facts and multiple records. That test has not been started.

The separate entrypoints are `tools/prepare_native256_positive_pilot.py`,
`tools/preflight_native256_positive_pilot.py`,
`tools/train_native256_positive_pilot.py`,
`tools/evaluate_native256_positive_pilot.py` and
`tools/verify_native256_positive_pilot.py`. Use their unique checkpoint format;
never resume these artifacts through the original native trainer.

Artifacts include `training-verification.json`, `evaluation/report.json`,
`evaluation/independent-verification.json`,
`evaluation/behavior-and-prior-audit.json`, all generated/teacher outputs,
reconstruction predictions, per-segment language arrays and protected hashes.
Frozen checkpoint SHA256:
`624cdc884cf6c6027fa60239640c25d94cce3d93ae58dd3f3e979d1a7fa8630b`.
Packing checkpoint SHA256:
`07de2f3df4447b8926734cfaac846117d1b4ff4cd48f914ab9d1560c3396ff20`.
