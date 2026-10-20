# Balanced single-token fixed-position packing pilot

## Status

Both arms finished exactly512updates. Training, evaluation and independent results
verification passed. **This curriculum establishes narrow, closed-vocabulary
reading from one compressed vector. The frozen encoder already works; trainable
packing adds only a small observed gain. General retrieval and binding remain
unsolved.**

The run is `dump/token-memory-64-balanced-fixed-pilot-20261017`. All workers from
this pilot finished. Do not restart, automatically extend or adopt either model
for broad use. Native256 and both previous pilots remain unchanged; all702
protected hashes match. One vector per64tokens/one attention position is intact.

## Setup

Both arms use identical copies of the original partial native step8750 state,
fresh Adam and the original packing weights. This is diagnostic continuation,
not continuation of a failed specialist and not fresh two-epoch language training.
Reader:23,209,216 parameters,21 layers, width256, raw64. The packing arm additionally
updates the existing2,080,000 packing weights. Encoder prefix and decoder stay
frozen. Production remains **64 old tokens → one256-dimensional vector → one
attention position**; no added inference operations or KV entries.

Eight training words (red/green/white/black/water/fire/earth/air) are each exactly
one token. Every16-example reading batch contains each twice.1024four-case units
give4096unique examples and two shuffled passes/8192exposures. Each unit asks two
paraphrases against two changed memories. The source inputs differ ONLY at token24;
fact span is[15,26), value24, query begins77 (zero-based). Facts are entirely in a
completed64-token block; whole query and value+EOS fit raw64. No unknown reading
targets or decoder-success filtering.

Each update uses16original language segments+16reading examples, micro8. Language
segments/order match the prior pilots exactly. Loss .75LMCE+.25answerCE+.05frozen-D
reconstruction; FP32 AdamW.01, clip1, LR3e-5cos3e-6/warm30, packing LR.1reader.
Cached frozen-prefix features are reused read-only; final codes are recomputed
through each arm's current packing weights for reading AND language. Only2944
fresh blocks add184MiB of features. Decoder expansion is a training auxiliary;
its expanded positions never enter reader attention.

Fresh groups cover development, familiar values, new names, new wording, eight
new one-token answer words, composed changes, absent facts and actual train fit.
Retain13previous single/two-record groups as already-inspected regressions.
See the [prospective protocol](native256_balanced_pilot_protocol.md).

## Checks

Six CPU tests passed, including actual objective/microbatch gradients, exact
serialized Adam resume, frozen isolation, cache indexing and balanced/fixed
source construction. Independent data audit checked all serialized cases,
source-derived labels, fixed positions, single changed token, batch balance,
two-pass orders, original language order and702protected hashes.

Fullsize zero-update GPU preflight passed: live/cached packing-gradient error
1.91e-6, microbatch gradient error1.02e-8, exact unchanged weights and empty Adam.
Heavy peak3.482GiB allocated/3.807reserved includes Adam-equivalent buffers.
Two cached F/B samples at micro4/8 took1.627/.636s, excluding Adam, preparation,
evaluation and saving; not an end-to-end speed comparison. Actual first update
was independently audited before resuming the same checkpoint.

Each arm received1,872,347LM labels,16,384reading labels (8192value/8192EOS), and
131,072reconstruction labels. All Adam states reached512, with exact matching
orders/cursors/LR/balanced classes. Frozen packing stayed bitwise unchanged;
trainable packing changed. Observed sequential training rates were about1.73
updates/s for both; timing is confounded and is not a controlled speed comparison.

## Verified results

Evaluation measures full value+EOS, first-value teacher decision separately from EOS, constant
12.5% baseline, changed-value pairs, all-four success, blocked/shuffled controls,
train fit, new values/names/wording, missing hallucinations, retained regressions,
full original7831-segment development set, historical312 and original decoder
fact reconstruction/code drift. Independently rederive outputs and metrics.
The prespecified unit label-flip test compares true and opposite-memory values,
preserving balance/query priors; it concerns fixed checkpoints, not seed uncertainty.

The copied parent scores0 on all new reading groups. The constant-answer baseline
is32/256 (12.5%) on the familiar/name/wording groups. All answers below require
the complete value and EOS, rather than teacher-forced piece accuracy.

| Measure | Frozen packing | Trainable packing |
|---|---:|---:|
| Development | 238/256 | 241/256 |
| Familiar values, new blocks | 229/256 (89.45%) | 234/256 (91.41%) |
| New entity names, one record | 221/256 (86.33%) | 222/256 (86.72%) |
| New question wording | 232/256 (90.63%) | 234/256 (91.41%) |
| Three known-value test groups combined | 682/768 (88.80%) | 690/768 (89.84%) |
| Familiar: both changed values correct, same query | 101/128 | 106/128 |
| Familiar: all four answers correct | 50/64 | 53/64 |
| Known test groups: both changed values correct | 302/384 | 310/384 |
| Known test groups: all four correct | 141/192 | 147/192 |
| Actual training fit | 124/128 | 124/128 |
| Training fit: all four correct | 30/32 | 30/32 |
| Eight held-out answer words | 0/256 | 0/256 |
| Composed held-out changes | 0/256 | 0/256 |
| Absent requested entity | 0/256 | 0/256 |
| Full-dev token accuracy | 46.488952% | 46.481154% |
| Full-dev CE | 2.678522355 | 2.678797924 |

The first-value teacher accuracy on familiar cases is89.45%/91.41%, matching free
answer accuracy. The higher94.73%/95.70% teacher-token accuracy includes the easy
EOS target and must not be presented as factual recovery.

## Evidence that the vector is used

Matched first64 familiar cases:

| Memory condition | Frozen | Packing |
|---|---:|---:|
| Correct vector | 54/64 | 56/64 |
| Fact vector blocked, count/positions preserved | 8/64 | 8/64 |
| Shuffled vector, scored against original fact | 0/64 | 0/64 |
| Shuffled vector, replaying donor's source value | 52/64 | 55/64 |

Each matched counterfactual changes only source token24; raw query, length and
positions remain identical. Blocking the vector makes swapped-memory outputs
identical. Many complete changed-value pairs and all-four units now succeed,
unlike a constant response. The prespecified unit label-flip checks reach the
10,000-draw Monte Carlo resolution (p≈0.0001) for all three known-value groups in
both arms. These fixed-checkpoint checks establish association with vector content
on this narrow task, not training-seed certainty or general QA.

The familiar per-value counts show the main remaining closed-set confusion:

| Value | Frozen correct /32 | Packing correct /32 |
|---|---:|---:|
| air | 32 | 32 |
| black | 26 | 26 |
| earth | 30 | 30 |
| fire | 32 | 32 |
| green | 13 | 18 |
| red | 32 | 32 |
| water | 32 | 32 |
| white | 32 | 32 |

Packing's five extra familiar hits are entirely on green. Across the three
known-value groups the observed gain is+1.0417pp; a descriptive stratified
balanced-batch bootstrap gives95%CI[-0.3906,+2.4740]pp. This is not a clear
additional gain and does not cover training-seed variation.

## What remains unsolved

All256 held-word cases in each arm receive a word from the training palette.
The models learned an eight-way readout, not arbitrary word copying. Both also
return the source record's actual value in235/256 absent-entity cases, despite
the question naming somebody else. Thus they can read the value while failing
to check whether it belongs to the requested entity.

New-name success here is not multi-entity binding: positive examples contain only
one fact. The two questions in an all-four unit are paraphrases about that one
record. There are no palette-valued distractors in filler, so the experiment does
not distinguish a value-presence detector from position-specific fact selection.

All13 retained earlier single/two-record groups have0 exact answers, including
their missing cases. These regressions change answer vocabulary/length, layout
and sometimes record count together; they show no transfer but do not isolate
which change causes failure. Historical notebook facts remain0/103, changed
notebook facts0/32 and compressed colors0/8. Historical missing answers remain
21/25, unchanged from the parent. One packing historical output contains a value
substring but no correct complete answer; it is not a retrieval win.

The new recipe jointly simplifies vocabulary, target length, layout and filler,
and balances labels. Shortening to value+EOS also gives the first value decision
a larger share of answer CE. This experiment cannot assign the improvement to
any one of those changes alone.

## Language and code preservation

Parent full-dev accuracy is45.601133% (CE2.726427284). Both continuations improve
over this partial parent while replaying language, but no LM-only arm establishes
the language opportunity cost. Compared with the previous positive pilot's
46.618027% frozen result, the new frozen arm is lower by0.129075pp; the reading
data/objective difficulty differ. This is not a fresh full-budget architecture
comparison against the original512-token reference.

Packing minus frozen language accuracy is-0.007798pp, with fixed-checkpoint paired
document95%CI[-0.014842,-0.000956]pp. The small known-value gain has a slight
language cost in this run. No production adoption follows.

The original reconstruction decoder recovers ALL44,484/44,484 fact tokens and
2,560/2,560 fact bodies for parent and both arms. In the fresh fixed-position
groups, value token24 is recovered for896/896 unique codes, including all held
values and composed cases. The novel words remain in the code even though the
reader cannot answer with them.

Whole-block reconstruction is163,838/163,840 tokens (2558/2560 exact blocks) for
parent/frozen and163,835/163,840 (2557/2560 exact) for packing. Hence the codec is
not globally lossless. Packing median relative code drift is0.769014%; its
gradients were active and its weights changed. Neither decoder-expanded positions
nor recovered raw history enter reader attention.

The next separate selection test could add a palette-valued distractor at a fixed
position and swap fact/distractor values while preserving the word bag. It would
test whether the reader selects the fact rather than recognizing a present value.
Position variation and broader held-out vocabulary remain separate next axes.
No next dataset, training stage or extension has been started.

This narrow task does not prove broad binding/retrieval, even if it succeeds.
Both arms receive extra language replay; there is no matched LM-only continuation.
No automatic extension or adoption. The broader factual-reading goal stays unmet.

Artifacts: `training-verification.json`, `evaluation/report.json`,
`evaluation/independent-verification.json`, `evaluation/readability-audit.json`,
all generation/teacher records, decoder predictions, language arrays and source
snapshots. Code is in `tools/*native256_balanced_pilot.py`; the analyzer is a
separate post-evaluation saved-output audit. Frozen checkpoint SHA256:
`d6bde428995aa0ed3b00929c4ba1b0cdfd02c1948a2731cda325ea66b30bb24a`.
Packing checkpoint SHA256:
`1e5eee2c6b73444382cbdfbb1ba2b1af77020e92413948374f95461438401452`.
