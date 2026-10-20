# Position-varied512: verified results

The new frozen-packing arm completed exactly512 updates. Independent data,
first-update, complete training and result audits passed. **Position variation
gave a small language gain but did not solve robust factual reading.** Do not
adopt or automatically extend this pilot. The full quality goal remains unmet.

Run: `dump/native256-position-curriculum-pilot-20261017`.
Data: `dump/native256-position-curriculum-data-20261017`.
The fixed-position control is the immutable completed balanced-pilot frozen arm;
it was evaluated on the same fresh cases and was not retrained. Native256 and all
older sources, weights, corpora and caches remain unchanged.

## Matched intervention

Both readers start from the exact same copied native step8750 tensors with fresh
Adam and original frozen encoder/packing. The new arm preserves the control's
1024 four-case units, eight balanced one-token values, source word bags, raw
queries, targets, language segments, order,512 updates and learning rates. It
changes the fact offset to0/8/16/24 according to original batch and reading epoch.
Each unit sees two different positions; each position receives2048 exposures.

Both use16LM+16reading cases/update, micro8, .75LMCE+.25answerCE+.05frozen-D
reconstruction, FP32 AdamW.01, clipping1, LR3e-5cos3e-6/warm30. Every batch has each
value twice. Histories have47 body tokens; query body begins77. Every fact is
entirely in the first completed64-token block; answers fit raw64. No gold position,
decoder feature or reconstructed history is passed to reader attention.

Reader size remains23,209,216 parameters; inference keeps one attention/KV slot
per64 old tokens and adds no operations. The deployed frozen encoder separately
has9,426,432 parameters. Thus these are23M-reader diagnostics, not a23M-total-system
claim. The final budget interpretation is awaiting clarification.

This is additional diagnostic training from a partial checkpoint, **not fresh
initialization or two original corpus epochs**. Two passes concern only the small
reading curriculum. A final matched all-raw/full-budget comparison is still needed.

## Fresh held-out generation

All counts require the complete value and EOS. Each offset has the same64 fresh
units/256 cases, with identical word bags and raw queries across offsets.

| Fact offset | Fixed-position training | Four-position training |
|---|---:|---:|
| 0 (trained by both) | 225/256 | 91/256 |
| 4 (unseen) | 34/256 | 35/256 |
| 8 (new arm trained) | 35/256 | 92/256 |
| 12 (unseen) | 34/256 | 36/256 |
| 16 (new arm trained) | 29/256 | 71/256 |
| 20 (unseen) | 29/256 | 43/256 |
| 24 (new arm trained) | 31/256 | 91/256 |
| 28 (unseen) | 31/256 | 44/256 |
| 32 (unseen) | 26/256 | 35/256 |
| 36 (unseen) | 26/256 | 40/256 |

Across four trained offsets, exact answers rise320→345/1024 (31.25→33.69%), but
both-changed-values success falls99→66/512 pairs and all-four success48→26/256
units. Across six unseen offsets, exact answers rise180→233/1536 (11.72→15.17%),
with only0→2/768 pairs and **zero all-four units in both models**. Blocked controls
score8/64 for familiar values, the balanced constant-word baseline.

Descriptive balanced-batch bootstrap intervals for exact-answer changes are
[-1.27,+6.25] percentage points at trained offsets and[-.39,+6.90] at unseen
offsets. Across all offsets the change is+3.047pp, interval[-.039,+6.055]. These
resample aligned16-case batches across positions and concern fixed checkpoints,
not training seeds. The intervals and failed pair/all-four tests do not establish
a robust overall retrieval improvement.

Other fresh results remain poor: new names at offset0 drop208→65/256, new wording
215→88/256. Their offset12 results are28→29 and36→42. Unseen answer words score
0/512 across offsets0+12, composed changes0/256, and absent-entity cases0/512 in
both arms. The new arm's actual training subset scores46/128, only1/64 pairs and
0/32 all-four units, showing substantial underfitting at this diagnostic budget.
This does not prove a capacity limit or failure after a full two-epoch schedule.

Already-inspected two-record regressions score65/256 and55/256 in the two row
orders, versus116/256 and115/256 for the old fixed reader. Both readers remain
zero all-four on these binding groups. Historical notebook facts0/103, changed
facts0/32 and compressed colors0/8 remain unsolved; historical missing21/25 is
unchanged. No transfer to general factual reading was established.

## Language and encoded content

Full original development validation uses7,831 segments and1,859,388 labels:

| Measure | Fixed control | Position-varied |
|---|---:|---:|
| CE | 2.678522355 | 2.667544632 |
| Token accuracy | 46.488952% | 46.689825% |

The+0.200873pp difference has paired-document95% interval[+.181533,+.219036]pp.
Accuracy improves in14/17 source groups; the memory source itself is-.0264pp.
Reasoning-source token metrics are not solved-task reasoning accuracy. Both arms
received identical replay, so their comparison isolates this curriculum change;
gains over the partial parent remain confounded with extra replay. One seed and
partial initialization prevent full-budget quality conclusions.

The original frozen decoder reconstructs all225,280 tokens in3,520 unique
evaluation blocks exactly. Independently checked fact spans contain83,072/83,072
correct tokens and7,040/7,040 exact case-level bodies (cases reuse blocks). Content
is available to that decoder on this suite. It is not a globally lossless-codec
claim, and those expanded positions never enter the reader.

A separate CPU oracle-position check selects a single trained unpack feature
without decoding all positions. It matches the full first unpack to9.54e-6, but
its features passed to the EXISTING post-decoder norm/word head score0/2,048 value
tokens, versus2,048/2,048 for the full decoder. Independent source-position and
full-unpack recomputation passed. Gold positions are supplied intentionally;
this is not retrieval or a test of a trained new bridge. It rules out assuming
that copying the unpack map alone immediately produces word identities.

## Checks and remaining action

Nine CPU tests passed, including actual gradients/microbatch/frozen isolation,
serialized Adam resume, position transformations, exposure schedules and
answer-free prefixes. All9,600 new blocks were independently re-encoded exactly;
new feature cache600MiB, original caches read-only. Zero-update fullsize preflight
had live-packing gradient error9.54e-7, microbatch gradient error8.85e-9, unchanged
weights and3.807GiB reserved including Adam-equivalent buffers. Timing samples
exclude Adam/cache/evaluation/saving and are not a speed comparison.

All Adam states reached512;1,872,347LM/16,384reading/131,072reconstruction labels
match the fixed control. Packing stayed bitwise unchanged. Evaluation verifies
unchanged weights and1,311 protected hashes; independent verification also
rechecks1,260 old protected assets. All workers finished. Checkpoint SHA256:
`cf46921b3bbaebd2bd79c07df2cc47512a2897d3f74852062aef93e673eeed4d`.

Simply varying positions at512 updates trades away strong fixed-slot reading
without reliable unseen-position or entity transfer. A next separate hypothesis
is explicit address supervision and a trained query-conditioned code readout,
with decoder-informed features rather than an untrained projection/head pairing.
It needs a parameter-budget audit, actual cache-layout proof, speed benchmark and
held-out word/address/binding tests before any quality claim. No implementation
or automatic further training of that hypothesis is claimed here.

See [protocol](native256_position_curriculum_protocol.md),
[research/component caveats](native256_compressed_reading_next_hypotheses.md) and
[full-goal evidence gaps](token_memory_64_goal_status.md). Final adoption still
requires fresh matched two-original-epoch training, an actual all-raw reference,
broad factual/reasoning evaluation and preserved language quality at64:1.
