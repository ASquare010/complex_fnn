# Native256 joint-packing pilot

## Outcome

Both arms completed 512 updates. Training, evaluation and independent results
verification passed. **Unfreezing the packing layers did not improve factual
retrieval in this pilot.**

The original native256 source, run, logs and checkpoints are unchanged. Both
arms keep **64 old tokens → one 256-dimensional code → one attention position**.
All 159 protected original files match their recorded hashes. Every worker has
finished; neither arm should be adopted or automatically extended.

## Matched comparison

Both arms start from an identical copy of the native256 checkpoint at step
8,750, with fresh Adam. This is a diagnostic continuation from a partially
trained reader, not a completed-native or fresh two-epoch comparison.

The reader has 23,209,216 parameters, width 256, 21 layers and a raw64 window.
The frozen arm updates only the reader. The packing arm additionally updates the
encoder's existing 2,080,000 packing weights: the shared 256→125 token map and
ordered 64×125→256 span map. No production layers, parameters or KV entries are
added. Unfreezing these existing weights adds training gradients and Adam state.

The frozen encoder prefix is precomputed in FP32. Its intermediate features
never enter reader attention. Final codes are recomputed using each arm's
current packing weights for both reading examples and language replay.

See the [prospective protocol](token_memory_64_pack_pilot_protocol.md).

## Data and training

- 8,192 fresh reading examples: 2,048 single-record examples followed by 6,144
  two-record examples, including both questions against both swapped memories.
- 8,192 original language segments for replay, with 16 language and 16 reading
  examples per update, microbatch 8.
- Seven development/test groups of 256 examples each, plus 128 actual
  training-fit examples. Held-out groups separate new combinations, names,
  wording, answer words and composed changes.
- Full prompts and fact-bearing blocks are split-disjoint. All facts are in a
  completed 64-token block; the whole question and complete answer/EOS fit raw64.
  No examples were filtered by decoder success.
- Loss: 0.75 language CE + 0.25 full-answer CE + 0.05 reconstruction CE through
  the original frozen decoder. Reconstruction expansion is training-only.
- FP32 AdamW, weight decay 0.01, joint gradient clipping at 1; reader learning
  rate 3e-5 cosine to 3e-6 with 30 warmup updates, packing rate one tenth of that.

The feature cache contains 137,033 genuine full blocks, about 8.36 GiB, covering
the selected training segments and all 7,831 original development segments.

Each arm received exactly 1,872,347 language labels, 34,264 reading labels and
131,072 reconstruction labels. All Adam states reached step 512. Frozen packing
weights are exactly unchanged; trainable packing weights changed.

## Results

Complete answers require the correct value and EOS.

| Measure | Partial native parent | Frozen encoder | Trainable packing |
|---|---:|---:|---:|
| Full development token accuracy | 45.601133% | 46.602000% | 46.603452% |
| Full development CE | 2.726427284 | 2.672051341 | 2.671986526 |
| Familiar positive facts | 0/230 | 0/230 | 0/230 |
| New combinations | 0/230 | 0/230 | 0/230 |
| New names | 0/230 | 0/230 | 0/230 |
| New wording | 0/230 | 0/230 | 0/230 |
| New answer words | 0/230 | 0/230 | 0/230 |
| Composed transfer | 0/230 | 0/230 | 0/230 |
| Actual training-fit positive facts | 0/104 | 0/104 | 0/104 |
| Familiar positive quartets, all four correct | 0/51 | 0/51 | 0/51 |
| Historical notebook facts | 0/103 | 0/103 | 0/103 |
| Historical changed notebook facts | 0/32 | 0/32 | 0/32 |

All other held-out groups also have zero positive quartet success. The separate
reading development group has 0/230 positive answers for all arms.

Both trained arms correctly answer 152/156 missing cases across the six test
groups. They also incorrectly answer `unknown` for 228/230 familiar positive
cases and 98/104 training-fit positive cases. All trained reading outputs finish
with EOS. Relaxed full-value substring checks are also zero in every positive
group, so answer formatting alone does not explain the failure.

Four successful aggregate training-fit quartets contain only missing answers in
single-record units. They are not factual-reading successes. Blocked and
shuffled-code controls are saved alongside the main predictions.

Language improves by about one percentage point for both continuations.
Packing minus frozen is only +0.001452 percentage points, with a fixed-checkpoint
paired-document 95% interval of [-0.001675, 0.004556]. This does not show an added
language benefit from unfreezing packing. These intervals do not measure
training-seed uncertainty.

## Information retained in the codes

The packing arm has active gradients and a median relative code change of
0.930607%. The original frozen decoder recovers 24,963/24,964 tested fact-body
tokens and 895/896 complete fact bodies for all three arms.

Across full blocks it recovers 57,342/57,344 tokens for parent/frozen and
57,341/57,344 for packing. The additional packing-arm error is outside the
fact-body intervals. Existing errors were retained; this does not establish a
globally lossless codec.

All arms score 0/8 historical compressed colors. Historical missing notebook
answers stay at 21/25. Every historical score key was independently rederived
in the supplementary behavior/fact audit.

## Verification and artifacts

Four CPU tests passed: mixed-objective microbatch gradients, frozen-pack
isolation, exact serialized Adam resume and complete-answer/EOS scoring.

Zero-update GPU checks passed, including live/cache code and packing-gradient
parity, public/packed logits, future-code isolation and exact unchanged initial
weights. Maximum full-gradient difference between microbatches 4 and 8 was
1.02e-8. Heavy peak allocation/reservation was 3.485/3.814 GiB.

Two cached forward/backward body samples took 1.391 seconds at microbatch 4 and
0.625 seconds at microbatch 8. They exclude Adam, preparation, evaluation and
saving. Actual training was about 1.64 frozen/1.74 packing updates per second;
sequential order and warm caches prevent a causal speed comparison.

Artifacts:

- Code: `tools/*token_memory_64_pack*.py`
- Data: `dump/token-memory-64-pack-data-20261017`
- Run: `dump/token-memory-64-pack-pilot-20261017`
- Evaluation: the run's `evaluation/` directory contains all predictions,
  controls, teacher records, per-segment language metrics, reconstruction
  predictions, `independent-verification.json` and
  `behavior-and-fact-audit.json`.

## Interpretation

This recipe improved answer-format/EOS compliance, missing-answer handling and
language accuracy. It did not teach positive fact reading, including on the
sampled training examples. Updating the packing layers preserved tested fact
content but did not make it accessible to the language reader.

This bounded run cannot establish a universal capacity limit or reject all
joint encoder training: it uses one partial native parent and a narrow synthetic
task. Before scaling, establish positive single-fact reading without collapse to
`unknown`, then require held-out binding/value success and language preservation.
No subsequent training has been started.
