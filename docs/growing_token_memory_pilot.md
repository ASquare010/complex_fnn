# Token output with a growing encoded prefix — 2026-10-13

Status: both matched arms completed and independently verified. No replacement
of the original baseline is recommended from this pilot.

## Completed pilot results

This implementation did not improve held-out fact recovery. Both arms fail the
paired factual-following criterion: neither correctly answers both the original
and changed value for any of the 204 answerable synthetic test cases.

| Free-generation measure | Fine-tuned baseline | Growing encoded prefix |
|---|---:|---:|
| Old synthetic facts, answerable, new names | 16/204 (7.84%) | 6/204 (2.94%) |
| Real SQuAD2 reading, answerable | 1/209 (0.48%) | 0/209 (0%) |
| All answerable questions | 17/413 (4.12%) | 6/413 (1.45%) |
| Explicit missing evidence, original test | 26/99 (26.26%) | 16/99 (16.16%) |
| Both original and changed fact correct | 0/204 | 0/204 |
| Output changed when only old fact changed | 4/204 | 24/204 |
| Changed-fact answers correct individually | 23/204 | 6/204 |
| Queried record removed: correct abstention | 55/204 | 41/204 |
| Wrong notebook: correct abstention | 62/256 | 80/256 |
| Old memory blocked: correct abstention | 62/256 | 0/256 |
| Proper end marker emitted | 512/512 | 210/512 |

Changing more outputs does not establish better reading: the growing arm's
extra changes do not track the new correct values. Wrong-notebook abstention
alone is likewise insufficient, especially when correct-evidence answers fail.
Some individual correct answers are compatible with guessing; paired factual
following is the stronger check.

The fresh familiar-name/wording diagnostic also gives 0/102 correctly followed
fact changes for both arms. Each scores 10/128 overall exact answers, including
missing cases. The baseline changes none of these 102 outputs when the fact
changes; the growing arm changes eight. This diagnostic is exploratory and
separate from the preregistered new-name test.

On the sampled 64 training answers, baseline exact scores are 9/35 synthetic
and 29/29 real; growing scores are 8/35 synthetic and 0/29 real. The baseline
has memorized sampled real training answers without learning general reading;
the growing arm also has poor free-running training fit. Teacher-forced loss
improvements must not be equated with autonomous factual competence.

Example `gv-test-3`: the original answer is `piano`, changed to `cello`. The
baseline says `piano` in both conditions. The growing model emits malformed
repetitions and end-marker fragments in both. All generated texts are retained
in `results.json`, including failures.

Training plus periodic development evaluation and checkpoint saves took
244.562 seconds for baseline and 109.422 seconds for growing. Initial evaluation,
preparation and final tests are excluded. Generating the 512 test answers took
5.844 versus 26.094 seconds, including runtime prefix encoding for growing and
using prepared old-memory codes for both. Output lengths and stopping differ;
these timings do not establish a quality-adjusted speed win.

This is one matched fine-tuning budget from a backbone already trained on the
baseline input interface. It does not establish what independently pretrained
models or longer schedules could achieve. The synthetic design is deliberately
limited: four relations and controlled wording; held-out-wording status is tied
to relation parity, so that subset cannot isolate a pure wording effect.
The result supports retaining the original baseline, not a general impossibility
claim about token feedback through a frozen encoder.

## Independent verification

Trainer session67322 completed with exit0, including exactly2,048 new optimizer
updates across the two arms. Independent verifier session34623 completed with
exit0 and zero optimizer updates. It checked all39,347 cached target-query
layouts against the actual preceding token prefixes, the source archive,
protected-file hashes, exact23,209,216 parameter counts, all Adam1024 counters,
and identical final/last model tensors. Both arms reproduced64 saved answers
and128 saved intervention answers exactly. Mutating target labels/references
did not change generation; changing fully blocked memory contents also had no
effect. Eight growing answers match a full-computation sampler, with cached/full
logit maxabs1.72e-5. Eight baseline blocked answers also match the independently
implemented fixed-rotary-offset variant described below.

There is a codec limitation worth separating from failed memory learning:
an additional reconstruction audit of446 genuine cached prefixes at available
lengths1..64 (11and12 were absent) recovered15,950/15,968 tokens and431/446 complete
prefixes. Fifteen prefixes had18 token errors, all at lengths1..8. Only2/8sampled
single-token prefixes reconstructed correctly. All failures are retained in
`verification.json`. Thus valid
encoder output does not guarantee perfect reconstruction at every short prefix,
even though all checked full old-fact records were recovered exactly. The
predictor does not use the reconstruction decoder, so these errors are a
diagnostic of the representation and its decoder, not proof that these specific
errors caused every failed next-token answer.

The paired case-bootstrap95% interval for growing minus baseline answerable
synthetic exact accuracy is[-8.82,-1.47] percentage points (point estimate-4.90).
For real reading it is[-1.44,0.00] points. These intervals concern this test
sample with one trained initialization, not variation across training seeds.

The original external Python environment became unavailable after training;
the repository environment was CPU-only. Verification used the already existing
WaveMixNN Python3.12/CUDA runtime (torch2.9.1+cu130), with repository packages
available for tokenization. No package installation or environment mutation was
performed. A first verifier run completed numerical checks but failed while
archiving its relative source path under `runpy`; resolving the path fixed this,
and the complete independent verification was rerun successfully. Details are
retained in `verification-attempts.json`. No training updates were repeated.

The user explicitly authorized this token-output design, a small matched pilot,
and model cleanup. This supersedes the older prohibition on deletion for this
specific cleanup and relaxes the earlier direct-vector-output restriction.
No full-v4 production run is resumed.

## What changes

Both arms start from the exact trained token-memory checkpoint at 48,762 updates
(`e17b9a8469b5a8c996cc64ff0677eb1c8f52410af52c1c179fd317263b4d9e2d`). Both have
23,209,216 trainable parameters and identical initial tensors. The original
encoder remains frozen and uses FP32. Its matching reconstruction decoder is
only a preparation-time diagnostic, not the generation head.

The baseline retains its eight vectors per completed 512-token history block
plus raw recent token inputs. The new arm reads completed 64-token vectors and
one active prefix vector, predicts one ordinary vocabulary token, appends that
token to a private encoder buffer, and re-encodes the committed prefix. The
active vector is replaced until 64 tokens are complete, when it is appended to
memory. The buffer is never supplied as raw token IDs to the predicting model.
The question is part of the same recent stream, so its full 64-token chunks
also become completed memory. A constant block-start query represents length
zero. A fixed sinusoidal feature conveys lengths 0..63; it adds no parameters.
The tied embedding remains the trainable vocabulary output matrix.

This is not the retired experiment that predicted growing code vectors and
fed them back unchanged. No new-code MSE or parallel codec output is used here.

Training shares completed history across independent prefix queries. Each
query can see only its own prefix state and causally preceding completed
blocks; prefix snapshots never attend to one another. Generation caches only
completed-memory computation and rebuilds that cache upon a block rollover.
The frozen encoder remains bidirectional, so each active prefix is encoded
afresh during free generation.

## Data and matched budget

- Train: 2,048 real SQuAD2 records plus 2,048 controlled notebook records.
- Development: 128 real plus 128 synthetic records. Test: 256 plus 256.
- Real records retain the previously verified article-disjoint split and
  literal-v4-overlap exclusions. They are a subset of an already explored test
  source, not a new hidden benchmark. All valid answer references are retained.
- Relevant old source is supplied as exactly 512 tokens (eight codec vectors),
  using neutral space fill after the source. No answer-span crop is used.
- Synthetic records contain three named facts plus five neutral blocks. Four
  relations are color, animal, city and instrument, with eight values each.
  Names are disjoint across splits. Half of dev/test questions use held-out
  wording. One fifth have no matching named fact and target `Not stated.`.
- The preset synthetic question lengths include 61, 63 and 125 tokens, with
  ordinary space padding, to exercise rollover. These offsets are tied to the
  relation index in this pilot; they are not an independent randomized factor.
- Eight epochs, batch32: 1,024 AdamW updates and 265,296 answer-token labels
  per arm. Same batches, initialization and objectives. Baseline microbatch8,
  growing microbatch32; floating-point grouping differs.
- Objective: 0.9 mean answer-body CE + 0.1 mean literal `[END]` CE. Prompt
  tokens are inputs only. AdamW starts fresh for each arm: warmup30, learning
  rate 1e-4 cosine to 1e-5, weight decay .01, gradient norm cap1, FP32.
- All training input prefixes are pre-encoded and reused. The 37,823 unique
  codes, input tokens and layouts occupy 56,118,377 bytes. Cache preparation
  took 28.141 seconds; this cost is separate from training time.

The frozen decoder recovered all 12,530 dev and 25,533 test synthetic
fact-record tokens checked, including changed-fact records. No failing cases
were discarded. Re-encoding 128 selected inputs individually agreed with the
batched bank within 1.24e-5 maximum absolute FP32 error.

## Evaluation and validity checks

Greedy free generation has a 32-token cap and stops on ordinary `[END]`; all
reference answers fit. Report exact normalized answers/F1 separately for real,
synthetic, answerable and missing cases. Report generated text and EOS, not
teacher-forced loss alone. The synthetic counterfactual controls preserve the
entire question and the eight old-memory slots:

1. Correct old facts and original answer.
2. Changed queried value and corresponding new answer. Require both original
   and changed answers to be correct to establish factual following.
3. Removed queried record, with distractors retained; answer `Not stated.`.
4. Wrong old notebook drawn from disjoint training names; answer `Not stated.`.
5. Old memory blocked by attention mask; answer `Not stated.`. Growing query
   positions are preserved. The existing baseline forward offsets recent-token
   rotary positions by the *valid* memory count, so blocking shifts that offset
   by eight. Relative recent-token distances remain identical. An independent
   fixed-offset check is included; the original saved protocol's statement that
   all blocked positions are preserved is too broad. Content changes inside
   fully blocked memory must have no effect. Changed/removed/wrong controls keep
   both slot counts and positions unchanged.

An additional exploratory evaluation uses 128 fresh notebook histories with
familiar training names and question templates. These input pairs are excluded
from training. It was specified during the matched run, before the growing
arm's final results, to separate familiar-vocabulary factual following from
transfer to new names. It adds no training and is reported separately from the
original held-out test.

Seven CPU tests cover identical state/parameter layouts, packed versus separate
query predictions and gradients, future/other-query exclusion, blocked-memory
invariance, cached versus full computation, prefix shifts/rollover, live feedback
from committed tokens, and optimizer replay (some tests cover multiple items).
The full-size GPU preflight made zero optimizer updates and verified unchanged
initial weights. Cached/full and separately truncated logits agree within
4.77e-6. Peak allocated memory including moment-sized buffers was 907,058,688
bytes for the baseline and 582,460,928 for growing. These are checked batches,
not universal worst-case memory or end-to-end throughput claims.

Independent checkpoint, source, frozen-encoder and free-generation verification
passed as reported above. Even a positive result on the synthetic tasks alone
would not establish general natural-language reading or a better chatbot.

## Cleanup and retained evidence

Before this experiment, seven `last.pt` optimizer snapshots from independently
verified completed full-answer, clean-code and conditional-head pilots were
removed, freeing 2,051,418,249 bytes. Each snapshot's model tensors were checked
against its retained final `predictor.pt` before deletion. Final weights,
reports, source archives, all datasets, original codec/baseline and paused
production runs remain. The deleted optimizer snapshots can no longer be
resumed, and historical verifiers requiring those snapshots cannot be rerun
unchanged. Their earlier verification evidence remains intact.

Audit: `dump/growing-token-cleanup-20261013.json`.
Pilot artifacts: `dump/growing-token-pilot-20261013/`.
Implementation: `src/models/growing_token_memory/transformer.py`,
`tools/growing_token_pilot_data.py`, `tools/growing_token_evaluation.py`,
`tools/prepare_growing_token_pilot.py`, `tools/train_growing_token_pilot.py`.
