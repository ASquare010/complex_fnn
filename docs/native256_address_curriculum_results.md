# Selective word access works; entity-to-address binding does not

Both fresh arms completed exactly512updates and passed training, independent
results and same-template counterfactual verification. **The trained word decoder
can recover source words, including held answer IDs. The LM still does not reliably
select the record named by the question. This is not a useful overall quality win.**

Run: `dump/native256-address-curriculum-pilot-20261017`. All workers finished;
checkpoints are immutable. Do not restart or automatically extend either arm.
Native256, old models, original data/caches and all1769 protected files are unchanged.
Budget is23,209,216 stored READER parameters, encoder separate, as the user confirmed.

## What was implemented

The new `token_memory_64_selective_lm` package integrates the frozen trained
one-position decoder and query-conditioned address controller. Each old64-token
block still occupies ONE256-wide memory attention/KV position per layer. The
last layer's existing V holds the original FP32 code, so no extra retained code
bank, decoder features, old token IDs or expanded KV sequence is needed.

The controller computes64 temporary scalar addresses per code plus NULL. It
selects one position, decodes that word and mixes its distribution with the normal
language head. NULL selects generation. Address targets enter only the training
loss; forward/generation never receives gold positions. The readout adds compute
but does not expand global memory attention positions. Actual incremental prefill,
steps, raw64 rollover and storage alias were verified before training.

Reader parameters include3,981,312 frozen decoder and36,864 controller parameters;
19,227,904 parameters are trainable. Core:21 layers/width256/24619 total FFN rows.
Encoder9,426,432 parameters are separate. The core/controller start fresh seed47;
the codec decoder is pretrained. This is not a continuation of native8750 or the
completed packing specialists. See [integration checks/timings](native256_selective_lm_results.md).

## Matched bounded recipe

Generation-only versus selective-output arms use the same complete architecture,
address supervision, initial tensors, batches, labels and LR. Only use of the
selected-word OUTPUT route differs. Encoder/packing stays frozen in both.
This is not another frozen-versus-trainable-packing comparison; that experiment
is already [completed and verified](native256_balanced_pilot_results.md).

16 original language segments plus16 QA examples/update, micro8, FP32 AdamW.01,
clip1, warm30, LR3e-4 cosine to3e-5. Loss .75LMCE+.25answerCE+.25addressCE. Each arm
receives1,872,347 original LM labels and16,896 QA/address labels, with all Adam
states reaching512. Reading data is4096examples/two passes: two records, both
queries against both value-swapped memories, balanced128 single-token values,
source offsets0/8/16/24, both record orders and6.25% missing examples.128 held
answer IDs, names, wording and positions are separate evaluations. Held answer
IDs are held from QA supervision only, not codec pretraining or language replay.

These are two NEW reading-data passes and only8192 original language-segment
exposures. They are **not two original corpus epochs**, a final architecture
comparison or a matched all-raw reference. The low language accuracies below are
from deliberately fresh512-update models; do not compare them directly with a
completed48,762-update baseline or the earlier copied8750 pilots.

Independent audits checked all serialization/source-derived addresses, word
balance, disjoint blocks, language orders, cursors/LR and frozen isolation. Actual
mixed-objective CPU microbatch/serialized Adam tests passed. Full GPU objective
microbatch gradient error5.12e-9, unchanged initial weights/empty Adam. Actual first
update was independently audited before resuming. Observed training~1.7updates/s
per arm is sequential/confounded, not a controlled throughput improvement.

## Main free-answer results

Complete value+EOS, not teacher-forced token pieces:

| Group | Generation only | Selective output |
|---|---:|---:|
| Familiar values, new blocks | 0/512 | 244/512 |
|128 held answer IDs | 0/512 | 251/512 |
| New entity names | 0/512 | 267/512 |
| New question wording | 0/512 | 259/512 |
| Held source positions | 0/512 | 256/512 |
| Composed changes | 0/512 | 248/512 |
| Missing group: positive subset | 0/256 | 137/256 |
| Missing group: genuinely absent subset | 256/256 | 5/256 |
| Full original development token accuracy |18.324578% |18.034536% |
| Full development CE |5.242833122 |5.270522839 |

The control emits unknown on every positive test case. With128 answer classes,
even6.25% missing examples make unknown more frequent than each individual word;
positive-class balance alone does not remove this prior. The selective route
avoids that positive-answer collapse, but generally chooses one of the two source
values without matching the requested entity. About50% correctness is therefore
not evidence of successful two-record binding.

On familiar cases all512 first predictions equal the word at the selected code
address.264 errors are the OTHER record's value, and4 are other source words.
On held values all512 first predictions likewise match the selected source word;
257 choose the other record.255 first values are correct but only251 complete
answers have correct EOS. Thus rendering a selected source word works across
held answer IDs, while choosing the correct source record remains the bottleneck.

## The stronger same-template test

The training/main evaluation assigns different templates to the two query roles.
Main all-four counts (familiar9/128, held values15/128, new names19/128) could
therefore reward template/position strategies. A separate read-only control asks
BOTH entities with the SAME template, keeping equal lengths/phases, source codes,
counterfactual memories and target values unchanged. Both What and Give templates
are tested separately, and results are stratified by both source record orders.

| Source group, selective model | Same What template | Same Give template | All-four units |
|---|---:|---:|---:|
| Familiar |254/512 |253/512 |0/128 for EACH |
| Held answer IDs |255/512 |251/512 |0/128 for EACH |
| New names |253/512 |256/512 |0/128 for EACH |
| Held positions |252/512 |254/512 |0/128 for EACH |
| Composed source changes |251/512 |249/512 |0/128 for EACH |
| Actual positive training units |256/512 |257/512 |0/128 for EACH |

All other same-template groups also have zero all-four success. This remains zero
in BOTH record orders. On familiar What questions the model gives identical
answers for the two different names in all256 matched memories; Give changes
only1/256. Positive training units show the same failure. The generation control
has zero positives/all-four throughout. Independent verification rederived every
case/metric and replayed112 complete answers through CPU PUBLIC forward, separate
from the incremental GPU evaluator. This is strong evidence of unresolved query
binding under this recipe, not a universal capacity or convergence proof.

The same-template controls reuse the held source histories; they are an additional
query control, not a new independent block sample. Wordings are deliberately
replaced, so their wording/composed labels refer to the original SOURCE cohorts,
not success on unseen question templates in this control.

## Representation, controls and language cost

The frozen one-position decoder recovers all7680 source value tokens when given
their source positions. A separate CPU diagnostic independently derives entity
BPE positions and recovers all15360 entity tokens. Sampled64 encoder blocks replay
exactly when using the preparer's original contiguous128-block batch shape.
Changing GPU batch shapes produced FP32 rounding differences; the audit was fixed
to replay the actual batch shape, without altering codes. These are ORACLE codec
diagnostics, not LM retrieval. Both identity and value information are readable
at their known positions in these cases.

Blocking familiar fact codes preserves count/positions and yields0/64 positives
for the selective model; shuffling also gives0/64 against original targets. This
supports source-word dependence, but cannot substitute for entity binding.
Historical312 cases remain0/103 notebook facts,0/32 changed notebook facts and
0/8 compressed colors for both fresh arms. Full historical outputs are retained.
The main first128 training-fit cases are missing-heavy by construction; the
separate512 POSITIVE training-case control above avoids that misleading subset.

Selective minus control full-dev accuracy is-.2900417pp; fixed-checkpoint paired
document bootstrap95%CI[-.3246316,-.2535656]pp over5000 resamples. This does not
cover training-seed uncertainty. Per-source metrics are in `descriptive-analysis.json`.
There is a language cost in this run and no broad factual/reasoning quality win.

## Evaluation recovery and provenance

The original evaluator completed all generation-arm QA and full development,
then historical SINGLE inference exposed an ambiguous zero-element reshape for
M=0 in the frozen new model. No trained tensors or sources were edited. A sibling
`token_memory_64_selective_compat` overrides ONLY the empty-memory path with
ordinary no-memory attention; nonempty calls use the exact original method.
No new parameters, attention or KV positions are added. Two CPU checks passed.

The recovery evaluator preserves all13 completed files by hash and finishes
historical and the unstarted selective arm. A full-size independent audit loads
BOTH trained states: empty-history hidden outputs and ALL core gradients match
the unchanged native no-memory math exactly; cached log-probability max difference
2.86e-6. Main results, same-template results and recovery provenance all passed
independent verification. Sources/checkpoints and1769 protected originals remain
unchanged. All workers are finished; no additional optimizer update occurred.

## Decision

Keep the result as evidence that pretrained selective word decoding removes a
word-rendering obstacle at unchanged memory attention/KV count. **Do not adopt or
extend this pilot as a fact-retrieval solution.** The controller ignores queried
identity even on positive training units. Next test should address entity-anchor
selection explicitly, with identical question templates, before value offsets;
another candidate is structured canonical token codes in a new branch. See
[unimplemented hypotheses and constraints](token_memory_64_address_next_hypotheses.md).
Fresh two-original-epoch training, a genuine matched all-raw reference, broad
reasoning and end-to-end efficiency are still required for the full goal.
