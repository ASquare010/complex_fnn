# Improving raw64 + compressed-memory factual reading

Latest verified18October2026: lexical64 at1,000 improves language to44.136780%
versus plain64's35.105476% and native256's33.170807% at matching original updates.
However, positive free answers21/285, changed-query both-correct0/64 and proper
notebook all-four0/64 remain weak. Independent full/source/public and notebook
verifiers passed. The local nine-token key contains no owner-name token at any
of570 discriminating source decisions. A separate untrained nine-slot/33-word
pooled-key model preserves exact fresh weights/RNG/23M/cache and passes five CPU
checks; GPU benchmarking precedes any training. Raw control is still training
toward1,000. No full-budget/adoption claim; goalACTIVE/UNMET. See the
[complete intermediate report](native256_lexical_1000_results.md).

Latest state18October2026: full-size reference and optimized lexical64 GPU/math/
cache/gradient gates and independent audits passed, with all protected old
artifacts unchanged. No lexical quality result exists yet. A fresh matched
original-corpus comparison is being prepared:23,209,216 trainable reader weights
per arm, frozen encoder separate,2 original passes/48,762 updates, unchanged
assistant CE and a genuine conventional1:1 reference. The zero-update v1 data/
initial/microbatch gates passed; v2 retains the same recipe and removes redundant
per-parameter CUDA synchronization after exact optimizer/resume/nonfinite tests.
v2 data/GPU gates and both actual first-update audits passed; compressed completed
and passed its1,000-update audit(7,416,257 labels). Raw is continuing on its same
schedule through250 and then1,000. The preliminary full-dev stage is44.136780%/
CE3.065599 versus old plain64's35.105476%/CE3.511001 at1,000 original updates.
This is an early architecture/representation gain; free facts, raw1,000 and
independent final results remain pending. Native256 and completed
diagnostics remain untouched. An evaluation-only original-notebook four-way
control now has512 audited full/completion cases+128blocked; prospective input
strata separate raw-query/name visibility and rollover. All conditional prefixes
are exactly fixed name/relation tokens, identical across swapped memories. See
[current run status](native256_lexical_full_status.md),
[GPU evidence and mixed efficiency](native256_lexical_gpu_results.md) and
[full comparison protocol](native256_lexical_full_protocol.md).

This variant deliberately adds exact literal IDs in existing FP32 low bits; it
does not demonstrate that the old semantic encoder is lossless. Main global
attention/KV still has one old position per64 words, while local lexical reading
adds computation. Batch4 forward improves and single-token decode remains slower
than conventional1:1 in the measured setup. Final broad quality and end-to-end
efficiency under the full training budget remain unverified; goalACTIVE/UNMET.

Latest continuation18October2026: fresh512 direct-NAME-gradient isolation is
complete and independently verified. Language recovers10.047930->16.481713%,
but remains below no-NAME19.656629%. Familiar exact407/512/all-four68/128 retains
some binding; new names119/512, positions188/512 and composed5/512 remain poor.
All2005 protected earlier files unchanged. No adoption or extension; see the
[isolation report](native256_name_isolation_results.md). A separate untrained
23M all-trainable lexical LM/cache now passes8 component and7 CPU integration
tests, using factored shared-embedding queries/exact-ID residuals. Full GPU/math/
cache/speed gates precede training; see [current implementation](native256_lexical_lm_results.md).
The full two-original-epoch/broad quality goal remains ACTIVE/UNMET.

Latest 18 October 2026: a separate fresh512 name-before-value curriculum now
achieves familiar exact500/512 and120/128 all-four binding units, versus266/512
and4/128 for its matched control. QA-held values484/512 and new wording498/512
also improve. However, new names129/512, new positions211/512 and combined changes
0/512 remain inadequate. Full original-dev accuracy falls from19.656629% to
10.047930%; generator-only10.061106% rules out copying alone as the explanation.
Both arms and independent training/results/head checks completed. All1901
protected old files remain unchanged. This is a narrow binding signal with a
large language cost, not adoption or two original corpus epochs. See the
[complete name-anchor report](native256_name_anchor_results.md). The overall
goal remains active and unmet; do not automatically extend completed pilots.

Separately, a new FP32 word-bit residual component carries exact lexical IDs in
the low mantissa bits of existing code slots. CPU/CUDA and independent audits
passed, with zero added KV bytes; decoder/native numerical samples changed no
predictions. This is deliberately added lexical information, not evidence that
the old encoder is lossless. There is no word-bit LM integration or training yet;
see [storage gate and limitations](native256_wordbits_results.md).

The preceding selective-address512 comparison rendered held words but failed
same-template counterfactual binding, even on positive training units. Its full
dev accuracy18.034536% versus18.324578% is recorded separately in the
[selective-address report](native256_address_curriculum_results.md).

Active objective, 16 October 2026: improve factual retrieval and exceed the
roughly61% token-accuracy reference while retaining the 64-token raw window,
one attention position per completed64-token code, the23,209,216 stored reader
parameter budget and efficient inference (encoder separate, user-confirmed).
Pretrained frozen readout weights count inside that reader budget. The growing-vector experiments remain
retired. No result below establishes that the full objective has been achieved.

## Reference and success criteria

The verified61% checkpoint is **raw512 plus older compressed memory**, not a
memory-free conventional transformer. Its test token accuracy is60.9990%; plain64
is53.5088%. The retained raw-only model was trained on a different earlier
recipe. A claim of beating a traditional raw-only model needs a matched raw-only
comparison; the existing numbers do not supply one.

Completion requires evidence for all of the following, not just a lower pilot
loss or a few successful examples:

1. Full language evaluation exceeding the specified reference under a disclosed,
   comparable training budget, including the eventual fresh-seed/two-epoch
   comparison. Extra fine-tuning must be counted separately.
2. Reliable free-generated exact factual answers on independently held-out data:
   unseen entities, values and bindings, wording, multiple records, distractors,
   corrections, missing information and positions across multiple64-token
   boundaries. Paired changed facts must change the correct answer. Blocked and
   wrong-memory controls must distinguish useful reading from guessing.
3. Preserve the main compression advantage. The user's updated goal permits
   slightly more compute or a little more VRAM. One slot per code remains the
   current design; modest reader overhead is now permitted if measured and
   justified. No full decoder expansion, hidden raw-history store or external
   answer lookup. Keep the frozen codec and ordinary token output.
4. Measured training and inference costs, including encoding/rollover and
   realistic generated lengths. Same attention dimensions alone are not a
   wall-clock efficiency result.

"Perfect retrieval" remains a goal. Zero observed errors on a finite suite would
not prove error-free retrieval of every possible fact. Do not replace the broad
goal with a small synthetic task and call it complete. The previously reused
test set is research evidence, not a new blind benchmark.

## Prior evidence rechecked

The [grounding diagnosis](v4_grounding_diagnosis.md) recovered all2,202 sampled
fact-body tokens with the frozen decoder while the LM failed factual questions.
The codec did make32 errors among4,096 complete chunk tokens, all EOS/turn
markers outside those fact bodies. This supports an interface/learning problem
in those examples; it does not make the codec globally lossless.

Only about0.12% of scored training tokens were estimated to directly teach the
variable location components in the notebook task. Most memory-source labels
were easier wording, acknowledgment and name tokens. That estimate is a token
share, not a measured gradient share. The same original model also struggled
with multi-record binding when the records were raw text.

The older [mixed retrieval experiment](mixed_memory_chat_results.md) reached
high controlled retrieval with explicit retrieval supervision, but used another
checkpoint/corpus, extra training, a restricted vocabulary and simpler outputs.
It is evidence that a one-position code can support learned retrieval in some
settings, not proof of general factual competence or of this raw64 model.

## New full-development diagnostic

Artifacts: `dump/token-memory-64-position-diagnosis-20261016`.
All7,831 development segments and1,859,388 assistant targets were scored. No
optimizer was constructed. Baseline checkpoints, encoder, cache and model
sources stayed unchanged. The new probe retains the exact baseline parameter
layout; only its explicitly selected position rule or memory-key access changes.

| Same saved plain64 weights unless noted | Dev CE | Token accuracy |
|---|---:|---:|
| Plain64, original positions | 2.214284 | 53.5046% |
| Plain64, all memory keys blocked; original counts/positions retained | 2.720478 | 47.5673% |
| Plain64, original-token-coordinate positions | 2.457806 | 49.0471% |
| Original512 checkpoint, its original positions | 1.804710 | 61.0132% |

Memory access improves plain64 accuracy by5.9373 percentage points in this
intervention. The model therefore does not simply ignore its vectors. This
does not mean it retrieves facts correctly. Document-bootstrap95% intervals
for the accuracy difference are[5.8198,6.0595] points for memory-on minus blocked,
and[-4.6362,-4.2832] for token-coordinate minus original positions. These are
development-sampling intervals for fixed weights, not uncertainty across seeds.

The position probe changes memory code `j` from position `j` to `64*j+63`, and
raw query positions from `visible_codes+offset` to `64*visible_codes+offset`.
The block-end anchor makes the first query one position after the last completed
code. It supplies outer block distances; it does not expose64 internal token
positions to the LLM. Compressed-index positions are a valid modeling choice,
not an established implementation bug. Replacing them on trained weights is an
out-of-distribution intervention, so this result alone cannot reject training
with the alternative convention.

## The 64-token boundary is a weakness, but not the entire gap

| Position inside each64-token block | Plain64 accuracy | Original512 accuracy | Contribution to overall accuracy gap |
|---|---:|---:|---:|
| First prediction | 22.0887% | 56.5431% | 0.5350 points |
| Offsets1–7 | 44.0284% | 59.1284% | 1.6418 points |
| Offsets8–31 | 53.7230% | 61.0975% | 2.7617 points |
| Offsets32–63 | 56.3702% | 61.4976% | 2.5701 points |

Both models are measured on the same targets in each row. Plain64 has only
memory and BOS at the first prediction after committing a block. Blocking its
memory reduces first-prediction accuracy further to2.5528%.

The complete accuracy gap is7.5086 points. Restoring only the first prediction
to original512 accuracy would recover just0.5350 points. A boundary-only repair
therefore cannot explain or close the whole measured gap. This is a decomposition
of observed errors, not proof that position is their cause.

## Factual generation and cost

Replayed all312 historical cases, including correct/blocked notebook prompts,
32 changed facts and eight colors under recent/compressed/blocked conditions.
Both existing models'624 outputs exactly reproduced the previous evaluation.

The position-switched plain64 still recovered0/103 notebook facts and0/32
changed facts exactly. Compressed-color score was2/8 versus1/8 for plain64, while
recent-color score stayed0/8 and correct missing-information responses fell from
17/25 to0/25. This is not evidence of a useful retrieval improvement.

Ten alternating warmed forward measurements on the identical cached16-example
batch gave median29.2981ms original versus29.4102ms token-coordinate positions
(about+0.38%, not an established speed difference). Attention shape was
`[16,8,369,369]` in both arms. There is still one position per code, no additional
KV entries and no additional trainable parameters. This benchmark excludes the
vocabulary head, encoding, optimizer, disk access and generation; it is not an
end-to-end throughput claim.

The diagnostic initially stopped in its generation stage because the orchestration
code treated a `(predictions, parity_error)` return as a prediction list. The
generation stage was repaired and rerun without repeating full-development
evaluation. Separate source archives preserve both stages. Token-coordinate
generation uses its own public forward, rather than the old-position reference
path. An independent CPU verifier recomputed corpus/phase counts, metrics and
generation scores from saved arrays and outputs and checked protected hashes.

## Fresh position-only pilot

To distinguish training mismatch from a weak position convention, a bounded
pilot starts fresh at seed47 and runs the original first1,000 optimizer batches.
The original48,762-update/two-epoch learning-rate schedule is retained; it is not
compressed into1,000 steps. This is a pilot, not two completed epochs.

It reuses the unchanged plain64 trainer, frozen encoder cache, original batch
ordering, targets, effective batch32, microbatch16, FP32 and AdamW. All initial
tensors and the23,209,216 parameter count match plain64 exactly. The process-local
adapter selects token-end positions and a distinct checkpoint format. Existing
model source and run directories are unchanged. Six CPU checks passed, including
all-parameter gradient parity, future/padding/blocked isolation and actual tiny
two-epoch optimizer/model/RNG/cursor resume.

Run: `dump/token-memory-64-token-positions-pilot-20261016`.
Command used (stops after1,000 new updates):

```powershell
& .\.venv-cuda64\Scripts\python.exe .\tools\train_token_memory_64_position_pilot.py `
  --name token-memory-64-token-positions-pilot-20261016 `
  --data .\dump\data\unified-chat-v4 `
  --cache-dir .\dump\token-memory-64-cache-v4 `
  --epochs 2 --batch-size 32 --microbatch 16 `
  --stop-after 1000 --save-every 250 --eval-every 1000
```

The prespecified comparison is original plain64 at step1,000:
full-dev CE3.511000690, accuracy35.1055%. A small early loss change would justify
further investigation, not a claim of61% language quality or perfect facts.
The pilot's completion and outcome are recorded below when verified.

## Research implications

Position is a cheap, controlled ablation. The stronger current hypothesis is
that dense compressed content needs more effective reading supervision while
the model preserves ordinary language. Next experiments should isolate position
from supervision, use paired counterfactual facts, include rollovers during
answers, and measure language retention rather than optimize only an easy
retrieval score. Keep architecture and inference cost fixed when testing that
hypothesis.

[RoFormer](https://arxiv.org/abs/2104.09864) describes how rotary positions affect
relative attention; it does not claim to unpack compressed token content.
[ICAE](https://arxiv.org/abs/2307.06945) combines reconstruction/language-model
pretraining with instruction training for compact-memory use. Its reported4x
compression on a much larger pretrained model is not evidence that our frozen
64:1 codec and23M reader will achieve the same outcome. These papers motivate
separating positional geometry from learning to use the representation; the
local experiments must determine what works here.

## Verified pilot outcome

The bounded pilot stopped cleanly at update1,000. Its full-development evaluation
was run separately because the unchanged trainer saves/pauses before periodic
validation when the stop limit is reached. It did not complete two epochs.

| Matched first1,000 updates, fresh seed47 | Dev CE | Token accuracy |
|---|---:|---:|
| Original plain64, recorded baseline validation | 3.511000690 | 35.1055% |
| Token-coordinate positions | 3.507397358 | 35.2248% |

The difference is-0.003603332 CE and+0.11929 percentage points (2,218 additional
correct tokens on1,859,388 targets). CE improved on12/17 sources. This is a small
early improvement on one seed, not proof of better final training or a completed
solution. All1,000 batches,32,000 sample exposures,7,416,257 assistant labels,
learning rates and cursor positions match the original baseline. All Adam counters
are1,000. Current and archived training sources, checkpoint identity and frozen
encoder identity passed independent checks.

The candidate still scored0/103 exact notebook answers,0/32 changed facts and
0/8 compressed colors. Original step1,000 weights were not retained, so a matched
factual-generation comparison at that step is unavailable. Do not compare its
factual outputs to the completed baseline as if the training budgets matched.

Evidence is in
`dump/token-memory-64-token-positions-pilot-20261016/evaluation-step1000`:
`report.json`, per-segment arrays, predictions, verification and an independently
recomputed metric check. The verifier's first attempt needed Windows archive
path separators normalized; it was fixed and rerun without further training.

## Original-weights control

Loaded all weights from the original completed512+memory model into the unchanged
plain64 inference rule. This is an inference-only weight/control experiment,
not a newly trained model or a curriculum comparison. It retains23,209,216
parameters and one position per memory code.

Full-dev CE was2.383691481 and accuracy50.9052%, below the dedicated plain64
checkpoint's53.5046%. First post-rollover prediction accuracy was5.8642%, versus
22.0887% for dedicated plain64 training. It recovered0/103 notebook facts and
0/32 changed facts. Merely using the stronger512-context weights under64-context
inference therefore did not solve the problem. This does not rule out a properly
trained context-length curriculum.

Evidence: `dump/token-memory-64-original-weights-control-20261016`. Checkpoint,
encoder, source and cache hashes remained unchanged; no optimizer was used.
All saved aggregate metrics were independently recomputed from per-segment sums.

## Next experiment and current completion audit

The broad objective remains **active and unmet**: no result exceeds the language
reference with reliable factual retrieval. Position-only training produced a
small early signal; it did not establish factual reading. No full run was started.
The1,000-update pilot remains intentionally stopped and recoverable.

Next, test a training-only change that increases the loss contribution of factual
answer content using the existing corpus's recorded fact/answer spans. Keep the
same inputs, frozen codes, attention positions and parameter count. Audit the
weight masks independently, keep the language loss, and compare against the
same-step position pilot with the same sample order. Weighting can use training
labels, but must never feed answer content or future facts into model inputs.
This tests underweighted reading supervision without adding an inference reader
or more KV entries. It is a proposed next experiment, not a result. Broader
held-out names, values, wording and factual tasks remain necessary before any
claim of general or perfect retrieval; better weighted loss on the original
narrow notebook task alone would not satisfy the goal.

## Answer-content weighting experiment (16 October 2026)

The exact immutable-corpus audit found214,412 variable location BPE labels among
180,678,552 assistant labels per epoch:0.1186704%. These are the color/container/
room words in31,411 answerable training memory dialogues. The7,859 missing-answer
dialogues retain ordinary loss. All selected positions were checked against actual
token IDs, assistant masks, stored answer spans and earlier fact records. The
development selection has1,966 labels in285 answerable dialogues. Metadata selects
loss positions only; no gold record IDs, values, answer spans or weights enter
model attention or generation. Audit: dump/token-memory-64-answer-audit-20261016.

Prespecified intervention: w=64 for those positive answer-value labels, w=1 for
all other assistant labels, L=sum(w*CE)/sum(w) over each original effective batch.
Unweighted CE remains the reported language metric. This differs from adding more
data or training steps. With the original seed47 order, the mean per-batch weighted
critical share is6.9665% across epoch1. This is a loss coefficient, not a measured
gradient share. Missing-answer behavior must be measured for possible regression.

The adapter tools/train_token_memory_64_answer_weight.py leaves baseline source
files unchanged and uses the identical token_end inference model from the position
pilot. Same23,209,216 parameters, one position/code, original masks/encoder/cache,
FP32/effective32/micro16/fresh47 and48,762-step LR schedule. A bounded1,000-update
comparison uses exactly the position pilot's first32,000 examples and7,416,257
assistant labels;8,531 of those labels receive increased weight. It is not a
completed two-epoch result and cannot establish the broad goal by itself.

Four CPU checks passed: actual answer offset/label validation and corruption,
weighted microbatch versus full-batch all-parameter gradients/Adam, unit weights
versus ordinary CE, and real tiny two-epoch serialized optimizer/cursor resume.
Fullsize zero-update GPU preflight verified exact fresh47 state, all gradients,
same parameters and no extra KV entries. Six alternating warmed samples measured
ordinary/weighted forward-backward medians0.322882/0.330329seconds (+2.31%), including
packing/loss/clipping, excluding Adam/encoder/evaluation/save. Heavy micro16 with
Adam-sized scratch used3.964GiB allocated/4.266GiB reserved. Inference has no weighting
branch. Evidence: dump/token-memory-64-answer-weight-preflight-20261016.json.

Use a distinct output and checkpoint format token-memory-64-answer-weight-v1;
never resume it through plain64 or position-only training entrypoints. Evaluate
full ordinary dev, variable-answer labels, exact free generation, changed facts,
and missing/blocked controls against the same-step position checkpoint. Preserve
all original reference weights and data. No success is claimed before evaluation.

## Verified answer-weighting outcome

The fresh pilot completed its bounded1,000 updates and stopped intentionally.
All1,000 original batches,7,416,257 assistant labels, learning rates and cursor
positions match the position-only control. Every Adam counter is1,000. The actual
saved step0 tensors match fresh47 with the same single-thread CPU/runtime setup;
Adam was empty. An initial verifier omitted those matched runtime settings and
was corrected without changing/restarting training. This is an early pilot, not
a completed two-epoch comparison.

| Matched fresh1,000 updates | Ordinary CE | Answer-value weight64 |
|---|---:|---:|
| Full-dev CE,1,859,388 labels | 3.507397358 | 3.524092590 |
| Full-dev token accuracy | 35.2248% | 34.9563% |
| Non-memory-source accuracy | 34.9243% | 34.6560% |
| Critical BPE-piece CE | 1.037994 | 0.902227 |
| Critical BPE-piece accuracy | 1221/1966(62.1058%) | 1226/1966(62.3601%) |
| First critical token accuracy | 29/285(10.1754%) | 29/285(10.1754%) |
| Exact answerable dev memory answers | 0/285 | 0/285 |
| Missing-information exact answers | 9/63 | 0/63 |
| Changed-value exact / correct original+changed pair | 0/64 /0/64 | 0/64 /0/64 |
| Changed-question exact / correct original+changed pair | 0/64 /0/64 | 0/64 /0/64 |
| Historical notebook exact / changed exact | 0/103 /0/32 | 0/103 /0/32 |

The language difference is-0.26842 percentage points. A5,000-draw paired
development-document bootstrap gives95%CI[-0.31635,-0.21891]pp. This measures
document-sampling uncertainty for these fixed checkpoints, not seed uncertainty
or final-training performance. Critical accuracy is teacher forced: it includes
predictable subword continuations after correct earlier pieces. Its62% cannot
be interpreted as62% recovered facts. The first informative-token check remained
29/285 even when the correct preceding answer/name text was supplied.

Actual free generation commonly ignored the requested entity and repeated a
training-name/value template. The weighted model output "Amina keeps the notebook
in the purple cabinet in the kitchen." on195/285 answerable dev cases. The ordinary
control also repeated defaults (its most common output109/285). Changing the fact
changed32/64 ordinary versus23/64 weighted outputs; changing the query changed
17/64 versus11/64. All paired exact scores remained zero. Output change is not
evidence of factual correctness. Neither arm recovered a correctly bound full
value even with relaxed value/name scoring. Blocking fact chunks preserved
positions and all later query content; neither arm produced EOS within64 output
tokens on those64 controls. This shows dependence on memory presence, not useful
fact recovery or reliable abstention.

The original frozen codec's reconstruction decoder recovered ALL24,793 fact-body
tokens in348 dev prompts and ALL4,553 fact-body tokens in64 changed-value prompts;
all412 fact bodies were exact. The decoder was diagnostic only and never entered
the LLM's input/training/inference path. This rules out missing fact information
in those specific codes, not codec errors on every possible text.

This intervention is **not adopted** as an improvement. Positive-only weighting
reduced factual-word CE while hurting early language/missing-answer performance
and failing to improve real retrieval. It does not prove every weighting choice
or a later training stage would fail. More of this same pilot is not automatically
justified by lower critical CE.

The next hypothesis to test is explicit question-to-fact binding: matched prompts
with changed fact values and targets, plus missing-information supervision and a
language-preservation component. The training signal must distinguish reading
the supplied value from outputting a frequent template. Keep one code per block,
the parameter budget and ordinary inference computation unchanged. Evaluate
held-out entities, wordings, relations and counterfactual pairs, then full matched
training if a reliable reading signal appears. A narrow task win still would not
meet the broad language/factual-retrieval goal.

Evidence: dump/token-memory-64-answer-weight-pilot-20261016/evaluation-step1000
contains reports, both arms'540 corpus-case outputs, historical outputs, per-segment
metrics, codec diagnostics, source archive, output-behavior analysis, and independent
verification. Independent text-derived train/dev mask audit passed. Independent
saved-metric/exact-answer recomputation and all124 protected baseline/source/data/
cache hashes passed. Training and evaluation processes exited successfully; no
worker from this experiment remains. The objective remains active and unachieved.

## Paired binding diagnostic: prespecified protocol

Next experiment uses two **copies of the trained plain64 step48,762 checkpoint**,
with fresh Adam and the original legacy position convention. This is additional
diagnostic continuation to test learnability, not a fresh initialization, a fair
two-epoch result, or a replacement for the eventual full-budget comparison.
The original checkpoint/source/cache stays untouched. Both arms have the same
23,209,216 trainable parameters and identical unmodified plain64 inference.

Prepared2,048 training pairs,128 development pairs and256 held-out test pairs.
Each pair has the same question, raw prefix, coordinate positions and preceding
answer prefix at its first differing answer token. Only strictly older fact text
changes: a value changes, or the queried entity is replaced so the answer becomes
unknown. Train/dev/test entity names and question wording are disjoint. The three
trained relations are color, notebook location and access word. Test also has an
untrained preferred-tool relation; this held-out relation is tested with value-only
answers, so it does not establish full-sentence transfer. Train/dev balance both
full-sentence and value-only answers.20% of pairs contain a missing side, giving
about10% missing examples. This is structured synthetic reading, not broad real QA.

Fact blocks start at0..3, raw answer start phases cover all64 positions in train,
and578/4096 training answers cross a64-token rollover. Independent audit verifies
every cache block equals its true earlier token block, pair differences are only
in fact text, labels/targets are ordinary token IDs, and split identities are
disjoint. The frozen original decoder reconstructs all192,658/11,970/22,728 fact
tokens for train/dev/test. Reconstructed features never enter reader training or
inference.18,357 unique blocks are cached in FP32 in
dump/token-memory-64-binding-data-20261016 (manifest and independent verification).

Each update uses16 original language segments and8 adjacent reading pairs. Both
arms receive identical examples/order, two passes over the reading data,512 updates,
FP32, micro8 for language, all16 reading examples in one small packed forward,
AdamW(.01), clip1, warm30,3e-5 cosine to3e-6. This is NOT two new epochs over the
entire original language corpus. Objective ordinary=.9*languageCE+.1*readingCE.
The interaction arm adds.05*mean softplus(1-C), where
C=[logit_A(a)-logit_A(b)]+[logit_B(b)-logit_B(a)] at the first differing target token.
A/B differ only in old memory at that decision and a/b are their correct tokens.
Shared vocabulary preferences cancel in C. The additional term is a train-only
scalar derived from existing hidden states; it adds no parameters, attention slots,
KV tensors or generation path. Both arms log ordinary CE and interaction scores.

Four CPU checks passed: paired causal/rollover layout, token-prior cancellation,
public live forward versus packed decision logits with future/padding isolation,
and exact serialized model/Adam continuation through actual weighted-loss updates.
Fullsize zero-update preflight retained exact parent weights and all finite gradients.
Median update-body timings .2463s ordinary/.2363s interaction from4 warmed samples
are noisy and do not establish a speed gain; they exclude Adam/encoding/evaluation/
saving. Heavy batches with Adam-sized scratch used2.149GiB allocated/2.486GiB
reserved. Inference is the identical model function in both arms.

Use tools/train_token_memory_64_binding.py with separate run
dump/token-memory-64-binding-pilot-20261016 and explicit --resume after a checkpointed
pause. All saved source/data/parent identities must match. First launch pauses after
one real update to verify checkpoint state before completing the bounded protocol.
Evaluate full original development language, exact free answers and paired success,
missing cases, held-out relation, blocked memory and original historical facts.
Neither easier prompts nor teacher-forced interaction scores count as goal completion.

## Paired binding diagnostic: completed, no useful held-out retrieval

Both arms completed the prescribed 512 updates with identical batches and labels:
1,868,770 original-language labels and 82,960 reading labels per arm. This remained
a continuation from the trained plain64 checkpoint, not fresh two-epoch training.
Training, evaluation and independent verification finished successfully. The
original 124 protected sources, checkpoints, data and cache files are unchanged.

| Arm | Full-dev CE | Full-dev accuracy | Test positive exact | Test missing exact | Test pairs both correct |
|---|---:|---:|---:|---:|---:|
| Trained plain64 parent | 2.214284 | 53.504594% | 0/460 | 0/52 | 0/256 |
| Ordinary reading loss | 2.217795 | 53.479209% | 3/460 | 15/52 | 0/256 |
| Additional interaction loss | 2.217899 | 53.474423% | 4/460 | 17/52 | 0/256 |

All language rows score the same 1,859,388 development labels. The interaction
arm is 0.030171 percentage points below the parent; its paired-document 95%
interval is [-0.055327, -0.005658] points. The interaction-minus-ordinary interval
includes zero. These intervals describe fixed checkpoints and documents, not
training-seed uncertainty. Both trained arms still score 0/103 historical notebook
facts and 0/32 changed facts. Only 2/64 pairs in a training subset were both correct
for the interaction arm (ordinary: 0/64), so the failure is not solely held-out names.

Teacher-forced first-divergence accuracy rose from 8/512 to 89/512 and 93/512;
mean paired logit interaction rose from 0.0061 to 0.0102 and 0.0470. Free answers
mostly learned formatting and termination: EOS increased from 38/512 to 508/512.
These changes did not establish correct use of the changed fact. Correct missing
answers must not be reported as successful recovery of positive facts.

A subsequent read-only attention diagnostic checked all 256 development cases.
At the first differing answer token, layer 7 head 8 assigned mean attention mass
0.1065 / 0.1676 / 0.4344 to the positive fact's code(s), respectively. Manual
weighted-value calculations matched the public attention output. The interaction
arm became more sensitive to old-code changes, yet still answered no held-out
pair correctly. Attention weights alone are not a causal explanation or a
retrieval metric; attending to a code also does not identify a record inside it.

Cached-code geometry offers no support for the claim that RMS normalization
erased all fact differences: median relative changed-code distance was 27.46%
before and 27.41% after normalization. This is descriptive, not an information
preservation proof. The original decoder reconstructed all tested fact bodies,
but no decoder features were supplied to the language model.

Artifacts: `dump/token-memory-64-binding-pilot-20261016/evaluation` contains all
outputs, metrics, source provenance and independent verification;
`attention-diagnosis` contains the attention trace and checkpoint guards.
The next read-only control compares short facts in the raw window against the
same facts in old compressed history, with generation kept away from rollover.
Its purpose is to separate basic reading ability from compressed-code access;
providing a short selected fact is a diagnostic convenience, not a production fix.

## Raw visibility and conditional completion controls (verified)

The next goal turn completed three read-only controls, without training or
changing any production model. The preceding cleanup recheck was no retrieval
progress; these controls supply new evidence about the failure. All 124 protected
files still match their recorded hashes. The quality/retrieval objective is unmet.

Prepared 96 counterfactual pairs (192 cases), balanced across color, word and tool;
one versus two records; and familiar versus held-out first names. The two-record
pairs swap values while keeping the bag of fact words fixed. Nearby prompts fit
in raw64 with eight output-token positions reserved. Distant prompts insert exactly
128 ordinary filler tokens between facts and the query, preserving the answer-start
phase and all fact/question wording. Their facts lie entirely in the first old
64-token block; the second old block is filler. Changed-fact sides have identical
raw prefixes. No generation crosses a rollover in any raw64 condition.

The initial question-answering control exposed a format confound: even the raw
models often began sentences instead of obeying "Value only." The eight-token cap
then truncates many such sentences. Plain64 had 4/192 exact nearby answers and
0/192 distant answers; ordinary and interaction arms had 12/192 and 14/192 nearby,
both 1/192 distant. No arm, including original512, answered both sides of a pair
exactly. These numbers alone do not measure all of the models' reading ability.
The teacher-forced first-value token score also penalizes capitalization, so it
must not substitute for decoded-text factual accuracy.

A second control supplies a fixed answer prefix such as "Sana's color is" before
generating the value. The prefix contains the requested name/relation, never its
value. This is conditional completion, not autonomous question answering. All
192 original facts/pair sides remain; raw prompt lengths are 33--47 tokens, and
gold completions are at most five tokens including EOS. The original frozen codec
reconstructs every one of the 2,444 fact tokens in all 192 fact bodies. Its decoded
features never enter the language model.

| Trained weights | Nearby raw: correct value first | Distant: correct value first | Distant pairs both correct |
|---|---:|---:|---:|
| Plain64 parent | 103/192 | 0/192 | 0/96 |
| Plain64 + ordinary paired reading | 128/192 | 14/192 | 0/96 |
| Plain64 + interaction loss | 126/192 | 13/192 | 0/96 |
| Original512 | 141/192 | 110/192 | 37/96 |

Here "correct value first" requires the entire expected value at the start of
decoded output, with a word boundary. Further text is allowed and scored separately.
For the three raw64 rows, the positive values in the table also terminate exactly.
Original512 has 133/192 exact nearby answers but zero exact distant answers: it
often gives the correct value and continues the filler narrative instead of
stopping. Thus its 110/192 is evidence of conditional content access, not 110
fully correct complete answers. Original512 sees the entire distant prompt as raw
tokens; it remains a checkpoint trained with raw512 plus older compressed memory,
not a separately matched memory-free training baseline.

Blocking the fact block leaves all pairs' outputs identical. The ordinary and
interaction arms each start with a correct value by coincidence on 16/192 blocked
cases, more than their 14/192 and 13/192 unblocked counts, with zero correct pairs
and no EOS. These single-example hits are not reliable memory retrieval.

The third control loads the exact same raw64 checkpoint tensors into the original
public512 forward, giving it the entire distant prompt as raw text. This deliberately
uses more attention only for diagnosis; it is not a proposed production change.
All 576 nearby outputs replay their raw64 counterparts exactly. Distant correct-value
starts are still only 0/192 parent, 3/192 ordinary and 2/192 interaction, with no
correct pairs. This long raw visibility was never trained in those weights, so
the result does not prove that raw text is intrinsically difficult. It does prevent
attributing the observed near/distant gap entirely to the codec or its projection:
learned distance/context behavior remains a confound. The separately trained
original512 checkpoint's stronger distant content access supports investigating
training and position-dependent reading, not simply widening inference.

Two CPU checks audited every constructed pair, fact binding, fixed word multisets,
target decode and rollover bounds, plus exact-versus-substring scoring. All public
and independent reference forwards matched in the GPU controls. An independent
saved-output verifier rederived every metric, paired result, full-value prefix
score, fact-body oracle and near replay, checked source archives, and rehashed all
protected files. All processes finished successfully. There were zero optimizer
updates, no production source edits and no new persistent production KV entries.

Evidence directories:

- `dump/token-memory-64-raw-reading-control-20261016`
- `dump/token-memory-64-completion-control-20261016`
- `dump/token-memory-64-visibility-control-20261016`

Each contains outputs, protocol, source identities and independent verification.
The completion adapter changes only the case builder in its own process; the
visibility control is explicitly an oracle with extra raw attention. Neither is
a new model or a replacement inference path.

Next concrete training hypothesis: preserve ordinary language and reading losses,
but directly supervise the first changing factual value against the full vocabulary.
The prior interaction term rewards a relative paired-logit difference; it can improve
that difference while the correct value still loses to other vocabulary tokens.
A matched diagnostic can replace its additional term with full-vocabulary decision
CE, retaining the completed-parent initialization, identical batches and 512-update
budget. Reuse the verified ordinary control rather than retraining it. This differs
from the unsuccessful fresh-1000 answer-weight pilot in initialization and paired
reading data; it remains a hypothesis, not a proven cure. Preserve free-generation,
changed-value pairs, missing answers, full-dev language and the existing raw/completion
controls as evaluation gates. Only a real reading gain justifies progressing to a
fresh full two-epoch comparison; none of these diagnostics satisfies that goal.

## Full-vocabulary decision-loss diagnostic: prespecified run

Implement the next arm in `tools/train_token_memory_64_decision.py`, outside all
production model sources. It copies the original completed plain64 step48,762
weights with fresh Adam and uses the exact 512 batches, sampling order, cached
codes, labels, learning rates and two reading-data passes of the verified ordinary
paired-binding control. The ordinary control will be reused, not retrained.
The objective is `.9 * languageCE + .1 * readingCE + .05 * decisionCE`, where
decisionCE is full-vocabulary cross-entropy at the first differing answer token
in each counterfactual pair. All labels remain ordinary target tokens; position
indices select loss terms only, not model inputs. Missing-side cases remain.

This is additional diagnostic continuation, not fresh initialization or two epochs
of the original language corpus. The model, legacy positions, 23,209,216 parameters,
one attention position per code and inference/KV path are unchanged. No pretrained
decoder features, raw-history expansion, new vocabulary or extra inference calls.
Run: `dump/token-memory-64-decision-pilot-20261016` with a distinct format and
source/data/checkpoint guards. First pause after one complete update to inspect
actual saved Adam/cursor/batch state, then explicitly resume the remaining updates.

Four CPU checks passed, covering the relative-margin/full-vocabulary distinction,
exact zero-coefficient parity with the original ordinary optimizer update,
all-parameter gradients against independent live past-only decision forwards,
and actual serialized Adam continuation. The fullsize real-data zero-update GPU
preflight passed: maximum all-parameter decision-gradient difference 3.34e-6,
parent weights unchanged, heavy batch 2.238 GiB allocated / 2.713 GiB reserved
including two Adam-sized buffers and one additional scratch buffer. Three warmed
update-body samples gave medians .209667s ordinary / .209033s decision; this small
sample establishes no speed improvement and excludes Adam, encoding, evaluation
and saving. Full inference remains exactly the same function.

Evaluation gates are full original development language, all paired dev/test
free answers, training-fit pairs, missing and blocked controls, historical
notebook/changed-fact answers, and the newly verified short raw/completion controls.
Correct free answers and correct counterfactual pairs are required; lower decision
CE, token accuracy over fixed wording or a larger paired-logit difference alone
will not justify adoption. The broad quality/budget objective remains unchanged.

## Decision-loss diagnostic completed: stronger abstention, no retrieval gain

The new arm completed all 512 updates. Its actual step1 checkpoint was audited,
then resumed through step512; no restart or duplicate ordinary-control training.
Every original-language batch, reading pair, target count and learning rate matches
the ordinary control. Totals: 1,868,770 language labels, 82,960 reading labels and
8,192 additional first-decision loss terms. All Adam counters are512. Effective
throughput was about4.53 updates/s, without changing the model's inference path.
This remains additional diagnostic continuation from the trained parent, not a
fresh two-epoch result on the original corpus.

| Metric | Ordinary reading control | Additional decision CE |
|---|---:|---:|
| Full-dev CE | 2.217795 | 2.219261 |
| Full-dev token accuracy | 53.479209% | 53.466786% |
| Positive test facts, exact | 3/460 | 3/460 |
| Test pairs both correct | 0/256 | 0/256 |
| Missing test answers correctly unknown | 15/52 | 32/52 |
| Positive test facts falsely answered unknown | 120/460 | 298/460 |
| Historical notebook facts | 0/103 | 0/103 |
| Historical changed facts | 0/32 | 0/32 |
| Nearby conditional value completion | 128/192 | 127/192 |
| Compressed conditional value completion | 14/192 | 12/192 |

The decision-minus-ordinary language accuracy difference is -0.012423 percentage
points; paired-development-document95%CI [-0.022895,-0.002067]. Against the original
plain64 parent, the difference is -0.037808 points,95%CI [-0.063245,-0.012960]. These
5,000-draw intervals describe fixed checkpoints/documents, not seed uncertainty.
The original full development set still contains1,859,388 labels.

The aggregate test exact count rose18/512 to35/512 entirely because missing answers
improved; positive recovery did not improve. Teacher-forced first-decision accuracy
rose only89/512 to94/512. Test mean paired-logit interaction was0.0132, versus0.0102
ordinary and0.0470 in the earlier interaction arm. Changed facts changed only15/256
outputs (ordinary28/256), with no pair both correct. On the training-fit subset,
positive exact remained5/115 and paired success0/64. Thus this was not just failure
on held-out names. Blocking facts produced84/128 correct unknown answers, but that
must be interpreted together with the298 false unknown answers on positive tests.

The short value-only raw-question control improved12/192 to28/192 nearby, including
3/96 pairs, but only1/192 to3/192 distant with zero pairs. Its formatting limitation
still applies. The prefix-supplied completion control preserves nearby reading
(127/192) while distant completion remains12/192 with zero pairs. When the fact
block is blocked,16/192 outputs start with a correct value by coincidence, again
with zero correct pairs. None establishes reliable compressed-memory reading.

All evaluations and independent audits finished successfully. They cover every
development segment/source metric, paired/category free-answer score, historical
exact score, raw/completion control and protected identity. All124 original protected
files remain unchanged. The first step1 audit attempt accidentally counted eight
fixed FFN scale buffers as trainable parameters; the independent auditor was corrected
to count actual parameters, and passed without any trainer/model change or restart.
The model has23,209,216 trainable parameters and the original inference/KV structure.

**Do not adopt or continue this arm simply because its decision loss fell.** The
result supports a stronger abstention tendency, not improved factual reading. It
does not establish that all data weighting or longer training would fail. Evidence
is under `dump/token-memory-64-decision-pilot-20261016`, including actual checkpoint,
all update rows, initialization/resume audit, complete outputs, source archives and
`evaluation/independent-verification.json`. No worker from this experiment remains.

The next investigation should target direct, position-specific access to code
contents during training, rather than another unmotivated answer-weight adjustment.
A bounded reconstruction/reading curriculum using the existing raw64 model can
test whether it can learn that access while retaining one memory slot per code.
Such an auxiliary task must remain training-only; no decoder expansion, reconstructed
raw history or extra KV may enter production inference. A small reconstruction win
would only justify a later retrieval/language experiment, not satisfy the broad
goal. This proposal was subsequently tested in the diagnostic below.

## Capacity question and training-only reconstruction diagnostic

The user now permits slightly more compute or VRAM if the main compression
advantage remains, and asks whether23M parameters alone explains failed retrieval.
The historical retrieval-only run provides a useful limited counterexample:
`dump/memory-retrieval-2000-20261012/result.json`, original saved predictions in
`after.json`, and frozen training source `source.py` were re-read. Recomputed
correct counts are512/512 new histories,464/512 held-out bindings,105/512 unseen
question wording after2000 additional updates. The task uses8 fixed names,
16 single-token values and a fixed128-code bank. It predicts over the full
vocabulary, but the task's answer set is tiny. This demonstrates constrained
retrieval at the same23,209,216-parameter budget, not arbitrary compressed-text
reading. Original language accuracy worsened41.30% to38.53% on that older recipe.
Do not compare those language numbers to today's61% reference.

The new experiment is prespecified in
[token_memory_64_reconstruction_protocol.md](token_memory_64_reconstruction_protocol.md).
Four CPU tests and full-size zero-update GPU checks pass. Data preparation and
independent audit establish4096train/512dev/512test distinct genuine blocks across
17 sources; original train/dev cached codes match exactly. Frozen decoder recovers
all sampled train/dev content tokens and32735/32736test content tokens. It is only
a diagnostic and supplies no model inputs or decoded production history.

Both copies of the completed plain64 parent finished512 auxiliary continuation
updates, fresh Adam, same data/order/schedule. Each saw1868770 original language
labels plus524288 reconstruction labels, half all-BOS position-only and half
causal teacher-prefix queries. One arm masks the code, one reads it, preserving
identical coordinates and one stored position per code. Loss is .5LM+.5reconstruction.
This is not a fresh two-epoch baseline comparison or retrieval-only specialist.
Actualstep1 was paused/audited then resumed; final audit verifies all Adam512,
counts/cursors/source identities and124 unchanged protected files. No src/model
or original checkpoint/cache changes. Training ran about4.6updates/s per arm.

Run: `dump/token-memory-64-reconstruction-pilot-20261016`.
Both512-update arms and read-only evaluation are complete. Independent verification
recomputed all saved reconstruction/source/position metrics, full-dev metrics,
free-answer/counterfactual and historical scores, and all protected hashes.

| Fixed checkpoint | Original full-dev CE | Original full-dev accuracy | Reconstruction test content accuracy, no raw target prefix | Same checkpoint with code blocked |
|---|---:|---:|---:|---:|
| Completed parent |2.214283787|53.504594%|728/32736 (2.2239%)|804/32736 (2.4560%)|
| Blocked-code auxiliary control |2.221130414|53.407519%|1052/32736 (3.2136%)|1026/32736 (3.1342%)|
| Code-enabled auxiliary |2.227575978|53.312326%|1888/32736 (5.7674%)|876/32736 (2.6760%)|

The code-enabled model drops to834/32736 (2.5477%) against the original labels
when codes are cyclically swapped. Its position-only predictions permute exactly
with the donor code. That verifies the input/label separation and some content
dependence, not accurate factual retrieval. Every arm has0/512 exact reconstructed
test blocks. Code-enabled reconstruction train-fit content accuracy is6.5143%
on128 selected training blocks; it has not mastered even this fitted subset.

With a correct teacher prefix, code-enabled test content accuracy is56.3691%,
versus54.8021% with code blocked. This is not free factual recovery: prefix
predictability supplies most of the score. Original language accuracy worsens
0.192268pp versus parent (paired-document95%CI[-.219316,-.164559]pp) and0.095193pp
versus the matched blocked-code auxiliary (CI[-.114195,-.076575]pp). These intervals
describe fixed checkpoints/documents, not training-seed uncertainty.

All three checkpoints return0/460 exact positive answers,0/52 correct missing
answers and0/256 correct pairs on the existing512-case factual test. Both new arms
also have0/230 positive dev,0/103 historical notebook and0/32 changed historical
facts. The old binding suite's128-case `train_fit` subset is reused as a transfer
probe; it is NOT the reconstruction pilot's training data. Neither new arm solves
its115 positive cases. The128 reconstruction `train_fit` blocks above ARE selected
from this pilot's own training blocks. Keep these two meanings distinct.

**Do not adopt or automatically extend this auxiliary recipe.** It teaches weak
code-content sensitivity but no useful free factual recovery and reduces language
accuracy. Failure after512 updates of this mixed objective does not prove that
23M is too small, that arbitrary code reading is impossible, or that a dedicated
retrieval curriculum cannot work. A subsequent capacity test should directly
compare retrieval-only raw/encoded facts with answer-only targets, fresh code
blocks and progressively varied bindings/wording; fixed-bank mastery is only a
starting control. Broad retrieval and the fresh two-epoch goal remain unmet.

All workers finished. Artifacts include per-token predictions/NLL, all generated
outputs, manifests, source archives, checkpoint/optimizer audits and
`evaluation/independent-verification.json`. No original checkpoint, source or cache
was changed; the production inference architecture remains unchanged.

## Direct retrieval-only raw/code capacity test

Following the user's capacity question, a stronger answer-only comparison is
specified in [token_memory_64_retrieval_only_protocol.md](token_memory_64_retrieval_only_protocol.md).
It compares two exact copies of the completed plain64 weights, fresh Adam: the
unchanged compressed-memory model versus a higher-cost raw-text control seeing
the same complete source histories/questions. The raw control is diagnostic,
not a replacement for compressed inference. Production sources remain unchanged.

The prepared data contains32768 train examples in16384 counterfactual pairs,
four256-example dev groups and six512-example test groups, with54437 genuine
FP32 code blocks. Every group's prompts and fact-bearing blocks are disjoint.
Tests separate new blocks, bindings, names, wording, values and combined changes.
Answers include multi-token access phrases and notebook locations, alongside
colors;10% of examples are missing. Finite values and one/two-record templates
remain limitations. Unlike the old128-code-bank task, test fact blocks are new.
Every held-out fact body is reconstructed exactly by the original frozen decoder,
used only as an information diagnostic. Small train errors were retained.

Five CPU checks passed, including actual gradient/public-forward alignment,
source/target causality, microbatch normalization, real serialized Adam resume
and teacher/free-answer scoring. Fullsize zero-update preflight passed with
unchanged weights and gradient errors below1e-5. Independent all-case audit
verifies actual answer semantics, held-binding partitions, original token blocks
and pair-prefix equality; all124 protected original files remain identical.

Both arms completed2048 updates, two passes over this NEW data:65536 examples and
354284 answer-label exposures each, effective32/micro16, FP32 AdamW.01/clip1,
warm50/LR3e-5 cosine3e-6. No language replay. This is additional specialist
continuation, not the eventual fresh two-epoch language comparison. Complete
logs, source hashes, cursor/order/LR/label checks and all Adam2048 counters passed.
Actual training throughput was approximately7.3 updates/s raw and8.8 code on this
short answer-only diagnostic; these are not general LM training/generation speeds.

Run: `dump/token-memory-64-retrieval-only-pilot-20261016`.
Read-only evaluation and independent results verification are complete. Saved
teacher predictions, free answers, EOS, paired/control scores, original-dev
language metrics and historical scores were independently rederived. All124
protected original files remain unchanged. No checkpoint has been adopted.

The following counts require the entire correct value followed by EOS, with
case/whitespace/trailing-period normalization. Each test has460 answerable cases
and52 missing cases. These are free answers, not teacher-forced token accuracy.

| Held-out test | Raw-text specialist | Compressed-memory specialist |
|---|---:|---:|
| New histories, familiar names/values/wording |351/460 (76.30%)|224/460 (48.70%)|
| Unseen name-value bindings, familiar components |347/460 (75.43%)|51/460 (11.09%)|
| Unseen names |236/460 (51.30%)|177/460 (38.48%)|
| Unseen question wording |360/460 (78.26%)|223/460 (48.48%)|
| Unseen value strings |56/460 (12.17%)|0/460|
| Combined unseen names/values/wording |12/460 (2.61%)|0/460|

Both unadapted parent controls score0/460 on every group under this strict
value-only answer format. After adaptation, familiar-test complete paired success
is171/256 raw versus60/256 code; held-binding pairs165/256 versus11/256; unseen-value
pairs10/256 versus0/256. Missing-answer success on the familiar test is52/52 raw
versus3/52 code. Thus the code specialist has learned some useful reading, but
neither memorized value production nor aggregate accuracy establishes binding.

A descriptive post-evaluation breakdown in `evaluation/binding-breakdown.json`
separates the one-record and two-record positive cases. On familiar histories,
raw solves230/230 single-record cases but121/230 two-record cases; code solves
148/230 and76/230 respectively. On held bindings, raw remains230/230 single-record
and117/230 two-record; code falls to39/230 and12/230. This supplies direct evidence
that23M can learn narrow retrieval, while even raw access does not reliably bind
two records under this recipe. The code path has an additional access/generalization
gap. These data do not isolate a universal parameter-capacity limit or one specific
architectural cause.

On the first64 familiar-test controls, code has28/57 correct positive answers with
the appropriate memory versus4/57 with fact codes blocked, and8/32 correct pairs
versus0/32. Swapping memory reproduces donor outputs exactly; blocked pairs have
identical outputs and cancel their paired decision contrast. This supports real
memory dependence for some cases, not reliable arbitrary reading. The frozen
codec decoder reconstructs all held-out fact bodies exactly; it is never used by
these readers and this sampled result does not establish a universally lossless
codec.

| Weights, all evaluated through the same original plain64 language path | Full-dev CE | Full-dev accuracy |
|---|---:|---:|
| Completed parent |2.214283787|53.504594%|
| Raw-text retrieval specialist |2.300275921|52.081115%|
| Compressed retrieval specialist |2.307603277|51.971563%|

The code specialist loses1.533031pp against parent (paired-document95%CI
[-1.579569,-1.486134]pp); raw-specialist weights lose1.423479pp
([-1.464097,-1.383175]pp). These intervals describe fixed checkpoints, not
training-seed uncertainty. The common plain64 transfer evaluation is distinct
from the raw specialist's higher-cost512-position retrieval control. Both
specialists still score0/103 historical notebook facts and0/32 changed facts.

**Do not adopt or automatically extend either specialist.** Retrieval-only
supervision demonstrates learnability at23M, but this finite-value curriculum
does not produce general factual reading and damages language quality. Name
co-occurrence within partition groups, one/two-record templates, finite values,
one seed and a bounded continuation are limitations. Better query-to-record
binding, broader value generalization and preservation of language quality are
the next unresolved problems; simply increasing size is not a demonstrated fix.
The original broad retrieval and fresh two-epoch quality/budget goal remains
unmet. All workers are finished; original sources/checkpoints/caches are unchanged.

## Four-way question/fact binding diagnosis and follow-up

The retrieval-only error audit found that raw familiar two-record predictions were
correct121/230;107/109 wrong answers were the other record's value. Compressed
predictions were correct76/230, with73 wrong-record answers and81 outside-record
answers. For the code model, correctness was60/119 when the query referred to the
first record but16/111 when it referred to the second. These are descriptive
counts from the existing outputs, independently parsed from the actual fact text.

The next diagnostic asks both entity questions against both swapped memories,
holding token lengths, code blocks, raw64 coordinates and the common answer prefix
fixed. Details and the prospective training recipe are in
[token_memory_64_query_binding_protocol.md](token_memory_64_query_binding_protocol.md).
All eligible pairs are included without prediction filtering; equal-length
selection limits coverage. Every source/target serialization and cached original
block was independently checked. No optimizer was used for this diagnosis.

On familiar test histories,60 eligible quartets produce240 free answers:

| Existing specialist | Exact complete answers | Both questions correct for one memory | All four correct | Answer changes when only question changes |
|---|---:|---:|---:|---:|
| Raw-text retrieval |117/240|6/120|0/60|14/120|
| Compressed-memory retrieval |76/240|3/120|0/60|49/120|

Every dev/test group and the32-quartet training fit probe has zero all-four
success for both specialists, including full-vocabulary first-decision accuracy
across all four conditions. On familiar test, the mean factorial logit interaction
is .002926 raw and -.004047 code. Blocked-memory controls cancel it exactly and
produce identical outputs when only the blocked memory changes. All124 protected
original hashes remain unchanged. Results:
`dump/token-memory-64-query-binding-diagnosis-20261016/independent-verification.json`.

This strengthens the evidence of a query-to-record binding failure. Some value
extraction is learned, but asking for the other person often returns the same
value. Changing an answer alone is not evidence of correct binding. This does
not establish a universal capacity limit or guarantee a particular remedy.
[Zoology](https://arxiv.org/abs/2312.04927) likewise motivates distinguishing simple
recall from multi-query associative recall; it does not test this codec.

A matched training-only follow-up ran in
`dump/token-memory-64-query-binding-pilot-20261016`. Both arms start from identical
copies of the completed code specialist2048, with fresh Adam. Each gets two passes
over3620 new four-way training quartets,1810 updates,16 original language segments
plus16 reading examples per update. Both use .75LMCE+.25readingCE; one adds
.25softplus(1-C), where C is the memory-by-question factorial logit interaction at
the first divergent answer token. Additive memory-only/query-only/value priors
cancel. This does not replace full-vocabulary CE or free-answer evaluation.

The loss adds no inference operations, parameters, attention positions or KV
entries. It is not the final fresh two-epoch language comparison. Five CPU tests
pass, including actual mixed-gradient/microbatch and serialized Adam resume.
Fullsize zero-update gradient comparison with original public forward has maximum
parameter error1.28e-6; heavy-batch preflight reserves2.566GiB including Adam-equivalent
scratch. The first actual saved update independently passed labels/cursor/Adam/
source audits before resuming. Training, evaluation and independent results
verification all passed; all124 protected original hashes remain unchanged.

The binding objective improved several individual familiar-style answer groups,
but not robust question-to-record binding:

| Test positive exact complete answer | Ordinary | Factorial-binding loss |
|---|---:|---:|
| Identity |136/460|163/460|
| Held bindings |22/460|32/460|
| New names |90/460|110/460|
| New wording |101/460|125/460|
| New values |0/460|0/460|
| Composed transfer |0/460|0/460|

The counterfactual test asks both questions against both swapped memories. The
ordinary arm has zero fully correct quartets in every group. The binding arm has
one fully correct quartet of57 in new wording and zero in identity (0/60), held
bindings (0/62), names (0/56), values (0/51) and composed transfer (0/36).
It also has zero fully correct training-fit quartets (0/32). Thus the objective
changes outputs more often but does not deliver reliable factor-specific retrieval.

Language stayed close to the original parent through language replay: ordinary
53.541058% and binding53.531969% full-dev accuracy, versus parent53.504594%.
The fixed-checkpoint paired-document95% interval for binding minus ordinary is
[-.01930,.00175] percentage points; it is not training-seed uncertainty. These
weights are diagnostic continuations from the retrieval-only code specialist,
not a fresh two-epoch baseline comparison. Do not adopt or extend either arm.
Results are independently rederived in
`dump/token-memory-64-query-binding-pilot-20261016/evaluation/independent-verification.json`.
Broad unseen-value retrieval and the61% reference remain unresolved.

## Native256 frozen versus trainable packing pilot (2026-10-17)

User authorized a separate512-update-per-arm comparison preserving one256-number
code per64 tokens and the raw64 reader. Both arms initialize from an immutable
copy of the partial native256 step8750 checkpoint, fresh Adam; the original
native source/run/logs/checkpoints remain unchanged. One arm trains the reader
only; the other also trains the existing2,080,000 packing weights. No production
positions, parameters or KV entries are added. Frozen-prefix features are cached;
current codes are recomputed for both factual tasks and language replay.

Both receive identical8192 new reading examples and8192 original language
segments:128 one-record updates, then384 two-record counterfactual updates,
16LM+16reading examples/update, micro8. Loss .75LMCE+.25answerCE+.05frozen-decoder
reconstruction (training only). Full prospective protocol and verified report:
[protocol](token_memory_64_pack_pilot_protocol.md),
[results](token_memory_64_pack_pilot_results.md).

Training, evaluation and independent results verification passed. Each arm has
1,872,347 language labels,34,264 reading labels,131,072 reconstruction labels and
all Adam states512. All159 protected files unchanged; all workers finished.
Full-dev parent/frozen/packing accuracy45.601133/46.602000/46.603452%.
Packing-frozen+.001452pp, document95%CI[-.001675,.004556]pp for fixed checkpoints.
Both have0/230 positive exact answers in every new held-out group,0/104 actual
training-fit positive answers,0/51 familiar positive two-record all-four success,
and zero such success in other held-out groups. Both incorrectly answer unknown
for228/230 familiar positive examples;152/156 missing test answers are correct.
The aggregate four training-fit solved quartets contain only missing answers.
Relaxed full-value checks also zero. Historical0/103 notebook,0/32 changed,0/8
compressed colors; historical missing21/25 unchanged from parent.

Packing weights changed with active gradients and median.930607% code drift.
The frozen codec decoder still recovers24,963/24,964 fact-body tokens and895/896
complete bodies in ALL arms. Thus the bounded unfreezing did not unlock reading;
the learned output mode is predominantly abstention. Codec content fidelity is
not reader success. This partial-parent experiment does not establish a universal
capacity limit or reject all joint training. Do not adopt or extend this recipe;
positive single-fact reading must be established before scaling. The original
broad retrieval/61% quality goal remains unmet.
