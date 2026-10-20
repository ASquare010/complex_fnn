# Fresh original-corpus query-identity pilot

18 October2026, prospective before run export or corpus optimization. Preparation,
all-case labels/surface checks, actual objective CPU/GPU/microbatch/capacity gates,
equal-budget timings and2,323 earlier protected hashes have passed. Previous goal
turn was progress; quality and the full goal remain unproven. New run only:
`dump/native256-query-entity-unified-v1`, format
`native256-query-entity-full-original-v1`. Never resume through another trainer.

Architecture is the frozen repaired `token_memory_64_query_entity_v2`: exactly
23,209,216 trainable READER parameters, frozen encoder separate, raw64, one global
OLD attention/KV position per64 BPE tokens, zero extra KV bytes. The query path
adds3,072 weights and16,384 fixed buffer bytes, removing six FFN units. Fresh47
initialization must reproduce this new architecture exactly with empty Adam.
It is not tensor-identical to Span: six FFN widths and query weights differ.
No old trained checkpoint, native run, source, original corpus or cache is edited.

Use the ORIGINAL780,186 training segments/180,678,552 assistant labels per epoch,
original orders seeded47+epoch,48,762 updates/two-original-epoch horizon and
361,357,104 total assistant labels. Effective batch32, micro8, FP32/noTF32,
AdamW.01, clip1, LR3e-4/warm200/cosine minimum3e-5. New objective is frozen global
assistant CE +0.05 query-slot CE +0.02 positive critical-source CE, global segment/
field/two-read-layer means. Aux embeddings/core h/old pointer/norm are detached;
only new selectors/gates receive auxiliary gradients. Missing cases have query
supervision but no forced generic-NULL source loss. No additional dialogs/passes.
All gold coordinates are training-only. Literal word-ID metadata is added to
the existing FP32 code values, not proof semantic compression is lossless.

First bounded pilot:1,000 ORIGINAL updates, matched to completed ordinary Span
at1,000, not two epochs completed. Intentionally stop after actual update1 and
independently audit saved fresh state, format/recipe/source hashes, original
batch/labels/LR/cursor, auxiliary kinds/denominators and ALL Adam counters. Resume
only this new checkpoint for999 further updates after that audit passes. Every
later continuation requires an explicit positive NEW-update bound and a recorded
evidence-based decision; do not automatically extend a failed/unevaluated pilot.
Skipping initial full DEV is permitted; step0 initial checkpoint and exact fresh
identity/empty Adam are still checked. No quality claim from training loss.

Before update1, independently check export freshness, orders, baseline all48,762
batch/label/LR provenance and all original corpus/cache identity. The existing
complete5-million-block/label audit can be inherited only by identical original
input hashes AND frozen data/source hashes, with provenance explicit; recheck
new initial tensors/RNG and all auxiliary label/surface receipts. Old protected
files and completed controls remain immutable. Strict source closure includes
new trainer/auditor/protocol and all executed model/objective/gate dependencies.
Do not edit guarded model/training/evaluation files while workers are live.

At1,000 audit every actual batch, label, LR, cursor, auxiliary count and Adam
counter. Evaluate full7,831-segment DEV/1,859,388 labels, natural full facts,
changed values/questions and blocked controls, missing answers, all seven legacy
groups, historical312 and identical capped numeric probes. Use frozen evaluator
and independent uncached CPU/source/output replays. Include full notebook
BOTH-query/BOTH-swapped-memory all-four controls and proper known-name completion,
with prefix-independence/visibility strata and blocked-swap invariance. Separately
measure query selection on actual train exposures and development questions;
correct selection/teacher loss alone is not free factual reading.

Compare same actual prompts, original orders/assistant exposures and fixed-weight
paired-document/source/unit intervals against ordinary Span1,000. Report the
six-FFN-unit reallocation and extra derived supervision rather than claim exact
initial weight/objective matching. Existing source-alignment failure may be a
descriptive extra control; never restart it. One seed/earlycheckpoint is not
final convergence or universal capacity evidence. Material timing cost remains:
body+23%, cached token+66% versus same-budget Span; no speed/adoption claim.

Advancement requires reviewing complete positive/binding/missing/language results
together, not lower auxiliary CE or a few familiar hits. Whole-goal acceptance
still requires verified fresh full2 epochs, strong/matched uncompressed reference,
original TEST and reserved novel literals AFTER actual48,762/361,357,104 exposure
gate, broad factual/reasoning quality and actual end-to-end efficiency accounting.
TEST is never a pilot tuning set. No subagents, growing resurrection or second
GPU worker.
