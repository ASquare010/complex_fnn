# Same-template name-anchor pilot: narrow binding gain, language failure

Completed and independently verified on 18 October 2026. The overall goal remains
**ACTIVE / UNMET**. These immutable checkpoints must not be extended automatically.

The new name-before-value curriculum fixes a concrete shortcut in the preceding
pilot: both entity questions now use the same wording and token phase. On each
four-way unit, the model must answer both questions against both value-swapped
memories. Source record order also varies. Familiar answers rise from 266/512 to
500/512, with 120/128 all-four units, versus 4/128 for the matched control. This is
useful evidence of compressed-code binding under this narrow curriculum.

It is not a broad retrieval or language win. New names, positions, combined
changes and original language performance remain inadequate.

## Comparison and budget

Both arms start from the **same fresh seed47 trainable reader tensors**, copied
from the preceding pilot's saved initial state, with fresh Adam. The existing
encoder and trained selective word decoder stay frozen. This is not continuation
of a trained language reader. Both have 23,209,216 stored reader parameters,
including 3,981,312 frozen decoder parameters; 19,227,904 parameters are trainable.
The 9,426,432-parameter encoder is separate, as the user specified. One existing
attention/KV position still represents each completed 64-token block; raw64 and
the inference implementation are unchanged. No gold source position enters
inference.

Both complete 512 updates with identical source examples, LM orders, optimizer
schedule and batch sizes. Steps 1–128 use .75 language CE; the name arm also uses
1.0 name-position CE. Steps 129–512 use .75 language CE + .25 answer CE + .25 value
address CE; the name arm adds .25 name-position CE. The name target is the source
entity's first BPE position, or NULL when missing, derived from input text offsets.
It is an auxiliary training label only. The control's name coefficient is zero.

These are two passes through **new QA data**, not two epochs through the original
language corpus. Each arm trains on 1,872,347 original language labels and 12,660
answer/value labels after warmup. Each sees 8,192 name queries; the name arm trains
on all of them, the control on none. Both finish with all Adam states at 512.

The 4,096 training examples use balanced single-token values and variable record
order. Both questions within each four-way unit share their template. The 1,792
new held source blocks are disjoint from all preceding address-pilot blocks.
Held answer IDs are held out from this QA training, not from the pretrained
encoder or original LM corpus. There are seven familiar training names and four
held names; this finite vocabulary is a limitation. Fact offsets vary within
training; separate position tests use offsets absent from training.

## Free answers

Exact means the correct value followed by EOS, with no gold answer prefix.
All-four requires every answer in a complete both-questions/both-memories unit.

| Held group | Control exact | Name-trained exact | Control all-four | Name-trained all-four |
|---|---:|---:|---:|---:|
| Familiar | 266/512 | **500/512** | 4/128 | **120/128** |
| QA-held answer IDs | 264/512 | **484/512** | 10/128 | **108/128** |
| New names | 252/512 | 129/512 | 3/128 | 4/128 |
| New wording | 285/512 | **498/512** | 13/128 | **120/128** |
| New positions | 251/512 | 211/512 | 2/128 | 21/128 |
| Missing-information cohort | 140/512 | 325/512 | 0/128 | 32/128 |
| Combined new names/values/wording/positions | 216/512 | **0/512** | 0/128 | 0/128 |

The missing-information cohort contains 256 positive and 256 absent answers.
Name training scores 245/256 positive and 80/256 absent, versus 140/256 and 0/256.
On 128 actual positive training examples it scores 127/128, with 31/32 all-four
units, versus 72/128 and 4/32. Familiar source-name position accuracy is 503/512;
it drops to 181/512 for new names and 214/512 for new positions. Name training
therefore learns familiar keys but does not establish general entity matching.

Familiar exact accuracy is high in both source orders: 248/256 and 252/256.
Neither the blocked-memory nor shuffled-memory 64-case control gets a correct
exact answer. This supports dependence on the correct stored content, beyond
answer priors. It does not prove broad reasoning or generalization. All historical
notebook facts (0/103), changed-fact cases (0/32) and compressed colors (0/8) still
fail for both models; all 312 historical outputs were saved.

## Language cost and head ablation

Full original development set: 7,831 segments, 1,859,388 assistant labels.

| Model/output head | Accuracy | CE |
|---|---:|---:|
| Control, normal mixture | 19.656629% | 5.189927408 |
| Name-trained, normal mixture | 10.047930% | 5.941782259 |
| Control, generator only | 19.454089% | 5.199159038 |
| Name-trained, generator only | 10.061106% | 5.941511974 |

Name training costs 9.608699 percentage points against its matched control;
the paired-document bootstrap 95% interval is [-9.783904, -9.425124] points.
Disabling copying restores only .013176 points in the name arm, interval
[.006215, .019041]. Its generator-only accuracy remains 9.392983 points below
the control's generator. Copy interference alone therefore cannot explain the
language failure. This ablation does not isolate which auxiliary gradient or
shared representation causes it. Intervals describe these fixed checkpoints,
not uncertainty across training seeds. These fresh 512-update models must not be
compared as full-budget alternatives to the completed 48,762-update baseline.

## Verification and artifacts

Protocol: [prospective recipe](native256_name_anchor_protocol.md). Three CPU tests
cover ordinary-objective parity, microbatch gradients, serialized optimizer
resume across the warmup boundary and future-code isolation. Full-size GPU
preflight passed with max gradient discrepancy 5.59e-9 and unchanged initial
weights. The actual first update, all training orders/labels/LR/cursors and all
Adam steps were independently verified. Independent results checks rederived all
case metrics and anchors, replayed 72 CPU public-forward generation cases and
reaggregated language and historical outcomes. The head ablation has a separate
independent aggregate/CPU/state audit. All 1,901 protected old files remain intact.

Run: `dump/native256-name-anchor-pilot-20261018`. Training session 61662, audit
12543, evaluator 1666, independent evaluator 35915, head ablation 30258 and head
verifier 98705 all finished successfully. No worker from this pilot remains.

Checkpoint SHA256:

- Control: `07e21f2192bbc803a26b89075003db4820408d37f6a65d82fdf7e687f38d1d20`
- Name: `99582f49cdcd98bb3316893cbbd03394a3ea5057e25b210e5a526f38bfbd1218`

## What this changes

Keep the evidence that explicit source-name supervision can teach familiar
binding at unchanged inference cost. Do not adopt or extend this model. The next
reader needs transferable full-span lexical keys and a way to train query/field
selection without sacrificing ordinary language learning. The separate word-bit
storage experiment tests exact lexical metadata within existing FP32 code slots;
it is only a storage/numerical gate, not an implemented or trained replacement.
Final acceptance still requires fresh 23M-reader/two-original-epoch training,
broad factual and reasoning checks, genuine matched 1:1 comparison, language
preservation and measured complete inference/training costs.
