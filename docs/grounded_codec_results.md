# Grounded reading comparison: no successful replacement

2026-10-13. Training and independent verification are complete. None of these
fresh short-answer pilots established useful factual reading. They do not
justify replacing the existing trained token + compressed-memory model or
restarting a long direct-vector run. The broader language-generation goal is
unresolved; production position-memory remains paused at update 1,001.

## Comparison and data

The [frozen protocol](grounded_codec_protocol.md) uses official
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) passages/questions:
8,192 train, 512 internal development and 512 exploratory test examples.
Selected article titles, passages and questions are disjoint. Literal overlap
with v4 was excluded; this is not semantic decontamination or a hidden benchmark.
The relevant full passage is provided, without answer-centered cropping. Gold
answer locations are not inference inputs or auxiliary losses in this first test.

Each arm has exactly 23,209,216 trainable parameters and starts fresh. Each trains
2,048 updates, batch 32, eight epochs, with the same question order and 629,296
assistant-label exposures (367,152 answer-body labels). Raw microbatch is eight;
vector microbatch is 32. These are matched exposure and parameter budgets, not
identical architectures or FLOP budgets. Original frozen codec parameters add
12,555,776 unique parameters to the vector system and are reported separately.

Raw predicts tokens autoregressively. Deterministic and conditional-flow arms
predict new eight-token codec vectors; free generation appends those predictions
unchanged. Their predictor and frozen decoder do not receive raw input token IDs.
Flow uses eight fixed Euler steps. All outputs stop at literal `[END]` or a
32-token cap; all reference answers fit. CE weights answer body and marker
separately (.9/.1); vector training uses normalized MSE, adding .02 CE after the
first 200 updates. Full details and hashes are in the protocol and run artifacts.

## Final held-out free generation

All percentages below concern generated answers, not teacher-forced accuracy.
Exact match and F1 use retained reference answers. The 410 answerable questions
and 102 missing-answer questions are separated to expose constant-response bias.

| Fresh arm | Answerable exact | Answerable F1 | Missing-answer exact | End-marker stops / 512 |
|---|---:|---:|---:|---:|
| Raw tokens | 1/410 (0.24%) | 0.60% | 35/102 (34.31%) | 511 |
| Deterministic vector | 0/410 | 0.36% | 0/102 | 0 |
| Conditional flow vector | 0/410 | 0.15% | 17/102 (16.67%) | 196 |

Development gives the same qualitative outcome: raw 1/410 answerable exact,
deterministic and flow 0/410. A hash-selected training check gives raw 15/49
answerable exact, versus 0/49 for both vector arms. Raw learned some familiar
answers but generalized poorly; the vectors also failed this small training
check. This is not evidence that every 23M model must fail, nor a clean diagnosis
of compression alone: the fresh raw control is also inadequate. These pilots
are not the existing v4 checkpoint and their scores cannot be compared directly
with its teacher-forced CE 1.80471 / 61.01% token accuracy.

When supplied the genuine target vectors, the original decoder recovers all
512 test answers and all 2,404 annotated test evidence tokens. On development it
recovers 510/512 complete answers and all 2,667 evidence tokens; errors were
retained. Information being recoverable by the codec does not establish learned
question answering or prediction of a new valid vector.

## Independent controls

The verifier reproduced 64 hash-selected test answers per arm exactly, including
predicted-code byte hashes. On a fixed 64-case passage intervention set, all
three arms score 0/53 answerable exact under correct, wrong and blocked passages.
Missing-answer cases account for all exact successes. Wrong/blocked scores keep
the original references: they diagnose dependence, not factual correctness
under a different or absent passage. The question tokens/codes are unchanged;
passage length and positions can change.

Wrong/blocked passages change raw outputs on 53/54 of 64 cases, deterministic
outputs on 64/64, and flow outputs on 57/61. Sensitivity is not useful reading.
Two further flow seeds still give 0/53 answerable exact each. Ten reviewed,
explicitly fictional date replacements produce zero exact changed answers in
every arm. The decoder recovers their changed evidence. Nine passage lengths
are unchanged; one grows by one token. These failures do not independently
isolate a memory mechanism when baseline answers are already poor.

## Timing and verification

| Arm | Training + periodic dev/save | Final 512 test generations |
|---|---:|---:|
| Raw | 509.875 s | 75.687 s |
| Deterministic | 206.875 s | 77.766 s |
| Flow | 237.375 s | 80.531 s |

Single local runs, with prepared inputs. Training timing excludes preparation
and final evaluation; generation excludes encoding. Stopping lengths differ.
There is no demonstrated quality-adjusted speed improvement.

Thirteen new CPU tests passed. Zero-update GPU preflight checked every arm;
actual update-one pause/resume was exercised. Training completed 6,144 optimizer
updates total. Independent verification performed zero optimizer updates and
checked each final Adam count (2,048), final/last weight identity, parameter
counts, frozen original decoder identity, source/data hashes and protected
original checkpoints/corpus. All passed. Trainer session 39200 and verifier
2594 both exited successfully. No production/full-v4 run was restarted.

Artifacts:

- `dump/grounded-codec-data-20261013`: pinned preparation, data audits and source archives.
- `dump/grounded-codec-20261013/results.json`: final metrics and every generated answer.
- `dump/grounded-codec-20261013/verification.json`: independent checks and intervention outputs.
- `tools/verify_grounded_codec_training.py`: reproducible read-only verification.

## Decision

The best retained foundation for broad language remains the trained token +
compressed-memory model. No tested strict-vector replacement has surpassed it
overall. This is an engineering choice from the evidence, not a matched proof
that its architecture is optimal.

The next useful change would explicitly train evidence reading on that existing
foundation while preserving language ability: source/span supervision where
available, answer loss on facts that require old memory, and held-out changed,
wrong and blocked-memory checks. Merely adding decoder position features has
already been tested and is not a new solution by itself. Selectively decoding
retrieved records into text offers a separate reliability control, but relaxes
vector-only inputs and spends more attention work. See the
[research assessment](frozen_codec_research_direction.md). No further training
is started or recommended as a blind full-corpus restart by this report.
