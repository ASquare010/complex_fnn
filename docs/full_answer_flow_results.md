# Complete-answer flow comparison: bounded experiment

2026-10-13. All three bounded training arms and independent reload verification
have completed. No useful-generation winner is established. The original
full-v4 position-memory training remains paused, and original codec, reference
weights and v4 dataset are protected by hashes. This is not a new full-corpus run.

## An interface error found before training

The first full-answer frozen-codec audit used the real chat EOS control token2.
Even genuine encoder targets did not reconstruct any of the 738 dev or778 test
EOS tokens. There were also220 dev and204 test non-EOS token errors. Thus perfect
prediction of these target vectors would not reliably terminate a conversation.
This does not explain every earlier language or memory failure, but it invalidates
the assumption that codec accuracy on ordinary text implies EOS compatibility.

We prepared a derived view with every control EOS rendered as the ordinary text
`[END]`, using existing tokenizer pieces `[61,3907,38,63]`. Every original selected
row was scanned for collisions, and conversion plus masks was checked reversible.
Short eight-token codec tails use ordinary space223, with padding excluded from
CE. Both raw and vector models receive the identical derived text and labels.
Original v4 binaries and codec weights are unchanged. This is a bounded-corpus
format, not a general escaping protocol for arbitrary future user content.

In the new genuine-code audit, all738/738 dev and778/778 test end markers recovered
exactly. Other token errors were215 dev and197 test; do not call the codec lossless.
Preparation took216.641seconds. Evidence:
`dump/full-answer-generation-check-20261013/verification.json` and
`dump/full-answer-literal-end-20261013/manifest.json`.

| Split | Dialogues | Scored rendered tokens | Eight-token vector targets |
| --- | ---: | ---: | ---: |
| Train | 4352 | 2600429 | 328044 |
| Dev | 489 | 285496 | 36015 |
| Test | 504 | 286419 | 36134 |

The selected text covers all17 v4 sources, includes every assistant label in
these dialogues, and retains prior assistant blocks in the same eight-token
format as generated feedback. Train-only target codes provide normalization.
Selection/exclusions are documented in [the preparation protocol](full_answer_flow_protocol.md).
The longest rendered training dialogue has2567tokens, from at most2560original
positions. Equal dialogues per source do not imply equal supervised tokens.

## Fixed comparison

Three fresh arms: raw tokens, deterministic vector regression, and conditional
flow matching plus frozen-decoder CE. Each has exactly23,209,216 trainable
parameters and sees the same two seeded shuffled epochs, effectivebatch8,
1088updates,5,200,858 rendered assistant-label exposures. Microbatch2 raw versus8
vector is a memory choice. Each arm starts at seed47. AdamW, clip1, weightdecay.01,
50update warmup to3e-4 and cosine decay to3e-5 are shared.

Raw uses ordinary teacher-forced CE. Deterministic uses normalized code MSE;
flow uses uniform-time straight-path velocity MSE. Both vector arms add .02
frozen-decoder CE after200updates. The codec networks remain frozen; target
codes never enter the current query as history. No EOS classifier or new
parameters were added. Original frozenD has no raw-token input.

The raw control uses factorized tied4096x256 token embeddings and1646-wide FFNs.
Vector models use1773/1774-wide FFNs and no trainable vocabulary table. Both use
the same attention/Branch-FFN family, but identical total parameters do not mean
identical architecture. Raw context capacity4096 accommodates rendered markers;
both arms receive the same source information.

Raw sees true earlier tokens within each eight-token block; vector outputs model
the block jointly. Therefore sampled-vector decoder CE is NOT raw language-model
NLL/perplexity. Pooled/per-source reference agreement is descriptive, and alternate
valid stochastic continuations need not match a single reference.

## Inference and verification

History KV is cached once across flow steps. Generated codes append unchanged
through one causal history update; no output re-encoding. Cached/full23M CUDA
sampled-code differences were below2e-6 with identical decoded tokens on the
tested short/long queries. Raw cached logits differed below5e-7. CPU tests cover
empty history, causality, future-query isolation, exact predicted-code feedback,
fixed caps and end markers spanning two output blocks.

22 CPU tests passed, including serialized same-microbatch Adam/model/RNG resume.
Pytest's Windows temporary-directory fixture hit a host ACL problem; the tests
use unique ordinary workspace directories instead. The real trainer performed
one update, saved at the requested boundary, then resumed at update1. No discarded
research updates. Frozen source snapshots and resume guards protect provenance.

Development evaluation occurs at each epoch end. Final flow solver selection
uses dev content agreement, then exact-block agreement, then fewer steps, among
Euler4/8/16. Test uses all504dialogues for teacher-history block scores and two
dialogues per source for final-answer generation, capped at128 rendered tokens.
The cap is independent of reference length; answers longer than it cannot achieve
full-reference exact match. End-marker tokens are also excluded in a separately
reported content-agreement metric.

An independent verifier reloads all final weights, checks Adam counters/frozenD/
source hashes, reproduces saved generation, and runs additional flow draws and
correct/changed/wrong/blocked old-note controls with the later question unchanged.
Verification finished successfully, with zero additional optimizer updates.

## Completed training results

Each arm finished all 1,088 updates. Total research optimizer updates: 3,264.
Development selection chose four Euler evaluations per output code for flow;
eight and sixteen did not improve development content agreement.

| Arm | Test content-token agreement | Stopped within cap, 34 answers | Mean repeated four-gram fraction | Train + periodic dev/save seconds |
| --- | ---: | ---: | ---: | ---: |
| Raw token control | 32.84% | 10/34 | 35.81% | 416.265 |
| Deterministic code MSE + CE | 3.366% | 0/34 | 70.85% | 224.297 |
| Flow matching + CE | 0.1670% | 0/34 | 0.00% | 258.125 |

All three generated zero full-reference exact matches among the 34 answers.
The cap is 128 rendered tokens and 22 of the 34 references are longer. Exact match is
also too strict for open-ended answers. Nevertheless, inspecting the generated
text shows raw-model repetition and generic answers, deterministic code collapse
to repeated fragments, and incoherent flow text. Flow's low repetition is not
good language quality: varied garbage also does not repeat. No arm is a working
chatbot in this bounded experiment.

These content-agreement percentages use teacher-provided history: all previous
tokens for raw, all previous block codes for vector arms. They are diagnostic,
not an apples-to-apples autoregressive accuracy comparison or a ranking of all
possible valid continuations. Complete answer generation uses each model's own
feedback; no answer length or future code is supplied. Decoder CE of a sampled
vector must not be compared as perplexity against raw token CE.

The recorded training durations include periodic dev and checkpoint saves, but
exclude cache preparation, final solver selection, final tests and generation.
The 34-answer loops took 58.094s raw, 20.203s deterministic and 49.250s flow, with
cached prompt codes. Raw sometimes stops early; vector arms always hit the cap.
These are single-run measurements on different-length, mostly bad outputs and
do not establish a quality-matched end-to-end speed improvement.

The literal-EOS correction made true-code end markers reconstruct correctly,
but neither vector predictor learned reliable generation or stopping. Fixing
the interface was necessary for a meaningful test, not sufficient for language
learning. Falling noisy-target endpoint loss during flow training did not
translate to successful sampling from pure noise. This remains a failed bounded
configuration, not a proof of a 23M capacity ceiling or of all latent-language
models being impossible.

## Independent reload and old-memory interventions

`verification.json` reports passed. All 34 saved answer sequences per arm were
reproduced exactly; both vector arms' saved predicted-code floats also reproduced
exactly. Final/last model tensors agree, every Adam counter is 1,088, every model
has exactly 23,209,216 trainable parameters, and the original decoder tensors
match. Source and protected original file hashes remain unchanged. The verifier
made zero optimizer updates. The trainer and verifier both exited successfully.

Four independent flow draws on the same full 504-dialogue test average 0.1684%
content-token agreement (individual 0.1670%, 0.1740%, 0.1708%, 0.1617%). All four
have zero exact blocks among 36,134 scored blocks each. A single unlucky draw is
therefore not a persuasive explanation for the observed poor samples.

Old-memory controls use the 32 selected test memory dialogues: 27 have a stored
answer and five ask about a missing entity. The entire later question, filler,
and intervening assistant text/codes remain byte-for-byte unchanged. Only the
initial note prefix is changed, replaced with a donor's notes, or removed. The
changed arm gives every case an answer, including adding a fact for originally
missing entities. Wrong-note expected values are derived from the donor's notes;
13 donor cases lack the queried entity. Removing/replacing prefixes changes
length and positions, so these are not pure same-position attention ablations.

| Arm | Correct notes: full value, 27 answerable | Changed notes: updated value, 32 cases | Blocked notes: missing-answer phrase hits, 32 cases |
| --- | ---: | ---: | ---: |
| Raw control | 0/27 | 0/32 | 0/32 |
| Deterministic vectors | 0/27 | 0/32 | 0/32 |
| Flow vectors | 0/27 | 0/32 | 0/32 |

The verifier's broad substring/abstention heuristic reports one raw wrong-note
match. Manual inspection shows `I don't have a notebook in the notebook in the
notebook.[END]`: this is a false positive for useful abstention, not successful
retrieval. We retain the original automatic score and output in the evidence;
do not present that 1/32 as a valid factual answer. All correct-note cases,
including the five missing cases, score zero even under that broad heuristic.
Fact scoring recognizes complete color/container/room values and rejects
multiple contradictory recognized values; it is a narrow synthetic metric,
not a general reading-comprehension evaluation.

Because this fresh raw control also fails, these controls do not establish that
compression alone causes the problem. They establish that neither candidate
passes the required task at this budget. The earlier full-v4 recent-token versus
old-vector contrast is separate evidence about that checkpoint's memory use;
do not merge its training budget or results with this fresh pilot.

For example, the true old note locates Rina's notebook in the green box in the
study. The raw answer is `I'm notebook in the notebook in the notebook.[END]`;
deterministic output is repeated character fragments and flow output is unrelated
fragments. The remaining examples and exact prompts are saved in `results.json`
and `verification.json` rather than selected after removing failures.

Run: `dump/full-answer-literal-flow-20261013`. Trainer:
`tools/train_full_answer_flow.py`; independent verifier:
`tools/verify_literal_flow_comparison.py`. Research goal remains open; the original
full-v4 run remains paused at step 1,001. No new long training is recommended.
