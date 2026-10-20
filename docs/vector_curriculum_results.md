# Frozen-codec composition and memory curriculum — 2026-10-13

The predictor can construct genuinely new answer vectors when the training task
covers enough word combinations. A shared-weight, two-pass version then learns
to select a relevant stored vector and construct an answer from it. This is a
controlled memory prototype, not a general language model or completed chatbot.
The original encoder/decoder remain frozen; there are exactly **23,209,216
trainable parameters**, reused across both passes. Frozen codec costs are extra.

## Composition coverage

Thirty-two single-token words in two answer positions create 1,024 combinations.
Each answer is eight tokens: `The access words are water and room.` The source
record uses different wording and is padded to 64 tokens. Two record layouts,
40 training names and 16 question templates are used. Eighteen names and four
question templates are held out. A deterministic split excludes 256 entire word
pairs from training. All individual words occur in both positions during training.

The sparse arm trains on 96 pairs; the expanded arm on 768. Both use the same
fresh initialization, 2,000 updates, batch 64, AdamW 3e-4 and normalized MSE,
adding 0.02 token CE after 200 updates. Normalization is computed separately
from each arm's training codes, so the change includes training-derived scaling.
No unseen target codes enter the predictor, loss or normalization during training.

In this diagnostic, the correct source record is supplied. On a fresh 512-case
test of unseen pairs, sparse training reaches **19.73% exact answers** and the
expanded curriculum reaches **100%**. The expanded arm follows changed values
100%; wrong records give 0.78% and blocked memory 0%. New pairs with new names
also reach 100%; new wording 99.61%. Names/wording are largely unnecessary when
the correct record is supplied, so these are not retrieval-understanding scores.

The frozen decoder reconstructs all 1,024 answer codes exactly. An independent
256-record check reconstructs all 16,384 input tokens, including all 512 variable
fact tokens. Thus neither the answer nor the facts were lost in these codec tests.

## Selection and construction

The expanded model was continued for 2,000 updates with one, two, four, then eight
records. Actual eight-record generation reached 33.20% on fresh unseen pairs.
Adding a 0.25 loss on the final attention layer's source-index score did not fix
one-pass generation: 18.36% on the same fresh episodes. It did learn selection:
90.43% correct record indices, versus 11.33% for the plain model's corresponding
score. Attention accuracy alone was insufficient to construct the answer.

The next variant reuses the same weights twice:

1. Encoded question + eight old vectors → learned source-index scores.
2. Selected original vector + encoded question → newly predicted answer vector.
3. Original frozen decoder → eight output tokens.

The output code is newly predicted, not a copy of the selected 64-token record.
The first pass uses phase embedding 2, the second phase 1. No new parameters,
token inputs, decoder training or target-code lookup are introduced. The true
source index is a training label only. Inference uses the model's argmax choice.
When memory is blocked, the selected vector stays blocked in the second pass.

This continuation starts from the attention-supervised checkpoint and runs 2,000
additional updates at 1e-4. The objective is normalized MSE + 0.02 decoded token
CE + 0.25 selection CE. Composition teacher-forces the selected source during
training. The checkpoint has 6,000 updates in its lineage, versus 4,000 for the
one-pass comparisons, and a different final LR. The gain cannot be attributed
solely to pass count under an equal-training-budget claim.

Independent reload, fresh 512 cases per family:

| Condition | Unseen pairs | Unseen pairs + new names | Unseen pairs + new wording |
|---|---:|---:|---:|
| Correct memory, exact full answer | 91.60% | 72.07% | 4.30% |
| Change queried old values; follow new answer | 91.21% | 71.29% | 3.71% |
| Shuffle record order | 92.58% | 72.66% | 4.49% |
| Wrong memory | 0.39% | 0.39% | 0% |
| Blocked memory | 0% | 0% | 0% |
| Remove question | 9.77% | 11.52% | 7.42% |

For familiar wording, source selection is 93.75% and composition given the true
record is 97.07%. With unseen wording, those fall to 13.67% and 7.81% respectively.
The question affects both stages, including construction, which should not need
its exact wording for this fixed-format task. Wording robustness is a clear next
problem; passing the familiar-template tests does not establish general memory QA.

Local batch-32 inference, excluding encoding: one predictor pass + decoding
21.0 ms; two passes + decoding 41.0 ms. This retains parameter count, not identical
compute. No end-to-end speed advantage over the token-memory model is established.

## Verification and scope

`verify_vector_composition_curriculum.py` re-encodes records, checks fact tokens,
training-only normalization and fresh tests. `verify_two_pass_vector_memory.py`
checks frozen decoder tensors against the original codec, exact parameter count,
blocked-memory invariance, and generation invariance after corrupting the entire
answer-code/label table. Those tables are evaluation labels, not inference inputs.
All protected checkpoint and codec hashes remain unchanged.

Evidence directories: `dump/vector-composition-curriculum-20261013`,
`dump/vector-attention-curriculum-20261013`, and
`dump/two-pass-vector-memory-20261013`. They retain per-case outputs, source
archives and diagnostic checkpoints. No full-corpus run was resumed or original
checkpoint overwritten. No DB or production trainer change was made.

Remaining requirements before a broad training recommendation include unseen
question/record layouts, missing records, combining multiple records, free-form
answers, probabilistic continuation, EOS and growing-vector feedback. This result
shows a learnable controlled operation in frozen codec space; it does not show
that the full language-generation objective has been achieved.

## Probabilistic control: sample a plan, then construct the vector

A separate experiment supplies one correct record and accepts either ordering
of its two words. Both resulting eight-token answers are valid. Training pairs
remain the same 768; evaluation holds out 240 non-identical pairs. Reversing a
held-out pair also remains held out. All arms start from the expanded composition
checkpoint and train for 2,000 updates with identical sampled records and target
orders, batch 64, AdamW 3e-4, 200 MSE-only updates then MSE + 0.02 CE.

The first arm predicts one code without knowing which valid order was sampled
as its training label. The second uses seven denoising passes. The third receives
a one-bit plan specifying order, represented by existing phase embeddings 1/2.
At inference this bit is sampled independently with probability 1/2. It contains
no factual values and is not read from an answer label. The predictor must still
read both values from the record and construct a new codec vector. Parameter
count remains 23,209,216. The plan has hand-defined meaning and distribution;
this is additional structured conditioning, not a learned language-planning prior.

Independent reload on 256 fresh records, eight samples each:

| Method | Valid full-answer samples | Both valid orders observed per record | Batch-32 sample + decode |
|---|---:|---:|---:|
| Unconditioned one-pass regression | 0% | 0% | 21.5 ms |
| Seven-step denoising | 10.84% | 3.91% | 135.8 ms |
| Sampled order plan + one vector prediction | 100% | 99.61% | 20.5 ms |

The sampled-plan arm produces 0.39% valid answers with wrong memory and 0% with
blocked memory. It follows its sampled plan on 100% of correct-memory samples.
A separate 512-case test likewise has 100% valid samples; 256 cases with new
names also have 100%. The source record is supplied in all these controls, so
they do not measure retrieval or question understanding.

Example samples from a single fresh record:

- `The access words are tree and road.`
- `The access words are road and tree.`

Both codes are newly predicted; inference never looks them up in the target-code
table. Reload verification mutates that entire table without changing generation,
checks blocked-memory invariance, different seeded samples and original frozen
decoder tensors. Source/target pair splits are disjoint. All reference hashes
remain unchanged. Development accuracy fluctuated during training; reported final
and fresh scores come from the saved 2,000-update checkpoint, not a selected peak.

Evidence: `dump/ambiguous-vector-generation-20261013`,
`dump/sampled-plan-vector-20261013`, and
`tools/verify_stochastic_vector_controls.py`. These scripts and their source
archives reproduce the bounded comparison; they are not full-corpus trainers.

The experiments above use **separate checkpoints**. A subsequent
[shared integration](shared_vector_planner_results.md) combines retrieval and a
learned contextual plan prior within one 23.21M model. It reaches 44.73% fresh
exact answers with correct memory, 0.39% with wrong memory, and 0% blocked;
unfamiliar question phrasing remains weak at 14.36%. Arbitrary continuations
require more than choosing between two predefined orders. Wording robustness,
multi-record reasoning and general language generation remain open.
