> Historical diagnostic: helper sources are archived in `dump/retired-block-vector-20261013.zip`. Superseded experimental checkpoints have been deleted; results remain. The current trainer scores only new tokens with CE, unlike this diagnostic.

# Frozen decoder and growing-vector diagnostics — 2026-10-13

The user authorized bounded real-language tests of the original frozen decoder
with MSE + CE, then clarified a single active vector replaced at lengths
8,16,24,32,40,48,56,64. No production architecture or existing checkpoint was
changed. The main block-vector run was already paused at step 7,096; it remains
paused. Its last checkpoint and the original codec/encoder/v4 model hashes were
verified unchanged before and after the tests.

## Protocol

Two independent-output pilots (64 tokens and 8 tokens) and one growing-vector
pilot each used 300 updates: 75 MSE-only warm-up updates, 75 updates ramping the
CE weight from zero to one, then joint normalized MSE + CE. Batch32/micro8,
AdamW3e-4,weight_decay.01,gradient clip1,seed47. Only the newly initialized
predictor was trainable. The original downloaded codec decoder was frozen, with
CE gradients allowed through its operations; no added decoder cross-attention.
No tokenizer IDs entered the predictor; it received encoded vectors and, for the
growing version, a learned phase tag indicating the intended prefix length.

Selected one full64 assistant span per distinct document:512training and64dev
examples from immutable v4. Known context was capped at the same last256tokens.
These are long assistant continuations, not a representative whole-corpus chat
evaluation or controlled fact-retrieval benchmark. Sample IDs and full tokens
are recorded in `dump/frozen-vector-real-300/protocol.json`. Fixed-output arms
use18,620,672trainable parameters; growing adds4,096phase parameters, totaling
18,624,768. The predictor remains below the prior hard cap; frozen codec costs
are extra. These are diagnostic models, not matched23M production benchmarks.

Independent-output history uses the arm's chunk length (64 or8). Both arms see
the same source context and examples, but predict different horizons. Growing
history stays64-token chunks (possibly a shorter final context chunk). One active
prefix vector is supplied and replaced, never accumulated. Targets are the
encoder's representations of the full growing target prefix. CE covers all
positions in that prefix; new-eight-token accuracy is reported separately.
Length-specific training means/stds normalize growing targets.

Training supplies genuine encoded prior prefixes. The growing free rollout
instead feeds the model's own last predicted vector directly back, then emits
only newly added positions. That difference is explicitly measured rather than
claimed solved. No fresh text encoding occurs inside its eight-step rollout.

## Results

With **genuine encoder vectors**, the original frozen decoder recovered63/64
8-token examples exactly (511/512tokens,99.80%). At each tested length16..64,
all64examples were exact. This establishes short-prefix codec compatibility on
this sample only; it does not make arbitrary predicted vectors valid codes.

| Fixed-output pilot | Dev full-output CE | Dev token accuracy | Dev first8 accuracy | Dev normalized MSE | Constant training-mean MSE |
|---|---:|---:|---:|---:|---:|
| 64-token prediction | 18.2339 | 0.46% | 2.15% | 1.3352 | 1.0143 |
| 8-token prediction | 14.1253 | 3.32% | 3.32% | 1.5763 | 1.0327 |

Both had0/64exact predicted blocks. The64arm fit training tokens to16.52%
accuracy; the8arm to5.79%. Neither achieved reliable held-out prediction. MSE
at update75 was1.2113/0.9759 on development; after introducing CE it rose to the
values above. This shows a loss trade-off in these runs, not proof that every
loss weighting or longer run must fail. The64and8whole-output CE values have
different prediction horizons and should not be ranked as equivalent LM scores.

The growing replacement pilot reached:

- Teacher-forced dev CE13.5819 and normalized MSE0.7628.
- Accuracy across all prefix positions7.19%; **new8positions1.95%**.
- Direct-feedback rollout on16held-out contexts: **0.59% newly emitted token
  accuracy** across1,024tokens.
- When a larger vector was decoded, **80.44% of previously emitted positions
  checked disagreed with the committed output**. This is a count over repeated
  old-position checks at each phase, not a fraction of unique answer tokens.

Rollouts have a fixed64-token horizon for diagnosis; they do not implement EOS
stopping/chat turn handling. The independent8arm also records exploratory
teacher/direct/re-encoded rollouts, but those grow beyond the training history
length and are not used for the replacement-vector conclusion.

## Interpretation and next decision

The clarified architecture is mechanically feasible: retain old64-token memory,
replace one active vector, and supply its known length8..64. Frozen codec lengths
work on this sample, so encoder retraining is not a prerequisite for a pilot.

However, replacement is not an automatic information-preserving operation. It
changes all256coordinates. MSE toward the correct complete prefix and CE on its
tokens did not ensure that the generated replacement preserved earlier emitted
text in this run. A low-ish aggregate vector MSE is not evidence of accurate
decoding or stable feedback.

These limited fresh-model experiments do not establish impossibility or a model
size ceiling. They also do not justify another long frozen-decoder run yet. The
next diagnostic would need to test explicit committed-prefix preservation and
training with predicted active vectors, separately from teacher-forced target
vectors. Those changes have **not** been implemented or trained here. An unchanged
target-prefix CE already supervises old correct tokens; merely adding the same
term again would only reweight it, not resolve mismatched generated histories.

Evidence and reproducible scripts:

- `tools/test_frozen_vector_prediction.py`, `dump/frozen-vector-real-300/`.
- `tools/test_growing_vector.py`, `dump/growing-vector-real-300/`.
- Each evidence directory contains protocol/results and a source archive. Both
  scripts refuse to overwrite an existing output directory and use the shared
  training-worker lock. No new production training command was introduced.

The growing script had a syntax error before any execution; it was fixed before
the single300-update run. No failed optimizer run was silently restarted.
