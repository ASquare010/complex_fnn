# Clean-code prediction versus velocity prediction

2026-10-13. The read-only diagnosis, preflight, both bounded training arms and
independent reload/old-fact verification are complete. Neither new arm produces
useful language; no quality winner is established.
Production full-v4 training remains paused at 1,001. Original weights, codec,
data and the previous comparison are protected by SHA hashes.

## Why this experiment

The previous [complete-answer comparison](full_answer_flow_results.md) failed
useful language generation. We checked whether this was merely held-out
generalization or too few integration steps. The diagnosis selects four
dialogues per source from train and dev (68 each, independent fixed seeds),
and up to 16 spread query positions per dialogue: 999 train and 970 dev blocks.
This does not rescore every training target and weights sources differently
from the full evaluation. It uses true prior block codes, not self-fed answers.

| Flow diagnostic | Train content agreement | Dev content agreement |
| --- | ---: | ---: |
| Genuine target code through frozen decoder | 99.9185% | 99.9580% |
| Sample from noise, Euler 4 | 0.2308% | 0.2519% |
| Sample from noise, Euler 16 | 0.1358% | 0.2099% |
| Sample from noise, Euler 64 | 0.1086% | 0.1679% |
| Sample from noise, Heun 32 (64 field calls) | 0.1358% | 0.1539% |
| Teacher endpoint from t=0.5 | 18.4929% | 18.8357% |
| Teacher endpoint from t=0.75 | 93.7678% | 93.1570% |
| Teacher endpoint from t=0.9 | 99.8371% | 99.9160% |

At t=0.9, the noisy input is already 90% of the true normalized target code plus
10% noise. That row is a denoising diagnostic, **not generation**. Even when
all context (including the question) is blocked, t=0.9 yields 99.6062%/99.7061%.
Near-target denoising therefore cannot be evidence of question understanding.
This is normal target corruption during training, not an accidental inference
target leak; the actual free sampler has no target input.

At t=0 the estimated endpoint's normalized MSE is 1.2041 train / 1.1993 dev,
worse than a fixed train-mean code's 0.9903/0.9919 on these queries. The context
does help slightly relative to blocking all inputs, but it does not produce
accurate new codes. More integration steps do not repair the failure. The
deterministic control also has only 7.9973%/7.3048% content agreement on these
selected train/dev positions. This broader run has not demonstrated strong
training fit, unlike the earlier tiny memorization experiment.

Evidence: `dump/full-answer-flow-diagnosis-20261013/diagnosis.json`, script
`tools/diagnose_full_answer_flow.py`. Zero optimizer updates; complete model
fingerprints and protected source/input hashes were unchanged.

## Hypothesis and matched variants

[Training Flow Matching: The Role of Weighting and Parameterization](https://arxiv.org/html/2603.06454v1)
separates what a network predicts from how its errors are weighted across noise
levels. Its image/synthetic results do not establish a universally superior
parameterization, or success with our text codec. They motivate this controlled
test rather than another unstructured model change.

Let `x_t = (1-t)*noise + t*target`, where target is a normalized genuine code.

| Arm | Network output | Regression loss |
| --- | --- | --- |
| Existing control | velocity v | mean square of `v - (target-noise)` |
| New velocity_cleanloss | velocity v | mean square of `x_t + (1-t)*v - target` |
| New data_cleanloss | clean code d | mean square of `d - target` |

The second arm changes weighting: its regression equals `(1-t)^2` times the
control's velocity loss. Comparing the two new arms holds the clean-code loss
fixed while changing output parameterization. Each adds 0.02 CE through the
same frozen original decoder after update 200, on its estimated clean code.

For the clean-code head, sampling uses velocity `(d-x_t)/(1-t)`. Euler steps
never evaluate at t=1; their final evaluation is t=1-1/N. This avoids dividing
by zero without a hidden clipping rule. This is still a stochastic iterative
sampler initialized from noise, not a deterministic replacement or target-fed
inference. Its generated codec vectors are appended directly without re-encoding.

Same 23,209,216 trainable parameters, initialization seed 47, frozen E/D, prepared
literal-EOS data and train-only normalization. Same 4,352 training dialogues,
two shuffled epochs, effective/micro batch 8, 1,088 updates per new arm,
5,200,858 rendered assistant-label exposures. Same batch/noise/time seed streams,
AdamW schedule, clipping, precision and CE onset as the previous control.
Neither predictor nor frozen decoder receives raw token IDs as inputs.

Final development data chooses Euler 4/8/16 by content agreement, then exact
blocks, then lower NFE. All 504 existing test dialogues are scored, followed by
34 complete answers capped at 128 tokens, and independent reload/old-fact
controls. This is a reused exploratory test set, not an untouched final benchmark.
Sampled decoder CE is not token-language-model NLL. A better denoising loss alone
does not pass the generation gate.

## Verification before training

Four CPU tests pass: clean/velocity path integration, no call at singular end
time, both chunked objectives versus naive all-parameter gradients, frozen-decoder
gradients, cached versus full sampling and future-history masking.

Full 23M CUDA preflight includes Adam-sized moment buffers but performs no
optimizer step. Both variants have finite gradients and unchanged parameter
fingerprints. The checked eight-dialogue batch has 2,571 labels/327 queries;
peak allocated memory is 1.402 GiB and reserved memory 1.549 GiB. This is one
batch, not a worst-case or whole-run throughput guarantee. The existing trainer
also checkpoints after its first real update and resumes that saved update;
no research update is discarded.

Code: `tools/data_prediction_flow.py`, `tools/train_data_prediction_flow.py`,
`tools/verify_data_prediction_flow.py`, `tests/test_data_prediction_flow.py`.
Outputs: `dump/data-prediction-flow-20261013`. No production model was edited.

## Completed training and generation

Both arms finished all 1,088 updates (2,176 new optimizer updates total). Each
processed the same two epochs. Final development selection chose Euler 4 for
velocity/clean loss and Euler 16 for the clean-code head.

| Arm | Test content-token agreement | Exact scored blocks / 36,134 | Exact free answers / 34 | Stopped on end marker / 34 |
| --- | ---: | ---: | ---: | ---: |
| Previous raw-token control | 32.8414% | Not the same block factorization | 0 | 10 |
| Previous deterministic code head | 3.3660% | See original report | 0 | 0 |
| Previous velocity / velocity-loss flow | 0.1670% | 0 | 0 | 0 |
| Velocity / clean-code loss | 0.3537% | 0 | 0 | 0 |
| Clean-code head / clean-code loss | 0.7892% | 88 | 0 | 0 |

Content agreement excludes literal end-marker tokens. Exact scored blocks may
include short tails and ending tokens, so 88 blocks are not 88 complete answers.
Next-block evaluation uses true previous codes; free answers append predicted
codes unchanged. All free vector answers reach the 128-token cap. Twenty-two
references exceed that cap, making full exact-answer matching impossible for
those cases; actual generated examples are incoherent rather than merely valid
paraphrases that miss reference wording. The raw control is weak too and is not
a successful general chatbot or memory baseline.

Changing weighting and then output parameterization improves the very low
reference-agreement score in this pilot, but falls far short of useful generation
or the matched raw control. It does not establish that longer training, a
different flow architecture or a different representation cannot work. The
bounded data still contains only about 2.6M distinct assistant target positions,
not the full v4 training exposure.

Training including periodic development checks/checkpoints took 255.297 and
255.500 seconds; preprocessing, final sampler selection/test/answer generation
are excluded. Generating the 34 cached-prompt answers took 48.516 and 163.781
seconds respectively (4 versus 16 denoising steps); encoder preparation is
excluded. There is no quality-adjusted speed win.

Independent verification of velocity/clean loss has finished: all 34 saved token
sequences and predicted vectors reproduce exactly on reload; original frozen
decoder tensors and all Adam step counters match. Four draws give content
agreement 0.3537%, 0.3752%, 0.3618%, 0.3452% (mean 0.3590%). Its 32 old-note
controls give zero successful complete-location/missing-answer responses in
every condition: correct, changed, wrong and blocked. The later question stays
unchanged; note length/positions can change.

The clean-code head also reproduces all 34 saved answers and predicted code
floats exactly, with matching final/last weights, Adam counters and original
frozen decoder tensors. Four draws give 0.7892%, 0.8260%, 0.8178%, 0.7931%
content agreement (mean 0.8065%). Its correct/changed/wrong/blocked old-note
conditions also all score zero. Correct notes include 27 answerable and five
missing cases; changed notes have 32 answerable cases. These models therefore
do not demonstrate factual retrieval on this test. The 20.59% next-block content
agreement within the memory source also includes acknowledgments and fixed
phrases, and must not be read as factual-answer accuracy.

Verifier session 14979 finished successfully with zero optimizer updates and
all protected/source hashes unchanged. `verification.json` contains the
per-case outputs, all four draws, reproduction checks and rescored controls;
`verification-sources.zip` preserves the script. Trainer session 44088 also
finished successfully. No research worker from these sessions remains running.

The old missing-answer heuristic incorrectly accepted a repeated notebook
phrase from the raw control. The new verifier requires an explicit missing
location statement as well as absence of a contradictory location. Rescoring
the saved original outputs with that same rule removes the false positive.
This remains a narrow automatic check, potentially missing valid paraphrases,
not a general answer-quality judge. The recorded generations are also inspected.

## Interpretation and next decision

This experiment tested a specific loss/parameterization change, not whether all
continuous language models are possible. Useful codec reconstruction, good
near-target denoising, and useful free generation are separate measurements.
The first two can be strong while the third fails. Likewise, a model that cannot
generate answers in any memory condition cannot demonstrate successful retrieval
from a small correct-minus-wrong difference.

For practical chat, the existing token-output model remains the stronger
reference. A grounded-reading curriculum and raw/reconstructed-text controls can
isolate whether its input interface or task learning is limiting old-fact use.
Using token output for that diagnostic explicitly relaxes the retained
direct-vector constraint; it is not a replacement made in this experiment.

For direct-vector research, the next distinct architecture hypothesis is a
causal history backbone followed by a small conditional denoising head, jointly
trained inside the fixed parameter budget. Unlike the current noisy query that
passes through all eight layers each refinement step, the history backbone
would produce one condition and the small head would handle repeated refinement.
See the [research review](frozen_codec_research_direction.md) for the primary
image-generation paper motivating this separation and its applicability limits.
This could reduce sampling cost; no language-quality benefit has been tested.

Before a new long run, require useful held-out generated answers, sensitivity
to changed facts with an unchanged question, and stable predicted-code feedback.
Do not choose another long schedule merely because denoising MSE decreases.
The full-v4 production run remains paused; originals remain protected.
