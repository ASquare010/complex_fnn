# Name-gradient isolation512: partial language recovery, no broad win

Completed and independently verified18 October2026. The overall goal remains
**ACTIVE / UNMET**. This diagnostic is complete; do not extend its checkpoint.

Stopping direct NAME-loss gradients at the LM hidden states recovers much of the
previous language loss while retaining some familiar binding. It does not
preserve language against the no-NAME control, and it does not fix new names,
positions, missing information or combined changes.

## Controlled comparison

One new `isolated_name` arm uses the exact same fresh seed47 initial tensors,
frozen encoder/packing/word decoder, fresh Adam and immutable data/order/schedule
as the completed name-anchor pilot. Its only intervention is `query=h.detach()`
for NAME scoring. Controller query/gate gradients remain active; ordinary LM,
answer and value-position gradients remain unchanged. The existing two saved
comparators are reused without training or modification.

All three readers have23,209,216 stored parameters, including3,981,312 frozen
decoder parameters;19,227,904 parameters are trainable. The encoder is separate.
Global attention/KV still has one position per64 old tokens, raw64 and unchanged
inference. This is512 updates/two new QA-data passes, **not two original corpus
epochs**. Each arm has1,872,347 LM labels,16,896 QA labels seen,12,660 answer/value
labels actually trained after warmup, and8,192 NAME queries. The isolated/shared
arms train NAME; the control does not.

## Strict free answers

Answers must contain the correct value and EOS. All-four requires both questions
against both swapped memories, with identical query template/phase and variable
source record order. Held answer IDs are held from QA training, not from the
frozen codec or original LM corpus.

| Held group | No-NAME control | Shared NAME | Isolated NAME |
|---|---:|---:|---:|
| Familiar exact |266/512|500/512|407/512|
| Familiar all-four |4/128|120/128|68/128|
| QA-held value IDs exact |264/512|484/512|402/512|
| QA-held value IDs all-four |10/128|108/128|68/128|
| New names exact |252/512|129/512|119/512|
| New names all-four |3/128|4/128|2/128|
| New wording exact |285/512|498/512|411/512|
| New wording all-four |13/128|120/128|72/128|
| New positions exact |251/512|211/512|188/512|
| New positions all-four |2/128|21/128|11/128|
| Combined changes exact |216/512|0/512|5/512|
| Combined changes all-four |0/128|0/128|0/128|

The isolated arm's missing cohort has212/256 positive and3/256 absent exact
answers; no all-four unit succeeds. Its actual positive training fit is105/128,
17/32 all-four; missing training fit has60/64 positive and0/64 absent answers.
Both source orders retain familiar reading (210/256 and197/256), with36/64 and
32/64 all-four units. Blocked/shuffled familiar controls each score0/64 exact.
All historical notebook facts0/103, changed facts0/32 and compressed colors0/8
remain unsuccessful. All312 historical outputs were retained.

## Original-language evaluation

Full7,831-development-segment set,1,859,388 original assistant labels:

| Model/head | Correct | Accuracy | CE |
|---|---:|---:|---:|
| No-NAME control, mixture |365493|19.656629%|5.189927408|
| Shared NAME, mixture |186830|10.047930%|5.941782259|
| Isolated NAME, mixture |306459|16.481713%|5.309317624|
| Isolated NAME, generator only |308146|16.572442%|5.306840690|

Isolated minus shared: **+6.433784pp**, paired-document95% bootstrap interval
[6.304229,6.566021]. Isolated minus no-NAME control: **-3.174916pp**, interval
[-3.257826,-3.093781]. Disabling copying restores only.090729pp in the isolated
arm, interval[.078742,.105351]. These intervals describe fixed checkpoints, not
training-seed uncertainty. Fresh512-update accuracy is not comparable as a final
model to the original48,762-update/two-epoch baseline.

Direct NAME gradients into the backbone are therefore one harmful interaction
under this recipe. Their removal is insufficient. NAME still updates the
controller used by language prediction, and all losses share global clipping and
Adam. Saved preclip norms illustrate that remaining coupling:

| Arm | Median warmup norm | Median later norm |
|---|---:|---:|
| No-NAME |.646694|1.601847|
| Shared NAME |21.917121|4.200008|
| Isolated NAME |30.195894|9.184114|

These are descriptive medians from the512 saved training rows. They do not
isolate clipping as a cause, establish convergence, or imply an equivalent
reduction in Adam's actual parameter updates. No additional clipping experiment
or automatic continuation is justified by these numbers alone.

## Verification and next implementation

Four new CPU tests and the full-size zero-update preflight verify independent
joint gradients, active controller/no direct NAME-core gradients, ordinary
inference parity, future isolation, microbatch gradients and exact serialized
Adam resume. GPU gradient discrepancy was1.12e-8 with exact unchanged initial
weights and empty Adam. Three paired body samples were approximately.563s for
both objectives, excluding Adam/cache/evaluation/save; no speed win is claimed.

The first actual update and all512 orders/LR/cursors/labels/Adam steps/frozen
decoder were independently verified. Independent results rederived saved case,
source-order, name, missing and control metrics, replayed36 CPU public generation
cases, reaggregated full mixture/generator language and historical results, and
recomputed paired document intervals. All2,005 protected earlier files remain
unchanged. Trainer98761, training audit39154, evaluator37108 and independent
verifier81591 all finished successfully. No worker remains.

Artifacts: `dump/native256-name-isolation-pilot-20261018`. Checkpoint SHA256:
`a4d17e2df70b277aa05526562c629a93be15b3696062a6f3547a89408a8e5782`.
Protocol: [prospective isolation recipe](native256_name_isolation_protocol.md).

A separate untrained lexical LM now uses exact-ID residuals and shared LM
embeddings to avoid the old codec-key alignment/frozen-word-decoder path. Its
full budget and CPU integration tests pass. This is a new implementation, not
an adopted solution; GPU/cache/math verification and complete timing must pass
before any lexical training. See [implementation and outstanding gates](native256_lexical_lm_results.md).
The full fresh23M/two-original-epoch/64:1 broad quality objective remains unmet.
