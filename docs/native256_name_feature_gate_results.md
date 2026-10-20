# Name-key feature gate

Both1024-update LINEAR component probes and independent verification completed.
No LM, encoder, decoder, old checkpoint or cache was trained or changed. Run:
`dump/native256-name-feature-gate-20261018`. Two CPU tests passed. All source
labels, saved predictions, oracle positions and optimizer steps were rederived.

The query's entity class is supplied as an ORACLE. These are source-key
learnability measurements, not question extraction, free QA or final quality.
Expanded feature banks are diagnostic files only; none enters production
attention/KV. The full23M/two-original-epoch/64:1 goal remains unmet.

| Oracle name-position selection | Linear unpack125 | Trained nonlinear256 |
|---|---:|---:|
| Familiar, known names |489/512 |512/512 |
| Familiar all-four units |114/128 |128/128 |
| Held answer IDs, known names |469/512 |512/512 |
| Held answer IDs all-four |103/128 |128/128 |
| Existing wording cohort |481/512 |512/512 |
| Held source positions |335/512 |512/512 |
| Held-position all-four |48/128 |128/128 |
| Missing cohort, including positives |272/512 |460/512 |
| Actual training cases, including missing |3624/4096 |4024/4096 |

Seven entity classes plus OTHER, balanced512-point batches,1024updates, AdamW
LR.02/no decay. Per-channel affine standardization uses training features only;
weights are folded back to raw coordinates for independent verification. The
resulting oracle score is name-class minus OTHER logit, so it remains linear
in the raw features. NULL uses a fixed zero margin threshold; absent-entity
performance is not optimized separately and should not be interpreted as a
fundamental representational limit. Existing evaluation source blocks are reused
from prior diagnostics; these are not new independent samples.

Linear training macro class accuracy rises from94.54% at128 to99.31% at1024.
The nonlinear probe reaches100%. High token classification does not guarantee
perfect64-way address selection or absence calibration, as the oracle table
shows. Finite optimization does not establish linear impossibility.

The current linear keys permit considerably better ordinary-name selection than
the failed LM learned. This justifies explicit name-anchor supervision first.
The trained nonlinear features are also more stable across source positions,
supporting a later architectural comparison if direct supervision still fails.
That comparison must preserve one global memory/KV position per64 tokens and
measure total compute, rather than retaining a64-feature cache silently.
