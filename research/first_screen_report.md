# First controlled FFN screen - completed 2026-09-06

## Finding
The tested bounded residual cubic activations do not improve parameter efficiency.
At approximately 75% fewer FFN weights, they have 3.2-3.4% worse validation NLL
than full references, exceeding the frozen 1% tolerance. Shared curves are
indistinguishable from narrow GELU in this small paired-seed comparison; grouped
curves are slightly worse in all three seeds. Neither earns longer training in
its current form. This is a negative result for this configuration, not for every
learned-activation method. The research target remains unmet.

## Protocol
Five variants, three seeds (17,29,43): 15 completed runs, 409,600 tokens each,
6,144,000 training tokens total. Four layers, width 192, six heads, context 128,
BF16 autocast, FP32 master weights, identical common tensors per seed, same data
sampling stream, same AdamW schedule, no attention changes. The frozen prefix
cache has 12,000 training stories / 2,457,760 BPE tokens and 1,000 validation
stories. Evaluate 32,768 fixed targets; no best-checkpoint selection.
See [full protocol](experimental_plan.md) and [data manifest](../results/data/manifest.json).

## Results
| Variant | FFN weights | Total weights | NLL mean +/- SD | Train tokens/s | Forward tokens/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| gelu | 1,179,648 | 2,557,632 | 4.0846 +/- 0.0068 | 26,816 | 83,283 | 245.5 |
| swiglu | 1,179,648 | 2,557,632 | 4.0850 +/- 0.0132 | 24,510 | 76,685 | 255.8 |
| gelu_narrow | 294,912 | 1,672,896 | 4.2160 +/- 0.0288 | 27,058 | 84,194 | 214.6 |
| bezier_shared | 294,920 | 1,672,904 | 4.2164 +/- 0.0235 | 22,809 | 67,433 | 251.8 |
| bezier_grouped | 294,976 | 1,672,960 | 4.2221 +/- 0.0294 | 23,266 | 72,104 | 251.8 |

NLL uncertainty is sample SD over seeds. Throughputs and memory are means.
Inference is full-sequence forward throughput, not cached generation. The curve
variants retain only about 65.4% of total model weights: a 75% FFN reduction is
not a 75% whole-model reduction. Runtime differences include small-workload
launch overhead and laptop variability; raw timing repetitions are saved.

## Paired checks
Shared minus narrow GELU: +0.000369 NLL; exploratory paired 95% Student-t interval
[-0.02418, +0.02492]. Grouped minus narrow: +0.006064 NLL, interval
[+0.000225, +0.011904]. Three-seed normality is unverified; these intervals are
unadjusted for multiple comparisons. The tiny grouped difference is not a
scientific discovery. Neither method improves over the larger full references.
Complete seed values, best/worst and comparisons are in
[seed_summary.json](../results/seed_summary.json).

## Mechanism and proofs
A static same-coordinate cubic bank collapses exactly to one cubic; adding bank
members cannot expand that function class. The implemented deformation has
magnitude <=.375 and derivative <=.75, and starts exactly at GELU with live
control gradients. These are elementary structural results, not claimed novelty
or proofs of language-model performance. See [derivations](theory.md).

Seed 17 curve deviations remain below .004 over [-6,6], with unsaturated controls
and little inter-group separation. All final layer activation diagnostics are
finite. Narrow/cubic models clip gradients more frequently than full controls;
the actual clipping rates and pre-clip norms remain in the raw metrics. Added
FP32 curve intermediates increase allocated memory over narrow GELU.

Fifteen separate initial-gradient probes span 4, 12 and 24 layers at width 96;
all have finite gradients. These are initialization probes, not evidence that
trained deep models avoid gradient failure. [Probe records](../results/stability_initialization.json).

## Complex-function screen
42 exploratory trials cover seven target families and six controls, 300 FP32
steps each with independent held-out inputs. Full GELU/SwiGLU have 385/386 weights;
narrow GELU/shared cubic have 97/99; budget GELU/grouped cubic have 103/105.
All mismatches are explicit. On the multiplicative target, SwiGLU obtains
normalized held-out MSE .00548 versus .44365 for full GELU and .53377 for grouped
cubic. This supports investigating compressed multiplicative features. Parity
and oscillatory errors remain high: no architecture here solves complex patterns
universally. One seed and finite optimization are insufficient for broad claims.
Function screening records aggregate errors and gradients; unlike the LM runs,
this first toy screen did not save model checkpoints or peak memory. It supports
branch selection, not efficiency acceptance. [All trial records](../results/functions_s17.json).

## Reproduction and artifacts
Run uv sync and uv run pytest -q. Reconstruct data with
uv run python -m src.core.cli prepare --tokenizer results/data/tokenizer.json.
Then run the five variants via src.core.cli train --steps 200 --seed SEED for
17,29,43. Use --config configs/screening.json for the frozen settings.
Each LM directory retains checkpoint.pt, source.zip, config.json, metrics.json,
history.jsonl and initial/final diagnostics. Heavy checkpoints/source ZIPs are
local artifacts excluded from git. Source hashes include the dirty worktree;
baseline source predates candidate registration but decoder/training semantics
are unchanged. CUDA seeding is controlled; bitwise determinism is not promised.
Generate tables with src.core.cli report and figures with src.core.plots.

## Decision and remaining uncertainty
ELIMINATE these cubic settings from promotion. Preserve the proof and bounded
residual implementation. BRANCH to structured wide GELU/SwiGLU to test nonlinear
feature width per parameter. These are established ideas used as serious controls,
not renamed inventions. A useful result must still pass equal tuning, more tokens,
compute matching, larger models, broader data and mechanism ablations. No SOTA
or breakthrough claim is currently supported.

Fresh-cache reconstruction was verified: all five raw-text, token-array and
tokenizer hashes exactly match the first experiments.
