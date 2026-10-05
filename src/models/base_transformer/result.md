# Dense baseline and controls: results

## Current compact FFN confirmation

The language comparison at width512/four layers completed all paired seeds
211/307/401, with 2000 updates and full last-checkpoint validation on both corpora.
These are language NLL results; the historical synthetic task screen below is a
separate comparison and does not enter the language ranking.

| Dense recipe | FFN weights | TinyStories mean NLL | WikiText mean NLL |
| --- | ---: | ---: | ---: |
| Full native SwiGLU h1376 | 8,454,144 | 2.542837 | 4.220415 |
| Full GELU h2064 | 8,454,144 | 2.589529 | 4.254606 |
| Compact SwiGLU h384 | 2,359,296 | 2.607348 | 4.282197 |
| Compact SwiGLU h346 | 2,125,824 | 2.614868 | 4.287453 |
| Compact GELU h576 | 2,359,296 | 2.617236 | 4.301414 |
| Compact GELU h518 | 2,121,728 | 2.625608 | 4.314461 |

Full SwiGLU has the best mean language loss in this 2000-update comparison.
Compact SwiGLU h384 is the strongest
ordinary compact control in the registered combined comparison. Curve-Wide has
lower mean loss than every compact dense control on both corpora, with fewer FFN
weights than h384. The independent seed509/8000-update evaluation is complete:
full SwiGLU test NLL is 1.974934/4.022206 on TinyStories/WikiText, compact h384
is 2.024020/3.971140 and Curve-Wide is 2.020291/3.966762. Full SwiGLU retains
the TinyStories lead but trails both compact recipes on WikiText at this longer
single-seed budget. Variable resource timings do not support a reliable speedup claim.

[Every confirmation run](../../../docs/compact_refinement_confirmation_result.md) /
[paired analysis](../../../docs/compact_refinement_confirmation_analysis.md) /
[independent evaluation](../../../docs/compact_refinement_independent_result.md) /
[resource limitations](../../../docs/compact_refinement_resource_analysis.md).

## Historical rewrite-v1 task screen

Study: `rewrite-v1`. Measured 2026-10-03 (America/Los_Angeles).
Status: completed single-seed development screen; no confirmed research winner.

## What was established

The rewritten framework executes and trains this model. All eight selected
one/two-pass variants completed 2,000 updates, and all 11 configured study
variants passed GPU execution/memory checks. These are fresh measurements.
Generated-answer performance remains weak; the screen establishes an initial
task baseline, not a strong reasoning model or a language-quality result.

## Fixed comparison recipe

* Hardware: NVIDIA GeForce RTX 4070 Laptop GPU; PyTorch 2.14.0+cu132; BF16 compute.
* Backbone: width 192, four layers, six heads, full causal attention, context 256, vocabulary budget 4,096.
* Training: batch eight, 2,000 updates, seed 101, AdamW at 0.0006, weight decay 0.1, 200 warmup updates.
* Data: `dump/data/tasks-v1`; 11,775 training and 569 validation examples after deduplication; answer-only loss.
* Evaluation: last checkpoint, complete validation split, greedy generated-answer exact match. Test/OOD remain unscored.
* Budget: 4,096,000 input positions including padding; 42,851 supervised answer targets per model.

Shared backbone tensors and sampled training examples match across runs.
FFN initialization follows the current implementations: normal dense weights
and orthogonal structured factors. The comparison is between these implemented
recipes, not an isolated initialization ablation. Periodic validation during
training uses 32 batches; the full scores below use all 569 examples.

## Framework checks

PASS: parameter counts, structured projection math and gradients, causal
behavior, shared backbone initialization, task oracles, loss masking, split
integrity, tokenizer isolation, save/load, exact CPU resume, held-out gates,
and tiny BF16 training in all three model packages. All 11 variants passed
an additional 25-step task-batch GPU check with finite loss and gradients.
All fit below the 6 GiB reserved-memory limit; the largest preflight peak was
636 MiB (four-pass BlockShuffle). These checks do not establish learning quality.

## Full-validation comparison

Lower NLL and composition error are better. Composition error gives equal
weight to chain and arithmetic error, pooling difficulty >=2 within each task.

| Variant | Total / FFN parameters | NLL | Lookup accuracy | Composition error |
| --- | ---: | ---: | ---: | ---: |
| full-swiglu | 2,557,632 / 1,179,648 | 1.522579 | 5.50% | 87.79% |
| full-gelu | 2,557,632 / 1,179,648 | 1.521821 | 5.50% | 89.36% |
| narrow-swiglu | 1,728,192 / 350,208 | 1.495532 | 9.00% | 86.71% |
| narrow-gelu | 1,728,192 / 350,208 | 1.518696 | 5.50% | 88.96% |
| blockshuffle | 1,728,192 / 350,208 | 1.401092 | 20.00% | 81.96% |
| blockshuffle-loop2 | 1,728,192 / 350,208 | 1.455769 | 10.50% | 85.75% |
| narrow-loop2 | 1,728,192 / 350,208 | 1.510771 | 2.50% | 88.14% |
| dense-compute2 | 2,078,400 / 700,416 | 1.552270 | 0.50% | 88.75% |

## Resource measurements for this model family

| Variant | Update seconds | Training-loop wall seconds | Padded input positions/s | Peak allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| full-swiglu | 116.14 | 127.01 | 35,269 | 251.82 / 294.00 |
| full-gelu | 99.59 | 109.23 | 41,130 | 241.20 / 270.00 |
| narrow-swiglu | 110.83 | 120.94 | 36,958 | 220.12 / 252.00 |
| narrow-gelu | 103.68 | 114.11 | 39,505 | 214.75 / 250.00 |
| narrow-loop2 | 144.41 | 156.57 | 28,364 | 247.03 / 288.00 |
| dense-compute2 | 113.38 | 124.21 | 36,128 | 233.98 / 270.00 |

Update time includes startup/warmup and data batching. Training-loop wall time
also includes periodic validation/checkpointing, but excludes model setup and
the later full-validation/generation pass. Throughput counts padding; it is
not useful-answer throughput. Memory is PyTorch allocation, not device-wide
peak usage. These single, fixed-order runs do not qualify a speed advantage.

## Generated answers by task and difficulty

| Variant | Task / difficulty | Correct / examples | Accuracy |
| --- | --- | ---: | ---: |
| full-swiglu | arithmetic/difficulty_1 | 6 / 29 | 20.69% |
| full-swiglu | arithmetic/difficulty_2 | 19 / 70 | 27.14% |
| full-swiglu | arithmetic/difficulty_3 | 11 / 70 | 15.71% |
| full-swiglu | chain/difficulty_2 | 3 / 95 | 3.16% |
| full-swiglu | chain/difficulty_3 | 3 / 105 | 2.86% |
| full-swiglu | lookup/difficulty_1 | 11 / 200 | 5.50% |
| full-gelu | arithmetic/difficulty_1 | 7 / 29 | 24.14% |
| full-gelu | arithmetic/difficulty_2 | 17 / 70 | 24.29% |
| full-gelu | arithmetic/difficulty_3 | 10 / 70 | 14.29% |
| full-gelu | chain/difficulty_2 | 2 / 95 | 2.11% |
| full-gelu | chain/difficulty_3 | 2 / 105 | 1.90% |
| full-gelu | lookup/difficulty_1 | 11 / 200 | 5.50% |
| narrow-swiglu | arithmetic/difficulty_1 | 7 / 29 | 24.14% |
| narrow-swiglu | arithmetic/difficulty_2 | 16 / 70 | 22.86% |
| narrow-swiglu | arithmetic/difficulty_3 | 10 / 70 | 14.29% |
| narrow-swiglu | chain/difficulty_2 | 11 / 95 | 11.58% |
| narrow-swiglu | chain/difficulty_3 | 5 / 105 | 4.76% |
| narrow-swiglu | lookup/difficulty_1 | 18 / 200 | 9.00% |
| narrow-gelu | arithmetic/difficulty_1 | 8 / 29 | 27.59% |
| narrow-gelu | arithmetic/difficulty_2 | 16 / 70 | 22.86% |
| narrow-gelu | arithmetic/difficulty_3 | 10 / 70 | 14.29% |
| narrow-gelu | chain/difficulty_2 | 3 / 95 | 3.16% |
| narrow-gelu | chain/difficulty_3 | 4 / 105 | 3.81% |
| narrow-gelu | lookup/difficulty_1 | 11 / 200 | 5.50% |
| narrow-loop2 | arithmetic/difficulty_1 | 10 / 29 | 34.48% |
| narrow-loop2 | arithmetic/difficulty_2 | 17 / 70 | 24.29% |
| narrow-loop2 | arithmetic/difficulty_3 | 12 / 70 | 17.14% |
| narrow-loop2 | chain/difficulty_2 | 5 / 95 | 5.26% |
| narrow-loop2 | chain/difficulty_3 | 1 / 105 | 0.95% |
| narrow-loop2 | lookup/difficulty_1 | 5 / 200 | 2.50% |
| dense-compute2 | arithmetic/difficulty_1 | 6 / 29 | 20.69% |
| dense-compute2 | arithmetic/difficulty_2 | 18 / 70 | 25.71% |
| dense-compute2 | arithmetic/difficulty_3 | 10 / 70 | 14.29% |
| dense-compute2 | chain/difficulty_2 | 5 / 95 | 5.26% |
| dense-compute2 | chain/difficulty_3 | 0 / 105 | 0.00% |
| dense-compute2 | lookup/difficulty_1 | 1 / 200 | 0.50% |

## Interpretation and next decision

The lowest full-dense NLL in this screen is **full-gelu** (1.521821);
the lowest single-pass compact-dense NLL is **narrow-swiglu** (1.495532).
These observations use one rate and seed, not equal multi-rate tuning.

Keep full SwiGLU as the fixed primary reference and full GELU as the
equal-parameter activation control. Narrow SwiGLU is the strongest compact
single-pass control by validation NLL here. All dense controls still show
weak generated-answer accuracy; tune them fairly before attributing a
structured advantage to a new computational mechanism.

* full-swiglu: NLL change vs full-gelu **+0.050%**; vs narrow-swiglu **+1.808%** (negative is better).
* full-gelu: NLL change vs full-gelu **+0.000%**; vs narrow-swiglu **+1.758%** (negative is better).
* narrow-swiglu: NLL change vs full-gelu **-1.727%**; vs narrow-swiglu **+0.000%** (negative is better).
* narrow-gelu: NLL change vs full-gelu **-0.205%**; vs narrow-swiglu **+1.549%** (negative is better).
* narrow-loop2: NLL change vs full-gelu **-0.726%**; vs narrow-swiglu **+1.019%** (negative is better).
* dense-compute2: NLL change vs full-gelu **+2.001%**; vs narrow-swiglu **+3.794%** (negative is better).

Retain this baseline and its task breakdown. No model is promoted to a confirmed
winner. Next, qualify learning with equal rate/seed tuning before the fresh-seed
confirmation and language experiments specified in Step 1. TinyStories and
WikiText were not trained here. Repeated, alternated timing and quality-matched
comparisons are still required for a speed claim. No inference benchmark was run.

The four-pass structured/dense variants and `dense-compute4` received execution
checks only; their 2,000-step learning studies remain deferred.

## Evidence and reproduction

* [Checks and all-variant preflight measurements](../../../records/rewrite-v1-checks.json)
* [full-swiglu complete compact record](../../../records/rewrite-v1-full-swiglu.json)
  — raw run: `dump\runs\20261004T023351Z-7d8811ad` (configuration, source snapshot, metrics and checkpoints).
* [full-gelu complete compact record](../../../records/rewrite-v1-full-gelu.json)
  — raw run: `dump\runs\20261004T023637Z-3dd62408` (configuration, source snapshot, metrics and checkpoints).
* [narrow-swiglu complete compact record](../../../records/rewrite-v1-narrow-swiglu.json)
  — raw run: `dump\runs\20261004T023907Z-7114d466` (configuration, source snapshot, metrics and checkpoints).
* [narrow-gelu complete compact record](../../../records/rewrite-v1-narrow-gelu.json)
  — raw run: `dump\runs\20261004T024132Z-4a9beae1` (configuration, source snapshot, metrics and checkpoints).
* [narrow-loop2 complete compact record](../../../records/rewrite-v1-narrow-loop2.json)
  — raw run: `dump\runs\20261004T025245Z-4b79a140` (configuration, source snapshot, metrics and checkpoints).
* [dense-compute2 complete compact record](../../../records/rewrite-v1-dense-compute2.json)
  — raw run: `dump\runs\20261004T025612Z-fdeca883` (configuration, source snapshot, metrics and checkpoints).
* [Step 1 research contract](../../../docs/step_1_ffn.md)
* [Other model reports](../../../README.md#results)

The exact executed launcher, original model source and generated configs are
preserved with the local runs. To train an existing configuration using the
simplified layout, run this from the repository root:

```powershell
uv run python -m main run dump/baselines/rewrite-v1/configs/full-swiglu.json
```

Choose the matching variant's JSON file to run that model. Replaying the historical
screen or resuming its checkpoints requires its original frozen source because
the trainer checks source hashes. These measured scores remain historical evidence.
Raw artifacts are ignored by Git; the linked compact evidence retains the
settings, source/data hashes and measured results for review.

## Layout cleanup verification

The current implementation is in [transformer.py](transformer.py), with shared
components and one trainer. The cleanup preserved exact initial weights, outputs,
gradients and counts across all 11 variants; existing checkpoints load into the
three full models. Short GPU training and exact CPU resume were checked again.
See [layout verification](../../../records/layout-check.json) for the record.

## PairFlux control rerun

Fresh controls were rerun under the same source fingerprint as the new FFN,
with the original 2,000-update, seed-101 task recipe. These remain development
measurements. Full comparison and attribution are in the
[PairFlux report](../../../docs/retired_models/pairflux_transformer/result.md).

| Variant | Complete validation NLL | Update seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| full-swiglu | 1.522579 | 73.93 | 252.95 |
| full-gelu | 1.521821 | 66.16 | 241.20 |
| narrow-swiglu | 1.495532 | 77.56 | 220.12 |
| narrow-gelu | 1.518696 | 72.05 | 214.75 |

## Language screen v1

No candidate meets both language margins on both corpora. Retire these fixed recipes and register a revision before further training.

tinystories: strongest full `full-swiglu`; strongest compact `narrow-swiglu-fused`. maxout: full loss cost 3.93%, compact loss gain -0.56%. curve-only: full loss cost 6.25%, compact loss gain -2.81%. blockshuffle: full loss cost 2.55%, compact loss gain 0.77%.

wikitext: strongest full `full-swiglu`; strongest compact `narrow-swiglu`. maxout: full loss cost 3.57%, compact loss gain -1.11%. curve-only: full loss cost 4.53%, compact loss gain -2.05%. blockshuffle: full loss cost 2.53%, compact loss gain -0.09%.

| Corpus | Variant | NLL | Update seconds | Allocated MiB |
| --- | --- | ---: | ---: | ---: |
| tinystories | full-swiglu | 2.883340 | 68.34 | 252.95 |
| tinystories | full-swiglu-fused | 2.884440 | 69.85 | 249.95 |
| tinystories | full-gelu | 2.979904 | 69.55 | 241.20 |
| tinystories | narrow-swiglu | 2.980331 | 60.01 | 218.00 |
| tinystories | narrow-swiglu-fused | 2.979796 | 66.18 | 216.56 |
| tinystories | narrow-gelu | 3.003410 | 66.01 | 212.62 |
| wikitext | full-swiglu | 4.523139 | 71.89 | 253.87 |
| wikitext | full-swiglu-fused | 4.523150 | 75.25 | 250.87 |
| wikitext | full-gelu | 4.641271 | 73.00 | 242.12 |
| wikitext | narrow-swiglu | 4.633178 | 78.83 | 218.92 |
| wikitext | narrow-swiglu-fused | 4.633594 | 45.92 | 217.48 |
| wikitext | narrow-gelu | 4.677101 | 69.12 | 213.55 |

[Complete comparison, profiles and limitations](../../../docs/retired_models/maxout_transformer/result.md).

## Shared-bank and gate-transport screen v2

[Registered hypothesis](../../../docs/shared_gate_transport_hypothesis.md).
Status: 12/12 completed quality runs, seed 101 only.

Retire this fixed plain-sharing recipe: it does not meet both language margins on both corpora.

tinystories: strongest full `full-swiglu-fused`, strongest compact `narrow-swiglu-fused`. Shared-bank full loss cost 2.57%; compact gain 0.86%.

wikitext: strongest full `full-swiglu-fused`, strongest compact `narrow-swiglu-fused`. Shared-bank full loss cost 1.42%; compact gain 0.32%.

### Hardware decision

| Corpus | Mechanism | Allocated MiB | Cut vs SwiGLU / GELU | Speed vs SwiGLU / GELU | Hardware screen |
| --- | --- | ---: | ---: | ---: | --- |
| tinystories | gate-transport | 442.7 | 21.0% / 17.9% | 1.019x / 1.010x | fail |
| tinystories | shared-basis | 425.7 | 24.1% / 21.1% | 1.103x / 1.093x | pass |
| wikitext | gate-transport | 442.7 | 21.0% / 17.9% | 1.052x / 0.998x | fail |
| wikitext | shared-basis | 425.7 | 24.1% / 21.1% | 1.123x / 1.065x | pass |

The gate-transport recipe failed against full GELU and was retired before language
training; its quality is unmeasured. Plain sharing passed the short memory screen
and advanced. No 1.2x speed claim is supported. Profiles used 20 warmup plus 100
measured updates, three alternating-order rounds on each corpus. Use median
throughput and the largest allocated peak; this is not sustained confirmation.

### Language measurements

Same width-512, four-layer, 16-head Transformer, context 256, vocabulary 4096,
batch eight, BF16. Each model receives 2,000 updates / 4,096,000 supervised targets,
rate 0.0006, warmup 200, decay 0.1. Full validation covers eligible nonoverlapping
windows, with the same short tail omitted for every model. Test loss is unscored.

| Corpus | Variant | NLL | Update seconds | Allocated / reserved MiB |
| --- | --- | ---: | ---: | ---: |
| tinystories | full-swiglu-fused | 2.538486 | 89.69 | 557.5 / 594.0 |
| tinystories | full-gelu | 2.592983 | 86.84 | 539.4 / 610.0 |
| tinystories | narrow-swiglu-fused | 2.626184 | 82.92 | 402.3 / 464.0 |
| tinystories | narrow-gelu | 2.630634 | 79.63 | 397.4 / 450.0 |
| tinystories | shared-basis | 2.603695 | 80.26 | 425.7 / 486.0 |
| tinystories | untied-basis | 2.586359 | 85.26 | 448.0 / 490.0 |
| wikitext | full-swiglu-fused | 4.237625 | 89.01 | 560.7 / 602.0 |
| wikitext | full-gelu | 4.269168 | 93.27 | 539.4 / 610.0 |
| wikitext | narrow-swiglu-fused | 4.311688 | 84.77 | 402.3 / 464.0 |
| wikitext | narrow-gelu | 4.330570 | 80.59 | 397.4 / 450.0 |
| wikitext | shared-basis | 4.297722 | 82.24 | 425.7 / 486.0 |
| wikitext | untied-basis | 4.272552 | 82.87 | 448.0 / 490.0 |

Shared hidden 608 has 1,867,776 unique FFN weights, 77.91% fewer than full SwiGLU.
Its total model has 8,163,840 weights versus 14,750,208 (44.65% fewer). Projection
arithmetic is 55.81% lower, but excludes pointwise work, attention and logits.
Compact SwiGLU hidden 304 and GELU hidden 456 have exactly the same FFN weight
count. Untied hidden 608 has twice those FFN weights; it measures the cost of
sharing, not performance at equal storage.

### Removal measurements

* tinystories/full-swiglu-fused: original 2.538486; half hidden features removed 3.030469; training-mean output 9.228604.
* tinystories/full-gelu: original 2.592983; half hidden features removed 2.905189; training-mean output 7.583977.
* tinystories/narrow-swiglu-fused: original 2.626184; half hidden features removed 3.358260; training-mean output 8.772416.
* tinystories/narrow-gelu: original 2.630634; half hidden features removed 2.997268; training-mean output 7.174017.
* tinystories/shared-basis: original 2.603695; half hidden features removed 3.196728; training-mean output 9.300469.
* tinystories/untied-basis: original 2.586359; half hidden features removed 3.228277; training-mean output 8.871957.
* wikitext/full-swiglu-fused: original 4.237625; half hidden features removed 4.771781; training-mean output 12.236745.
* wikitext/full-gelu: original 4.269168; half hidden features removed 4.625870; training-mean output 10.499083.
* wikitext/narrow-swiglu-fused: original 4.311688; half hidden features removed 4.972029; training-mean output 9.321081.
* wikitext/narrow-gelu: original 4.330570; half hidden features removed 4.773334; training-mean output 10.269190.
* wikitext/shared-basis: original 4.297722; half hidden features removed 4.990164; training-mean output 11.054945.
* wikitext/untied-basis: original 4.272552; half hidden features removed 4.932004; training-mean output 9.988460.

Even-index removal zeros half the hidden features at the output projection.
Training-mean intervention uses 32 batches at fixed seed 2027 from training only.
Both interventions change activation statistics and diagnose functional dependence;
they do not identify semantic roles. Retrained compact and untied controls remain
necessary. Evidence: `records/shared-gate-v2-*.json`, raw interventions and frozen
source under `dump/shared-gate-v2/` and the saved runs. Numerical equation/gradient,
shared alias, count, causality, save/load and exact CPU resume checks passed.
No architecture is qualified; originality is unverified and sharing is prior art.

## Group-product language screen v4

Retire this fixed group-product recipe: it misses a language margin or its mechanism-control comparison on a required corpus.

| Corpus | Variant | Validation NLL | Targets | Update / wall seconds | Allocated / reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| tinystories | full-swiglu-kernel | 2.544240 | 199,936 | 88.34 / 101.31 | 542.5 / 574.0 |
| tinystories | full-swiglu-fused | 2.538486 | 199,936 | 85.46 / 97.79 | 560.7 / 602.0 |
| tinystories | full-gelu | 2.592983 | 199,936 | 84.03 / 96.68 | 539.4 / 610.0 |
| tinystories | narrow-swiglu-kernel | 2.617045 | 199,936 | 83.49 / 93.81 | 405.7 / 464.0 |
| tinystories | narrow-gelu | 2.629526 | 199,936 | 79.82 / 90.13 | 400.9 / 450.0 |
| tinystories | plain-silu | 2.673912 | 199,936 | 78.53 / 88.94 | 400.9 / 450.0 |
| tinystories | square | 2.645681 | 199,936 | 85.25 / 96.16 | 422.9 / 470.0 |
| tinystories | group-product | 2.651852 | 199,936 | 82.85 / 93.40 | 401.9 / 452.0 |
| wikitext | full-swiglu-kernel | 4.239026 | 322,560 | 88.13 / 100.85 | 540.5 / 582.0 |
| wikitext | full-swiglu-fused | 4.237625 | 322,560 | 89.10 / 102.25 | 560.7 / 602.0 |
| wikitext | full-gelu | 4.269168 | 322,560 | 87.89 / 101.21 | 539.4 / 610.0 |
| wikitext | narrow-swiglu-kernel | 4.308211 | 322,560 | 84.10 / 94.78 | 405.7 / 464.0 |
| wikitext | narrow-gelu | 4.312711 | 322,560 | 82.54 / 93.23 | 400.9 / 450.0 |
| wikitext | plain-silu | 4.369832 | 322,560 | 78.49 / 88.64 | 400.9 / 450.0 |
| wikitext | square | 4.326767 | 322,560 | 88.36 / 99.45 | 422.9 / 470.0 |
| wikitext | group-product | 4.333677 | 322,560 | 86.12 / 96.80 | 401.9 / 452.0 |

[Complete candidate, hardware and removal report](../../../docs/retired_models/group_product_transformer/result.md).

## Context-bank language screen v1

Retire this fixed context-bank recipe: a required language margin or mechanism comparison fails.

| Corpus | Variant | Validation NLL | Targets | Update / wall seconds | Allocated / reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| tinystories | full-swiglu-kernel | 2.544240 | 199,936 | 90.53 / 103.81 | 542.5 / 574.0 |
| tinystories | full-swiglu-fused | 2.538486 | 199,936 | 87.90 / 100.66 | 560.7 / 602.0 |
| tinystories | full-gelu | 2.592983 | 199,936 | 87.31 / 100.00 | 539.4 / 610.0 |
| tinystories | narrow-swiglu-kernel | 2.615981 | 199,936 | 83.73 / 94.40 | 405.9 / 464.0 |
| tinystories | narrow-gelu | 2.623549 | 199,936 | 79.99 / 90.66 | 404.8 / 468.0 |
| tinystories | plain-silu | 2.673912 | 199,936 | 79.52 / 89.91 | 400.9 / 450.0 |
| tinystories | context-additive | 2.674518 | 199,936 | 87.70 / 98.52 | 408.2 / 468.0 |
| tinystories | context-product | 2.672406 | 199,936 | 85.75 / 96.18 | 408.2 / 468.0 |
| wikitext | full-swiglu-kernel | 4.239026 | 322,560 | 86.59 / 99.22 | 540.5 / 582.0 |
| wikitext | full-swiglu-fused | 4.237625 | 322,560 | 90.89 / 103.93 | 560.7 / 602.0 |
| wikitext | full-gelu | 4.269168 | 322,560 | 89.29 / 102.36 | 539.4 / 610.0 |
| wikitext | narrow-swiglu-kernel | 4.297109 | 322,560 | 86.35 / 97.52 | 405.9 / 464.0 |
| wikitext | narrow-gelu | 4.321449 | 322,560 | 82.23 / 93.21 | 404.8 / 468.0 |
| wikitext | plain-silu | 4.369832 | 322,560 | 81.25 / 91.77 | 400.9 / 450.0 |
| wikitext | context-additive | 4.365825 | 322,560 | 87.73 / 98.51 | 408.2 / 468.0 |
| wikitext | context-product | 4.367791 | 322,560 | 85.02 / 95.63 | 408.2 / 468.0 |

[Candidate and removal report](../../../docs/retired_models/context_bank_transformer/result.md).

## Signed-context language screen v1

Retire both fixed signed recipes: neither passes all resource, quality and mechanism comparisons on both corpora.

| Corpus | Variant | Validation NLL | Targets | Update / wall seconds | Allocated / reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| tinystories | full-swiglu-kernel | 2.544240 | 199,936 | 100.23 / 114.57 | 542.5 / 574.0 |
| tinystories | full-swiglu-fused | 2.538486 | 199,936 | 108.14 / 123.43 | 560.7 / 602.0 |
| tinystories | full-gelu | 2.592983 | 199,936 | 89.06 / 102.05 | 539.4 / 610.0 |
| tinystories | narrow-swiglu-kernel | 2.606365 | 199,936 | 84.02 / 94.97 | 407.6 / 450.0 |
| tinystories | narrow-gelu | 2.616391 | 199,936 | 80.39 / 90.99 | 408.9 / 468.0 |
| tinystories | plain-silu | 2.666190 | 199,936 | 78.77 / 89.06 | 406.7 / 468.0 |
| tinystories | closed-gain | 2.667904 | 199,936 | 87.36 / 98.18 | 415.9 / 468.0 |
| tinystories | open-gain | 2.619272 | 199,936 | 87.08 / 97.84 | 415.9 / 468.0 |
| tinystories | signed-linear | 2.603935 | 199,936 | 85.52 / 96.30 | 415.9 / 468.0 |
| tinystories | signed-tanh | 2.617161 | 199,936 | 83.56 / 94.29 | 415.9 / 468.0 |
| wikitext | full-swiglu-kernel | 4.239026 | 322,560 | 90.18 / 103.35 | 540.5 / 582.0 |
| wikitext | full-swiglu-fused | 4.237625 | 322,560 | 90.58 / 104.21 | 560.7 / 602.0 |
| wikitext | full-gelu | 4.269168 | 322,560 | 88.43 / 101.39 | 539.4 / 610.0 |
| wikitext | narrow-swiglu-kernel | 4.288883 | 322,560 | 83.81 / 94.49 | 407.6 / 450.0 |
| wikitext | narrow-gelu | 4.317237 | 322,560 | 82.54 / 93.34 | 408.9 / 468.0 |
| wikitext | plain-silu | 4.377303 | 322,560 | 80.56 / 91.16 | 406.7 / 468.0 |
| wikitext | closed-gain | 4.373279 | 322,560 | 85.83 / 96.98 | 415.9 / 468.0 |
| wikitext | open-gain | 4.309518 | 322,560 | 85.99 / 96.86 | 415.9 / 468.0 |
| wikitext | signed-linear | 4.277294 | 322,560 | 84.23 / 95.00 | 415.9 / 468.0 |
| wikitext | signed-tanh | 4.302946 | 322,560 | 85.78 / 96.63 | 415.9 / 468.0 |

[Candidate, hardware and removal report](../../../ffn_experiments/signed_context_transformer/result.md).
