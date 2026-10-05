# BlockShuffle: post-rewrite results

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
| blockshuffle | 155.82 | 169.43 | 26,286 | 285.12 / 316.00 |

Update time includes startup/warmup and data batching. Training-loop wall time
also includes periodic validation/checkpointing, but excludes model setup and
the later full-validation/generation pass. Throughput counts padding; it is
not useful-answer throughput. Memory is PyTorch allocation, not device-wide
peak usage. These single, fixed-order runs do not qualify a speed advantage.

## Generated answers by task and difficulty

| Variant | Task / difficulty | Correct / examples | Accuracy |
| --- | --- | ---: | ---: |
| blockshuffle | arithmetic/difficulty_1 | 9 / 29 | 31.03% |
| blockshuffle | arithmetic/difficulty_2 | 16 / 70 | 22.86% |
| blockshuffle | arithmetic/difficulty_3 | 10 / 70 | 14.29% |
| blockshuffle | chain/difficulty_2 | 19 / 95 | 20.00% |
| blockshuffle | chain/difficulty_3 | 16 / 105 | 15.24% |
| blockshuffle | lookup/difficulty_1 | 40 / 200 | 20.00% |

## Interpretation and next decision

The lowest full-dense NLL in this screen is **full-gelu** (1.521821);
the lowest single-pass compact-dense NLL is **narrow-swiglu** (1.495532).
These observations use one rate and seed, not equal multi-rate tuning.

Plain BlockShuffle has the best NLL among these eight runs, with improved
lookup and chain accuracy. Arithmetic does not show a matching improvement.
Retain it for equal-effort tuning. It does not meet the combined Step 1 goal:
the composition-error gain is below 20%, and measured speed and allocated
memory regress against full SwiGLU in this run.

* blockshuffle: NLL change vs full-gelu **-7.933%**; vs narrow-swiglu **-6.315%** (negative is better).

Against full SwiGLU, this model uses **70.3125% fewer FFN weights**
and **32.43% fewer total weights**. Observed padded-input throughput
was **0.745x**, and peak allocated training memory was **1.132x**
the full baseline. These are descriptive single-run ratios, not stable speed estimates.

| Comparison control | Candidate NLL change | Candidate composition error reduction | Lookup change (percentage points) |
| --- | ---: | ---: | ---: |
| narrow-swiglu | -6.315% | +5.478% | +11.00 |

Retain this baseline and its task breakdown. No model is promoted to a confirmed
winner. Next, qualify learning with equal rate/seed tuning before the fresh-seed
confirmation and language experiments specified in Step 1. TinyStories and
WikiText were not trained here. Repeated, alternated timing and quality-matched
comparisons are still required for a speed claim. No inference benchmark was run.

The four-pass structured/dense variants and `dense-compute4` received execution
checks only; their 2,000-step learning studies remain deferred.

## Evidence and reproduction

* [Checks and all-variant preflight measurements](../../../records/rewrite-v1-checks.json)
* [blockshuffle complete compact record](../../../records/rewrite-v1-blockshuffle.json)
  — raw run: `dump\runs\20261004T024407Z-c40b110b` (configuration, source snapshot, metrics and checkpoints).
* [Step 1 research contract](../../step_1_ffn.md)
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

At the earlier layout cleanup, the implementation used shared components and one
trainer. It has since been retired from active source. That cleanup preserved exact initial weights, outputs,
gradients and counts across all 11 variants; existing checkpoints load into the
three full models. Short GPU training and exact CPU resume were checked again.
See [layout verification](../../../records/layout-check.json) for the record.

## PairFlux control rerun

Fresh controls were rerun under the same source fingerprint as the new FFN,
with the original 2,000-update, seed-101 task recipe. These remain development
measurements. Full comparison and attribution are in the
[PairFlux report](../pairflux_transformer/result.md).

| Variant | Complete validation NLL | Update seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| blockshuffle | 1.401092 | 107.64 | 285.12 |

## Language screen v1

No candidate meets both language margins on both corpora. Retire these fixed recipes and register a revision before further training.

tinystories: strongest full `full-swiglu`; strongest compact `narrow-swiglu-fused`. maxout: full loss cost 3.93%, compact loss gain -0.56%. curve-only: full loss cost 6.25%, compact loss gain -2.81%. blockshuffle: full loss cost 2.55%, compact loss gain 0.77%.

wikitext: strongest full `full-swiglu`; strongest compact `narrow-swiglu`. maxout: full loss cost 3.57%, compact loss gain -1.11%. curve-only: full loss cost 4.53%, compact loss gain -2.05%. blockshuffle: full loss cost 2.53%, compact loss gain -0.09%.

| Corpus | Variant | NLL | Update seconds | Allocated MiB |
| --- | --- | ---: | ---: | ---: |
| tinystories | blockshuffle | 2.956800 | 98.13 | 284.67 |
| wikitext | blockshuffle | 4.637398 | 123.26 | 284.67 |

[Complete comparison, profiles and limitations](../maxout_transformer/result.md).
