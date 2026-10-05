# PairFlux: first mechanism screen

Measured 2026-10-03 (America/Los_Angeles), study `pairflux-v1`.
Decision: **retire this combined exchange/curve recipe; no combined research win.**
The complete candidate is worse than the retrained curve-only control. The learned-curve
branch deserves a separate question, not a claim that the exchange worked.

## What was tested

[The hypothesis](../../pairflux_hypothesis.md) was registered before
training. Only the FFN changes. Every model retains embeddings and full causal
attention. The candidate uses two input-dependent conservative pair matchings
and a bounded local quadratic correction to SiLU, with hidden width 224.

All nine recipes completed 2,000 AdamW updates on `dump/data/tasks-v1`, seed 101,
rate 0.0006, 200 warmup updates, batch eight, context 256 and CUDA BF16 on an
RTX 4070 Laptop GPU with PyTorch 2.14.0+cu132. The backbone is width 192, four
layers, six heads and vocabulary budget 4096. Shared initialization and sampled
batches match across the four mechanism ablations. All runs use one source
fingerprint and one data fingerprint, 4,096,000 padded input positions and
42,851 supervised answer targets. Test and OOD remain unscored.

Full last-checkpoint validation covers all 569 examples and 1,538 answer targets.
Composition error equally weights chain and arithmetic errors, pooling difficulty
two and above within each task. Lower NLL/error is better. Ablations remove their
unused scalar parameters. Compact dense controls have 1.26% more FFN parameters
than the complete candidate; this is an approximate budget match, not equality.

## Quality

| Variant | Total / FFN weights | Full-valid NLL | Lookup accuracy | Composition error |
| --- | ---: | ---: | ---: | ---: |
| pairflux | 1,723,840 / 345,856 | 1.499664 | 11.00% | 85.86% |
| no-exchange | 1,722,944 / 344,960 | 1.462825 | 10.00% | 86.00% |
| no-curve | 1,722,944 / 344,960 | 1.486412 | 11.00% | 86.11% |
| plain-silu | 1,722,048 / 344,064 | 1.510481 | 7.50% | 86.86% |
| full-swiglu | 2,557,632 / 1,179,648 | 1.522579 | 5.50% | 87.79% |
| full-gelu | 2,557,632 / 1,179,648 | 1.521821 | 5.50% | 89.36% |
| narrow-swiglu | 1,728,192 / 350,208 | 1.495532 | 9.00% | 86.71% |
| narrow-gelu | 1,728,192 / 350,208 | 1.518696 | 5.50% | 88.96% |
| blockshuffle | 1,728,192 / 350,208 | 1.401092 | 20.00% | 81.96% |

Retraining without exchange improves NLL by 0.036839 nats.
The curve-only branch improves NLL 3.16% against identical-width
plain SiLU and 2.19% against `narrow-swiglu` in this seed.
Exchange alone improves NLL 1.59% against plain SiLU, but the
complete combination is worse than either component alone. This is evidence
against the registered combination, not evidence that exchange always hurts.
These are development observations without an across-seed confidence interval.
Generated-answer accuracy and composition remain weak; do not infer general
reasoning quality or LLM improvement from the NLL differences.

## Actual resource cost

| Variant | Update seconds | Wall seconds | Allocated / reserved MiB | Profile median [range] positions/s |
| --- | ---: | ---: | ---: | ---: |
| pairflux | 100.46 | 109.25 | 242.58 / 280.00 | 44,000 [42,674, 44,300] |
| no-exchange | 76.64 | 84.22 | 225.06 / 260.00 | 53,422 [52,271, 66,966] |
| no-curve | 92.04 | 100.41 | 232.06 / 264.00 | 44,908 [41,720, 52,985] |
| plain-silu | 61.07 | 67.70 | 214.54 / 250.00 | 65,954 [58,971, 85,080] |
| full-swiglu | 73.93 | 81.58 | 252.95 / 294.00 | 55,159 [52,051, 80,882] |
| full-gelu | 66.16 | 73.30 | 241.20 / 290.00 | 64,715 [63,826, 78,529] |
| narrow-swiglu | 77.56 | 85.22 | 220.12 / 252.00 | 57,450 [52,265, 65,524] |
| narrow-gelu | 72.05 | 79.67 | 214.75 / 250.00 | 58,810 [56,925, 66,914] |
| blockshuffle | 107.64 | 117.31 | 285.12 / 336.00 | 40,885 [40,367, 43,852] |

The complete FFN has 345,856 weights versus 1,179,648 for full SwiGLU: 70.68%
fewer. Total parameters fall from 2,557,632 to 1,723,840: 32.60% fewer.
Projection FLOPs fall from 2,359,296 to 688,128 per token across four layers,
but exclude exchange, division and activation costs. The extra operations
consume execution time and backward intermediates. The complete candidate
does not meet the combined speed/memory targets.

Profiles use actual task batches, five warmup and 20 measured updates, three
rounds with reversed order in round two. Each starts from fresh weights and
optimizer. The reported median and range show substantial timing variability;
these short profiles qualify execution and memory fit, not sustained speed or
serving latency. Update time includes startup and batching; wall time also
includes periodic validation/checkpointing, but excludes later full evaluation.
Positions/s includes padding. PyTorch memory is not device-wide peak memory.
All profiles had finite loss/gradients and stayed below 6 GiB reserved memory.

## Correctness and interpretation

Passed independent scalar-equation agreement, pair-sum preservation, float64
transport gradient checks, full-FFN gradient checks with nonzero curvature,
token locality, Transformer causality, exact zero-coefficient plain-SiLU
equivalence, parameter counts, BF16 backward and exact interrupted CPU resume.

The sum-preserving exchange is more constrained than a general gated bilinear
FFN. Bounded updates and a fixed sparse graph may limit useful interactions;
this screen does not isolate those explanations. It establishes a negative
result for the registered recipe, not impossibility of neuron interaction.

### Do the final weights need the new operations?

Additional frozen-weight interventions produce a different result from retraining:

| Trained model | Original NLL | NLL after disabling its added component |
| --- | ---: | ---: |
| Curve only | 1.462825 | 1.462794 |
| Exchange only | 1.486412 | 1.486403 |
| Combined | 1.499664 | 1.499658, both disabled |

The tiny changes do not support the claim that these final models depend on
richer inference computation from the added operations. Learned coefficients
are nonzero, but that alone does not demonstrate useful feature extraction.
Differences between retrained models may reflect training dynamics, BF16 rounding
or one-seed trajectory variation; these measurements do not distinguish them.
Do not attribute the 3.16% curve-only training-run gain to proven additional
information flow. A separate study could test these operations as training
scaffolding and remove them at inference, with fresh seeds, affine-gain and
numerical-perturbation controls. This is another hypothesis, not a qualified win.

Do not promote the complete candidate or launch its expensive confirmation.
If pursuing the curve-only observation, register a separate study with equal
rate/seed tuning, an affine-gain control, and TinyStories/WikiText confirmation.
An exchange-only follow-up would also need to overcome its execution overhead.
The gain control is necessary because the quadratic residual can change local
slope as well as shape. Learned activations and bilinear interactions already
have prior art. No originality or broad LLM performance claim is established.

## Reproduce and inspect

The repository [study recipe](../../study_configs/pairflux_study.json) freezes all nine
configurations. The ignored `dump/pairflux-v1/` contains generated configs,
profiles and run mapping; each run retains checkpoints, logs and frozen source.
`dump/pairflux_screen.py` is the exact local sequential launcher; it is research
provenance, not another active framework. Use a fresh study/output identity for
a new campaign; never overwrite recorded evaluations.

Compact [comparison](../../../records/pairflux-v1-comparison.json),
[numerical checks](../../../records/pairflux-checks.json),
[resume check](../../../records/pairflux-resume.json) and
[profiles](../../../records/pairflux-profiles.json) retain the evidence.
The [combined-model interventions](../../../records/pairflux-diagnostics.json)
and [isolated-component interventions](../../../records/pairflux-component-diagnostics.json)
record the post-training removal tests.
Individual `records/pairflux-v1-*.json` include complete configurations,
training summaries and task/difficulty validation counts.

## Language screen v1

No candidate meets both language margins on both corpora. Retire these fixed recipes and register a revision before further training.

tinystories: strongest full `full-swiglu`; strongest compact `narrow-swiglu-fused`. maxout: full loss cost 3.93%, compact loss gain -0.56%. curve-only: full loss cost 6.25%, compact loss gain -2.81%. blockshuffle: full loss cost 2.55%, compact loss gain 0.77%.

wikitext: strongest full `full-swiglu`; strongest compact `narrow-swiglu`. maxout: full loss cost 3.57%, compact loss gain -1.11%. curve-only: full loss cost 4.53%, compact loss gain -2.05%. blockshuffle: full loss cost 2.53%, compact loss gain -0.09%.

| Corpus | Variant | NLL | Update seconds | Allocated MiB |
| --- | --- | ---: | ---: | ---: |
| tinystories | plain-silu | 3.067467 | 67.27 | 212.41 |
| tinystories | curve-only | 3.063614 | 73.63 | 222.93 |
| wikitext | plain-silu | 4.740294 | 89.27 | 213.34 |
| wikitext | curve-only | 4.728093 | 84.00 | 223.86 |

[Complete comparison, profiles and limitations](../maxout_transformer/result.md).
