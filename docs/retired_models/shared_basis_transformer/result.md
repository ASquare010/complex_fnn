# Shared FFN basis: first hardware screen

The initial registered recipes were implemented, numerically checked and retired
after failing the hardware screen. Their quality was not measured. The separate
v2 study and its current results appear below.
[Hypothesis and failure criteria](../../shared_basis_hypothesis.md).

The up/gate projection is shared across four layers; each layer keeps an
independent output projection. At width 192 and hidden 304, unique FFN parameters
are 350,208 (70.31% fewer than full SwiGLU). Projection FLOPs are 1,400,832/token
across four layers (40.63% fewer). Shared storage does not eliminate repeated
projection execution. The equally small dense SwiGLU control has hidden 152.

Aliasing, unique optimizer parameters, summed shared gradients, identical initial
untied control, save/load and causality checks passed on the prototype. The
integrated model repeats those checks under the production configuration schema.

The registered hardware screen compares widths 192, 384 and 512 with fixed
backbones within each comparison, both corpora, full fused SwiGLU/GELU controls,
and an attention-only memory diagnostic. It selects the smallest size meeting
the predeclared resource screen. No language-quality or confirmed resource claim
is available yet. Cross-layer sharing and wider shared FFNs have substantial
prior art; this study makes no novelty claim.

## Hardware screen v1

Completed 72 profiles: three widths, both corpora, four variants, three alternating-order rounds. Each round measured 100 updates after 20 warmup updates. Throughput is the median; allocated memory is the largest peak across rounds. All runs stayed below 6 GiB reserved.

| Width | Corpus | Shared allocated MiB | Cut vs full SwiGLU | Cut vs full GELU | Speed vs SwiGLU | Speed vs GELU | Screen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 192 | tinystories | 226.2 | 9.9% | 6.7% | 1.013x | 0.976x | fail |
| 192 | wikitext | 226.2 | 9.9% | 6.7% | 1.039x | 0.997x | fail |
| 384 | tinystories | 351.2 | 17.2% | 14.5% | 1.042x | 1.057x | fail |
| 384 | wikitext | 351.2 | 17.2% | 14.5% | 1.058x | 1.044x | fail |
| 512 | tinystories | 452.8 | 20.2% | 17.0% | 1.052x | 1.040x | fail |
| 512 | wikitext | 452.8 | 20.2% | 17.0% | 1.117x | 1.072x | fail |

No width passes against both controls on both corpora. These fixed recipes are retired before language training. Quality is unmeasured, and no language failure is inferred. Fewer stored weights alone did not meet the training-resource target.

Evidence: `records/shared-basis-v1-profiles.json` and `records/shared-basis-v1-hardware-decision.json`. Hardware: RTX 4070 Laptop GPU; BF16; batch 8, context 256, four layers. Short timings are sensitive to laptop power and temperature; they are a screen, not independent confirmation.

## Shared-bank and gate-transport screen v2

[Registered hypothesis](../../shared_gate_transport_hypothesis.md).
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
