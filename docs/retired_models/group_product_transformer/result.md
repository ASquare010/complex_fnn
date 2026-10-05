# Group-product FFN: fused implementation and language screen

Status: 16/16 completed language runs, seed 101 only.
[Hypothesis and acceptance contract](../../group_product_v4_hypothesis.md).

Retire this fixed group-product recipe: it misses a language margin or its mechanism-control comparison on a required corpus.

tinystories: strongest full `full-swiglu-fused`, strongest compact dense `narrow-swiglu-kernel`. Group loss cost 4.47%; compact gain -1.33%; beats plain/square: False.

wikitext: strongest full `full-swiglu-fused`, strongest compact dense `narrow-swiglu-kernel`. Group loss cost 2.27%; compact gain -0.59%; beats plain/square: False.

## Proposed primitive and cost

Project each token into 512 features, grouped into 32 sets of 16. A feature's
quadratic residual is its value times the sum of the other 15 in its group,
scaled by sqrt(15). Learned per-feature coefficients mix this with direct SiLU;
an independent output projection maps it to the residual stream. U/D/coefficient
weights are independent in each layer. The group sum avoids a pairwise matrix.
Useful feature organization and quality are hypotheses, not established semantics.

FFN weights: 2,099,200, 75.17% fewer than full SwiGLU's 8,454,144. Total model:
8,395,264 versus 14,750,208, 43.08% fewer. Projection FLOPs: 4,194,304/token,
75.19% fewer; excludes sums, nonlinearities, gradients, attention and logits.
Compact SwiGLU has 0.68% more FFN weights; compact GELU/plain have 0.10% fewer.
Square has exactly the candidate's count and tests a learned unary alternative.

## Hardware gate

| Corpus | Allocated MiB | Cut vs native SwiGLU / kernel SwiGLU / GELU | Throughput vs native / kernel / GELU |
| --- | ---: | ---: | ---: |
| tinystories | 402.0 | 28.31% / 25.62% / 25.48% | 1.065x / 1.036x / 1.024x |
| wikitext | 402.0 | 28.31% / 25.62% / 25.48% | 1.025x / 1.010x / 1.012x |

Passed both corpora against all three full controls. Thirty short profiles:
three alternating-order rounds, 20 warmup plus 100 measured updates. Kernel
compilation occurs in warmup and its time is recorded separately. Use median
throughput and the largest allocated peak; all fit below 6 GiB reserved. Hardware:
RTX 4070 Laptop GPU, PyTorch 2.14.0+cu132, BF16, batch eight, context 256.
This supports advancement, not sustained speed or a confirmed memory claim.
No 1.2x training-speed claim is supported. The dense SwiGLU engineering control
also fuses its activation/adjoint, avoiding a deliberately weaker baseline.

## Language measurements

Same width-512, four-layer, 16-head Transformer; vocabulary 4096. All receive
2,000 updates / 4,096,000 supervised targets, rate 0.0006, warmup 200, AdamW
decay 0.1, the same seed and sampled windows. Full validation covers eligible
nonoverlapping context windows, omitting the same short tail. Test losses remain
unscored. NLL comparisons are within each corpus, whose tokenizers differ.

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

Update/wall times are single complete-run observations and include cold/runtime
effects; use repeated profiles for the hardware screen. Reserved memory differs
from allocated and device-wide usage. These are small models, not production LLMs.

## Mechanism checks and limitations

* tinystories/group-product: trained 2.651852; direct-path replacement 2.795028; half features removed 2.993326; training-mean output 7.221203; mean |tanh(a)| 0.043283.
* tinystories/group-product: coefficient zero with the original arithmetic backend 2.795028.
* tinystories/square: trained 2.645681; direct-path replacement 2.850513; half features removed 3.012673; training-mean output 7.409508; mean |tanh(a)| 0.078615.
* tinystories/square: coefficient zero with the original arithmetic backend 2.850513.
* wikitext/group-product: trained 4.333677; direct-path replacement 4.561932; half features removed 4.696038; training-mean output 8.895543; mean |tanh(a)| 0.045776.
* wikitext/group-product: coefficient zero with the original arithmetic backend 4.561932.
* wikitext/square: trained 4.326767; direct-path replacement 4.604490; half features removed 4.710523; training-mean output 8.717128; mean |tanh(a)| 0.088251.
* wikitext/square: coefficient zero with the original arithmetic backend 4.604490.

Direct-path replacement uses native SiLU; parameter-zero measurements after the
campaign preserve the original arithmetic backend and isolate coefficient removal.
Training-mean output uses 32 fixed training batches, seed 2027. Half-feature
removal and constant replacement alter statistics; they diagnose dependence,
not semantic roles. Compare retrained plain and square controls too.

CPU FP64 independent dense-adjacency references and finite differences passed.
CUDA FP32/BF16 activation/gradient references passed, with observed differences
recorded. Production matches the staged model, including nonzero coefficients,
and both the group model and compiled dense control resume exactly on CPU.
Full Transformer locality, causality, counts and save/load passed. FP32 activation
arithmetic changes rounding relative to the earlier native BF16 implementation.

Prior work includes GLUs, multiplicative features and inexpensive channel context.
Originality is unverified. A one-seed screen does not establish an architectural
contribution; equal tuning, fresh seeds, longer resources and independent
confirmation remain required. Evidence: `records/group-product-v4-*.json`,
`records/group-product-language-v4-comparison.json`, frozen run sources and raw
interventions under `dump/group-product-language-v4/`. Earlier failures are in
[the research history](../../group_product_result.md).
