# Independent context-bank FFN: result

[Registered hypothesis](../../context_bank_hypothesis.md).

Retire this fixed context-bank recipe: a required language margin or mechanism comparison fails. Completed 16/16 quality runs.

tinystories: strongest full `full-swiglu-fused`, strongest compact dense `narrow-swiglu-kernel`. Full loss cost 5.28%; compact gain -2.16%; beats additive/plain: True.

wikitext: strongest full `full-swiglu-fused`, strongest compact dense `narrow-swiglu-kernel`. Full loss cost 3.07%; compact gain -1.64%; beats additive/plain: False.

## Computation and size

512 learned detector features, 64 independently learned context predicates.
Eight detectors share one context. `r_i=SiLU(u_i)*(1+tanh(a_i)*tanh(v_group(i)))`.
U/C execute as one 576-wide input projection; D reads 512 features. All weights
are independent across layers. A matched additive control tests whether the
context's unary features suffice without products. This is token-local gating.

2,230,272 FFN weights, 73.62% fewer than full SwiGLU. Projection FLOPs 4,456,448
per token, 73.64% fewer, excluding nonlinear/reduction work. Compact SwiGLU's
2,260,992 weights are 1.38% more; compact GELU's 2,228,224 are 0.09% fewer.
Plain SiLU is a smaller ordinary dense control and is included in selection.

## Hardware and verification

The [short hardware screen](../../context_bank_result.md) passed both
corpora against all three full controls: 408.2 MiB allocated, 24.33% less than
GELU and 24.47% less than fused-activation SwiGLU. It does not support 1.2x faster
training or sustained/production-LLM resource claims. Numerical forward and
gradient references, finite differences, causality, locality, save/load and
prototype recovery passed. Production weights/counts and nonzero-coefficient
outputs/gradients matched CPU/CUDA prototypes exactly in recorded checks.
Production Trainer resume was exact for product, additive and plain on CPU.

## Language screen

Same width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight,
BF16. Each variant receives 2,000 updates / 4,096,000 supervised targets, rate
0.0006, warmup 200, AdamW decay 0.1, seed 101 and identical sampled windows.
Full eligible nonoverlapping validation windows omit the same short tail.
Test loss is unopened. Compare NLL only within a corpus; tokenizers differ.

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

Update/wall times are single observations with cold/runtime effects. Use repeated
profiles for the short hardware decision. Allocated differs from reserved and
device-wide usage. Full controls select the strongest measured language result.

## Registered removals

* tinystories/full-swiglu-kernel: trained 2.544240; half detectors removed 3.015803; train-mean output 9.090187.
* tinystories/full-swiglu-fused: trained 2.538486; half detectors removed 3.030469; train-mean output 9.228604.
* tinystories/full-gelu: trained 2.592983; half detectors removed 2.905189; train-mean output 7.583977.
* tinystories/narrow-swiglu-kernel: trained 2.615981; half detectors removed 3.312487; train-mean output 7.859602.
* tinystories/narrow-gelu: trained 2.623549; half detectors removed 3.010113; train-mean output 8.153454.
* tinystories/plain-silu: trained 2.673912; half detectors removed 3.035531; train-mean output 8.151718.
* tinystories/context-additive: trained 2.674518; half detectors removed 3.027494; train-mean output 7.562970. Same-backend coefficient zero: 2.678384; mean |tanh(a)| 0.016323.
* tinystories/context-product: trained 2.672406; half detectors removed 3.013075; train-mean output 7.487061. Same-backend coefficient zero: 2.674651; mean |tanh(a)| 0.059296.
* wikitext/full-swiglu-kernel: trained 4.239026; half detectors removed 4.758692; train-mean output 11.917761.
* wikitext/full-swiglu-fused: trained 4.237625; half detectors removed 4.771781; train-mean output 12.236745.
* wikitext/full-gelu: trained 4.269168; half detectors removed 4.625870; train-mean output 10.499083.
* wikitext/narrow-swiglu-kernel: trained 4.297109; half detectors removed 4.943720; train-mean output 9.549778.
* wikitext/narrow-gelu: trained 4.321449; half detectors removed 4.738714; train-mean output 10.652382.
* wikitext/plain-silu: trained 4.369832; half detectors removed 4.739931; train-mean output 9.129613.
* wikitext/context-additive: trained 4.365825; half detectors removed 4.727573; train-mean output 9.378052. Same-backend coefficient zero: 4.375125; mean |tanh(a)| 0.026915.
* wikitext/context-product: trained 4.367791; half detectors removed 4.735106; train-mean output 9.328992. Same-backend coefficient zero: 4.370834; mean |tanh(a)| 0.071203.

Coefficient zero preserves the identical fused/FP32 arithmetic. Half-detector
removal zeros even indices. Constant output is estimated from 32 fixed training
batches, seed 2027. Altered statistics diagnose dependence, not semantic roles;
compare retrained plain and additive controls before explaining richer inference.

Gating/channel modulation have established prior work; originality is unverified.
One-seed development results cannot qualify this architecture. Equal tuning,
fresh seeds, sustained resources and independent confirmation remain required.
Evidence: `records/context-bank-v1-*.json`,
`records/context-bank-language-v1-comparison.json`, frozen source snapshots and
raw interventions under `dump/context-bank-language-v1/`.
