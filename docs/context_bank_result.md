# Independent context-bank FFN: resource result

[Registered hypothesis](context_bank_hypothesis.md).

Advance to production equivalence and the registered language screen; quality remains unknown.

Subsequent production integration passed: weights, counts and nonzero-coefficient
CPU/CUDA outputs and gradients match the prototype exactly in the recorded checks.
Shared-Trainer CPU resume is exact for product, additive and plain modes.
The [production model report](retired_models/context_bank_transformer/result.md)
records the completed eight-variant, two-corpus language screen.

## Completed language decision

Retire this fixed recipe. TinyStories product NLL 2.672406 is worse than full
SwiGLU 2.538486 and compact SwiGLU 2.615981. WikiText product NLL 4.367791 is
worse than full SwiGLU 4.237625 and compact SwiGLU 4.297109. Both required language
margins fail on both corpora. It also trails additive context on WikiText.

Same-backend coefficient zero changes product NLL to 2.674651 / 4.370834 on
TinyStories / WikiText. These small removal effects do not establish useful
richer inference computation. Frozen-weight dependence cannot substitute for
superiority over independently trained controls. Test loss remains unopened.

The [signed-context hypothesis](signed_context_hypothesis.md) separately tests
active signed predicates and open/closed gain controls at a larger, preregistered
detector budget. It does not change this experiment's failed decision.

## Computation and cost

512 detector features and 64 independently learned context predicates per layer.
Eight detectors share one context. `r_i = SiLU(u_i)*(1+tanh(a_i)*tanh(v_group(i)))`;
an independent learned output projection maps this to the residual stream.
An exactly matched additive control uses `SiLU(u_i)+tanh(a_i)*tanh(v_group(i))`.
The product is a restricted gating allocation; it does not create information.

2,230,272 FFN weights: 73.62% fewer than full SwiGLU. 4,456,448 projection FLOPs
per token: 73.64% fewer, excluding nonlinear and gradient-reduction work.

## Short hardware screen

| Corpus | Allocated MiB | Cut vs native SwiGLU / kernel SwiGLU / GELU | Throughput vs native / kernel / GELU | Screen |
| --- | ---: | ---: | ---: | --- |
| tinystories | 408.2 | 27.20% / 24.47% / 24.33% | 1.040x / 1.051x / 1.028x | pass |
| wikitext | 408.2 | 27.20% / 24.47% / 24.33% | 1.017x / 1.061x / 1.018x | pass |

Same width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight,
BF16 and four CPU threads. RTX 4070 Laptop GPU. Three alternating-order rounds
on each corpus, 20 warmup plus 100 measured updates. Largest allocated peak and
median supervised targets/second. Cold warmup/compilation is separately recorded.
All runs finite and below 6 GiB reserved. This short screen does not establish
sustained resource savings or a 1.2x speed improvement.

## Numerical and model verification

CPU FP64 finite differences and independent detector-to-context selection passed.
CUDA FP32/BF16 output and gradient checks passed. Independent FP32 scatter
accumulation in the reference matches the fused context-gradient reduction before
one BF16 cast. BF16 output and projected-bank gradients matched exactly in these
checks; coefficient gradient differences were below 5e-7. Full-model causality,
token locality, initial controls, save/load, counts and exact CPU optimizer/model
recovery passed. Production Trainer integration/resume is a separate required gate.

Gating and channel modulation have established prior work. Originality remains
unverified. No language, reasoning or production-LLM accuracy claim follows from
these tests. Evidence: `records/context-bank-v1-checks.json` and
`records/context-bank-v1-profiles.json`; frozen source/protocol under
`dump/context-bank-v1/`. Test loss remains unopened.
