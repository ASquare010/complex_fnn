# Polynomial feature sketches: research results

These are FFN-only experiments inside a fixed Transformer. Neither prototype has established a language-quality benefit or originality. Full equations, controls and failure criteria were registered before each implementation.

## polynomial-sketch-v1

[Registered hypothesis](polynomial_sketch_hypothesis.md).

| Corpus | Allocated MiB | Cut vs SwiGLU / GELU | Throughput vs SwiGLU / GELU | Screen |
| --- | ---: | ---: | ---: | --- |
| tinystories | 409.0 | 27.1% / 24.2% | 0.884x / 0.873x | fail |
| wikitext | 409.0 | 27.1% / 24.2% | 0.833x / 0.817x | fail |

Retire this fixed recipe before language training: its memory reduction does not compensate for violating the no-more-than-5% slowdown limit. Its language quality remains unknown.

Raw evidence: `records/polynomial-sketch-v1-profiles.json`. Exact staged sources are archived in `dump/polynomial-sketch-v1/source/`.

## single-sketch-v2

[Registered hypothesis](single_sketch_hypothesis.md).

| Corpus | Allocated MiB | Cut vs SwiGLU / GELU | Throughput vs SwiGLU / GELU | Screen |
| --- | ---: | ---: | ---: | --- |
| tinystories | 425.1 | 24.2% / 21.2% | 0.938x / 0.908x | fail |
| wikitext | 425.1 | 24.2% / 21.2% | 0.884x / 0.871x | fail |

Retire this fixed recipe before language training: its memory reduction does not compensate for violating the no-more-than-5% slowdown limit. Its language quality remains unknown.

Raw evidence: `records/single-sketch-v2-profiles.json`. Exact staged sources are archived in `dump/single-sketch-v2/source/`.

## Interpretation and verification

Same width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight, BF16. Three alternating-order rounds per corpus; each measures 100 updates after 20 warmup updates. Throughput uses medians; memory uses the largest allocated peak. All profiles stayed below 6 GiB reserved. These short laptop GPU measurements do not qualify sustained performance.

Both sketch variants have 2,099,200 FFN weights (75.17% fewer than full SwiGLU) and 4,194,304 projection FLOPs/token (75.19% fewer). Those FLOPs exclude FFTs and pointwise work. V1 recomputes two streams in backward; v2 caches one FP32-complex spectrum and changes the feature map. The change trades memory for fewer transforms and is not merely the same function implemented faster.

FP64 direct convolution, analytic gradients and finite differences passed. Staged complete Transformers passed token locality, causality, finite nonzero interaction gradients, identical initial controls, counts, save/load and exact CPU model/optimizer/sampler recovery. Production Trainer resume remains required before language training. Mathematical correctness does not demonstrate useful learned feature interactions.
