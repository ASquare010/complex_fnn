# Group-product FFN: results

The proposed FFN adds products between a feature and the sum of its group neighbors to a direct SiLU path. Learned input/output projections and per-feature coefficients train the transformation. This is a hypothesis about useful feature combinations; it does not create new information. The fused v4 recipe passes its short hardware screen but misses the language targets on both corpora and is retired. No improved quality or novelty claim is established. The [production model report](retired_models/group_product_transformer/result.md) contains the completed two-corpus language results and inference removals.

## group-product-v1

[Registered hypothesis](group_product_hypothesis.md).

| Corpus | Allocated MiB | Cut vs SwiGLU / GELU | Throughput vs SwiGLU / GELU | Screen |
| --- | ---: | ---: | ---: | --- |
| tinystories | 433.0 | 22.78% / 19.73% | 0.988x / 0.963x | fail |
| wikitext | 433.0 | 22.78% / 19.73% | 1.004x / 0.953x | fail |

Retire this fixed size before language training. The >=20% memory gate is unchanged; a near miss is a failure. Language quality is unknown.

Evidence: `records/group-product-v1-profiles.json`; frozen staged source in `dump/group-product-v1/source/`.

## group-product-v2

[Registered hypothesis](group_product_v2_hypothesis.md).

| Corpus | Allocated MiB | Cut vs SwiGLU / GELU | Throughput vs SwiGLU / GELU | Screen |
| --- | ---: | ---: | ---: | --- |
| tinystories | 428.5 | 23.58% / 20.56% | 0.951x / 0.932x | fail |
| wikitext | 428.5 | 23.58% / 20.56% | 0.982x / 0.949x | fail |

Retire this fixed size before language training. The >=20% memory gate is unchanged; a near miss is a failure. Language quality is unknown.

Evidence: `records/group-product-v2-profiles.json`; frozen staged source in `dump/group-product-v2/source/`.

## group-product-v3

[Registered hypothesis](group_product_v3_hypothesis.md). Same hidden 512 and equation as v1, with a custom gradient that recomputes the neighbor signal rather than saving it.

| Corpus | Allocated MiB | Cut vs SwiGLU / GELU | Throughput vs SwiGLU / GELU | Screen |
| --- | ---: | ---: | ---: | --- |
| tinystories | 415.0 | 25.99% / 23.06% | 0.922x / 0.957x | fail |
| wikitext | 415.0 | 25.99% / 23.06% | 0.983x / 0.963x | pass |

Retire the fixed execution recipe before language training: it exceeds the 5% slowdown limit against full SwiGLU on TinyStories. Language quality is unknown. Evidence: `records/group-product-v3-profiles.json`; frozen source in `dump/group-product-v3/source/`.

## group-product-v4

[Registered hypothesis](group_product_v4_hypothesis.md). Fuse SiLU, the bounded coefficient, the neighbor product and their derivatives using PyTorch's bundled runtime CUDA compiler. Compute activation arithmetic and group sums in FP32, then cast to BF16. CPU FP64 equations are unchanged. Include a dense SwiGLU control that also fuses its activation and derivatives.

| Corpus | Allocated MiB | Cut vs native SwiGLU / kernel SwiGLU / GELU | Throughput vs native / kernel / GELU | Screen |
| --- | ---: | ---: | ---: | ---: | --- |
| tinystories | 402.0 | 28.31% / 25.62% / 25.48% | 1.065x / 1.036x / 1.024x | pass |
| wikitext | 402.0 | 28.31% / 25.62% / 25.48% | 1.025x / 1.010x / 1.012x | pass |

This qualifies for the language screen through the memory branch, not a 1.2x speed gain. Production integration and exact CPU Trainer resume passed before training. Eight variants receive the same 2,000-update budget on each corpus. TinyStories group NLL is 2.651852 versus full SwiGLU 2.538486, compact SwiGLU 2.617045 and the same-sized learned-square control 2.645681. This misses both registered language margins and fails to demonstrate a benefit of grouped cross-products over learned self-squares.

WikiText group NLL is 4.333677 versus full SwiGLU 4.237625, compact SwiGLU 4.308211 and learned square 4.326767. This also misses both language margins and trails the square control. Retire the fixed recipe. At frozen trained weights, zeroing the coefficients with the identical arithmetic backend raises group NLL to 2.795028 on TinyStories and 4.561932 on WikiText. The interaction is used at inference, but reliance on it does not establish superiority over a separately trained control or richer semantic computation.

Evidence: `records/group-product-v4-profiles.json`, `records/group-product-v4-integration-checks.json`, `records/group-product-v4-activation-checks.json`, and `records/group-product-language-v4-comparison.json`. The [model report](retired_models/group_product_transformer/result.md) records the complete language screen and coefficient-zero interventions with the same arithmetic backend. Retain the measured implementation and evidence; an efficient kernel does not establish a better FFN architecture.

## Scope and verification

Same width 512, four-layer, 16-head backbone, context 256, vocabulary 4096, batch eight, BF16. Three alternating-order rounds on each corpus, 20 warmup plus 100 measured steps. Largest allocated peak and median throughput. Device: RTX 4070 Laptop GPU. All profiles fit below 6 GiB reserved. Short hardware screens require sustained independent confirmation before a resource claim.

V1 hidden 512 / 32 groups: 2,099,200 FFN weights, 75.17% fewer than full SwiGLU. V2 hidden 480 / 30 groups: 1,968,000 weights, 76.72% fewer. Both keep 16 features per group. Projection FLOPs exclude group reductions and products. V2 was selected from hardware evidence before any group-product language results, not by rounding or relaxing the v1 failure.

CPU equation and gradient comparisons against explicit dense complete-graph adjacency, finite differences, full-Transformer causality/locality, initial controls, counts, save/load and prototype exact resume passed. V4 additionally passed production equivalence, CPU Trainer resume and CUDA FP32/BF16 forward/gradient reference checks. The FFN includes established gating/quadratic ideas; originality of this recipe is unverified.
