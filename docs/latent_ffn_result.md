# Latent FFN: hardware result

[Registered hypothesis](latent_ffn_hypothesis.md). Retire this fixed recipe after
all 18 language runs and registered removals completed. It passed the short
resource screen but fails both dense quality margins on both corpora.

The completed TinyStories wide-recompute run scores NLL 2.726152, versus full
SwiGLU 2.538486 and compact kernel SwiGLU 2.606365. This already rules out the
required quality qualification for the fixed recipe. Conditional tuning and
confirmation will not launch. [The full model report](retired_models/latent_transformer/result.md)
preserves every control, resource measure and removal. Evidence is in
`records/latent-language-v1-comparison.json` and the frozen source/protocols.

WikiText wide-recompute is also complete: NLL 4.400809 versus strongest full
4.237625 and ordinary compact 4.288883. It misses both dense quality margins
there as well. Cached/recomputed NLL matches exactly on both corpora. Thin
TinyStories is 2.725223, calibrated thin 2.701581; WikiText is 4.438517 and
4.406500. No allocation passes. The candidate's full loss cost is 7.39%/3.85%
and compact loss deterioration is 4.60%/2.61% on TinyStories/WikiText. These results
do not identify which bottleneck or optimization effect caused the deficit.

Frozen-weight TinyStories removals score 3.153861 when 640 detector features are
zeroed before B, 8.857284 with unit expanded gates, and 7.796377 with training-mean
FFN outputs. This establishes sensitivity to those interventions under changed
statistics, not superiority to dense FFNs or an explanation of the quality deficit.

| Corpus | Allocated MiB | Reference | Memory reduction | Throughput ratio | Pass |
| --- | ---: | --- | ---: | ---: | --- |
| tinystories | 396.0 | full-gelu | 26.58% | 0.965x | True |
| tinystories | 396.0 | full-swiglu-fused | 29.37% | 0.977x | True |
| tinystories | 396.0 | full-swiglu-kernel | 26.72% | 0.970x | True |
| wikitext | 396.0 | full-gelu | 26.58% | 0.957x | True |
| wikitext | 396.0 | full-swiglu-fused | 29.37% | 0.982x | True |
| wikitext | 396.0 | full-swiglu-kernel | 26.72% | 0.967x | True |

42 profiles: seven variants, three alternating rounds on each corpus; 20 warmup and 100 measured updates. Maximum allocated memory and median throughput. BF16, batch eight, context 256, width 512, four layers, RTX 4070 Laptop. All updates finite, reserved memory below 6 GiB. Cold warmup is recorded separately.

2,490,368 FFN weights, 70.54% fewer than full SwiGLU. Forward projection work is 4,980,736 FLOPs/token; recomputation adds 2,621,440/token to backward, for 17,563,648 total training projection FLOPs/token. These counts exclude other operations.

CPU FP64 independent equations, all inner gradients, finite differences and interventions passed. CUDA FP32/BF16 reference checks, cached/recomputed outputs and gradients, complete-model causality/locality, save/load and exact CPU optimizer recovery passed. BF16 gradient tolerance and observed errors are recorded; numerical equality is not claimed for independent differently ordered gradients.

Evidence: `records/latent-v1-checks.json`, `records/latent-v1-profiles.json`; frozen source/protocol in `dump/latent-v1/`. Originality remains unverified; learned low-rank projections and recomputation are established techniques.
