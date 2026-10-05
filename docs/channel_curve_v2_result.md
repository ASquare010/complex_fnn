# Coupled channel curves version 2: retired after language screen

[Registration](channel_curve_v2_hypothesis.md). The fused execution passed the short hardware screen, but the completed 18-run language screen fails the quality contract. Retire this fixed recipe; no tuning/confirmation launch. Sustained resources remain unconfirmed.

The first completed TinyStories candidate scores NLL 2.626786, versus strongest
full 2.538486 and matched compact kernel SwiGLU 2.613739. It already misses both
dense quality margins, so this fixed recipe cannot qualify or advance to tuning.
All mechanism controls, WikiText and registered removals completed unchanged.

The completed TinyStories self-gate/fixed-shape/plain controls score
2.598287/2.628363/2.673912. WikiText candidate 4.298236 also misses both margins
against full 4.237625 and compact kernel 4.308013. WikiText self-gate/fixed-shape/plain
score 4.288112/4.304249/4.369832. The self-gate control beats the coupled candidate
on both corpora; this does not support the proposed benefit from neighboring gates.
The next [basis-readout hypothesis](basis_readout_hypothesis.md) tests separate
learned readout weights for nonlinear responses; it is unmeasured.

| Corpus | Reference | Allocated MiB | Memory reduction | Throughput ratio | Pass |
| --- | --- | ---: | ---: | ---: | --- |
| tinystories | full-gelu | 401.2 | 25.62% | 1.024x | True |
| tinystories | full-swiglu-fused | 401.2 | 28.45% | 1.043x | True |
| tinystories | full-swiglu-kernel | 401.2 | 25.77% | 1.045x | True |
| wikitext | full-gelu | 401.2 | 25.62% | 1.037x | True |
| wikitext | full-swiglu-fused | 401.2 | 28.45% | 1.027x | True |
| wikitext | full-swiglu-kernel | 401.2 | 25.77% | 1.044x | True |

42 profiles: seven variants, three alternating-order rounds on each corpus, 20 warmup and 100 timed updates. Maximum allocated peak and median throughput; cold compilation separately recorded. Same width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight, BF16, four threads, RTX 4070 Laptop. All updates finite and reserved memory below 6 GiB. This is not sustained-resource confirmation or a 1.2x training-speed claim.

2,121,728 FFN weights, 74.90% fewer than full SwiGLU. Forward projection work 4,194,304 FLOPs/token; training projection work 12,582,912 plus scalar activation/reconstruction/reduction work. The fused input adjoint evaluates additional scalar nonlinearities to avoid token-sized derivative banks. Windows backend uses the bundled NVRTC and Torch-owned tensors on the calling stream with the existing Torch CUDA context.

CPU FP64 independent equations, gradients/finite differences, quadrature convergence and all four complete-model causality/locality/save-load/exact optimizer-recovery checks passed. CUDA FP32/BF16 independent adjoints, complete/incomplete tiles, self-gate paths and all complete-model/removal references passed with recorded tolerances/errors. BF16 rounding bits match Torch casts on tested halfway ties, signed zeros, subnormals and infinities; NaN classification matches.

Production CPU/CUDA weights/counts/outputs/gradients match staged prototypes exactly in recorded checks. Shared-Trainer CPU checkpoint recovery is exact for all four modes. Code formatting/lint passed before source freeze. Implementation preparation failures and their repairs are retained rather than treated as resource results.

Evidence: `records/channel-curve-v2-checks.json`, `records/channel-curve-v2-profiles.json`, `records/channel-curve-v2-integration-checks.json`, `records/channel-curve-v2-development-errors.json`; frozen source/protocol under `dump/channel-curve-v2/`. The [model result](../src/models/channel_curve_transformer/result.md) tracks the 18-run language screen and removals. Learned activations/gating/recomputation are established; originality remains unverified.
