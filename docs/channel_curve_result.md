# Coupled channel curves: resource result

[Registered hypothesis](channel_curve_hypothesis.md). Retire execution version 1 before language training: all numerical checks passed, but the short hardware screen fails the slowdown limit on both corpora. Language quality is unmeasured.

| Corpus | Reference | Allocated MiB | Memory reduction | Throughput ratio | Pass |
| --- | --- | ---: | ---: | ---: | --- |
| tinystories | full-gelu | 399.2 | 25.99% | 0.812x | False |
| tinystories | full-swiglu-fused | 399.2 | 28.80% | 0.847x | False |
| tinystories | full-swiglu-kernel | 399.2 | 26.14% | 0.829x | False |
| wikitext | full-gelu | 399.2 | 25.99% | 0.822x | False |
| wikitext | full-swiglu-fused | 399.2 | 28.80% | 0.844x | False |
| wikitext | full-swiglu-kernel | 399.2 | 26.14% | 0.850x | False |

42 profiles: seven variants, three alternating rounds on each corpus; 20 warmup plus 100 timed updates. Maximum allocated peak and median throughput. Same width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight, BF16, four threads, RTX 4070 Laptop. All updates finite, reserved memory below 6 GiB. Cold warmup recorded separately.

2,121,728 FFN weights, 74.90% fewer than full SwiGLU. Forward projection work 4,194,304 FLOPs/token; total training projection work 12,582,912 plus scalar activation/reduction work. Reduced projection FLOPs do not imply faster training.

CPU FP64 independent equations, gradients/finite differences and quadrature-convergence checks passed. CUDA FP32/BF16 independent width-512 references and complete-model output/gradient/removal checks passed at recorded tolerances. All four complete modes passed causality/locality, save/load and exact CPU optimizer recovery. Error magnitudes and sampled initialization moments are retained.

Evidence: `records/channel-curve-v1-checks.json`, `records/channel-curve-v1-profiles.json`, frozen source/protocol under `dump/channel-curve-v1/`. No production integration or language runs for this execution recipe. Learned activations and gating are established; originality is unverified. A different execution requires a new registration and complete numerical/resource verification.
