# Readout reuse v1: resource results

Retire execution v1 before language: a resource gate fails.

[Hypothesis](readout_reuse_hypothesis.md). Completed 54 profiles: three alternating rounds, 20 warmup/100 measured updates, same full-attention Transformer and real train windows on both corpora. Primary 2,506,752 FFN weights (70.35% fewer than full SwiGLU), reused P readout and learned groups of 32. Ordinary nonreentrant FFN recomputation; native BF16 projections and FP32 response/group operations, TF32 disabled. Independent CPU equations/all gradients/finite differences/recovery and 144 CUDA fixtures plus 24 complete models passed. Language quality is unknown; no new primitive claimed.

| Corpus | Round | Variant | Seconds | Allocated MiB | Reserved MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| tinystories | 1 | full-swiglu-fused | 4.221 | 557.5 | 594.0 |
| tinystories | 1 | full-swiglu-kernel | 4.254 | 540.4 | 582.0 |
| tinystories | 1 | full-gelu | 4.127 | 539.4 | 610.0 |
| tinystories | 1 | reuse-group | 5.431 | 396.1 | 438.0 |
| tinystories | 1 | reuse-diagonal | 5.522 | 394.5 | 434.0 |
| tinystories | 1 | reuse-fixed | 4.986 | 394.7 | 434.0 |
| tinystories | 1 | reuse-fixed-matched | 5.160 | 393.9 | 434.0 |
| tinystories | 1 | reuse-untied | 4.900 | 395.7 | 444.0 |
| tinystories | 1 | reuse-cached | 4.446 | 531.7 | 586.0 |
| tinystories | 2 | reuse-cached | 4.527 | 531.7 | 586.0 |
| tinystories | 2 | reuse-untied | 4.840 | 395.7 | 444.0 |
| tinystories | 2 | reuse-fixed-matched | 4.908 | 393.9 | 434.0 |
| tinystories | 2 | reuse-fixed | 4.838 | 394.7 | 434.0 |
| tinystories | 2 | reuse-diagonal | 5.566 | 394.5 | 434.0 |
| tinystories | 2 | reuse-group | 5.581 | 396.1 | 438.0 |
| tinystories | 2 | full-gelu | 4.113 | 539.4 | 610.0 |
| tinystories | 2 | full-swiglu-kernel | 4.220 | 540.4 | 582.0 |
| tinystories | 2 | full-swiglu-fused | 4.137 | 560.7 | 602.0 |
| tinystories | 3 | full-swiglu-fused | 4.034 | 560.7 | 602.0 |
| tinystories | 3 | full-swiglu-kernel | 4.219 | 540.4 | 582.0 |
| tinystories | 3 | full-gelu | 4.010 | 539.4 | 610.0 |
| tinystories | 3 | reuse-group | 5.395 | 396.1 | 438.0 |
| tinystories | 3 | reuse-diagonal | 5.341 | 394.5 | 434.0 |
| tinystories | 3 | reuse-fixed | 5.108 | 394.7 | 434.0 |
| tinystories | 3 | reuse-fixed-matched | 5.115 | 393.9 | 434.0 |
| tinystories | 3 | reuse-untied | 4.958 | 395.7 | 444.0 |
| tinystories | 3 | reuse-cached | 4.691 | 531.7 | 586.0 |
| wikitext | 1 | full-swiglu-fused | 4.553 | 560.7 | 602.0 |
| wikitext | 1 | full-swiglu-kernel | 4.232 | 540.4 | 582.0 |
| wikitext | 1 | full-gelu | 4.372 | 539.4 | 610.0 |
| wikitext | 1 | reuse-group | 5.829 | 396.1 | 438.0 |
| wikitext | 1 | reuse-diagonal | 5.556 | 394.5 | 434.0 |
| wikitext | 1 | reuse-fixed | 4.963 | 394.7 | 434.0 |
| wikitext | 1 | reuse-fixed-matched | 4.993 | 393.9 | 434.0 |
| wikitext | 1 | reuse-untied | 5.000 | 395.7 | 444.0 |
| wikitext | 1 | reuse-cached | 4.642 | 531.7 | 586.0 |
| wikitext | 2 | reuse-cached | 4.568 | 531.7 | 586.0 |
| wikitext | 2 | reuse-untied | 4.945 | 395.7 | 444.0 |
| wikitext | 2 | reuse-fixed-matched | 5.107 | 393.9 | 434.0 |
| wikitext | 2 | reuse-fixed | 4.913 | 394.7 | 434.0 |
| wikitext | 2 | reuse-diagonal | 5.680 | 394.5 | 434.0 |
| wikitext | 2 | reuse-group | 5.447 | 396.1 | 438.0 |
| wikitext | 2 | full-gelu | 4.279 | 539.4 | 610.0 |
| wikitext | 2 | full-swiglu-kernel | 4.392 | 540.4 | 582.0 |
| wikitext | 2 | full-swiglu-fused | 4.380 | 560.7 | 602.0 |
| wikitext | 3 | full-swiglu-fused | 4.348 | 560.7 | 602.0 |
| wikitext | 3 | full-swiglu-kernel | 4.249 | 540.4 | 582.0 |
| wikitext | 3 | full-gelu | 4.288 | 539.4 | 610.0 |
| wikitext | 3 | reuse-group | 5.412 | 396.1 | 438.0 |
| wikitext | 3 | reuse-diagonal | 5.417 | 394.5 | 434.0 |
| wikitext | 3 | reuse-fixed | 4.951 | 394.7 | 434.0 |
| wikitext | 3 | reuse-fixed-matched | 5.069 | 393.9 | 434.0 |
| wikitext | 3 | reuse-untied | 5.040 | 395.7 | 444.0 |
| wikitext | 3 | reuse-cached | 4.656 | 531.7 | 586.0 |

tinystories: resource passes False.

Against full-swiglu-fused: throughput 0.762x; memory reduction 29.35%; passes False.

Against full-swiglu-kernel: throughput 0.777x; memory reduction 26.71%; passes False.

Against full-gelu: throughput 0.757x; memory reduction 26.56%; passes False.


wikitext: resource passes False.

Against full-swiglu-fused: throughput 0.804x; memory reduction 29.35%; passes False.

Against full-swiglu-kernel: throughput 0.780x; memory reduction 26.71%; passes False.

Against full-gelu: throughput 0.787x; memory reduction 26.56%; passes False.

Evidence: records/readout-reuse-v1-{cpu-checks,cuda-checks,profiles}.json; frozen sources/protocol and all profiles under dump/readout-reuse-v1. Controls are not post-hoc candidates. Sustained resources and language quality remain unconfirmed.
