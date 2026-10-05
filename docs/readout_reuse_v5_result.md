# Readout reuse v5: resource results

Retire execution v5 before language: a resource gate fails.

[Hypothesis](readout_reuse_v5_hypothesis.md). Completed 54 profiles: three alternating rounds, 20 warmup/100 measured updates, same full-attention Transformer and real train windows on both corpora. Primary 2,506,752 FFN weights (70.35% fewer than full SwiGLU), reused P readout and learned groups of 32. Saved projected features with scalar-forward fusion and native grouped/scalar adjoints; native BF16 projections and FP32 response/group operations, TF32 disabled. Independent CPU equations/all gradients/finite differences/recovery and 192 CUDA fixtures plus 24 complete models and three saved-failure regressions passed. Language quality is unknown; no new primitive claimed.

| Corpus | Round | Variant | Seconds | Allocated MiB | Reserved MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| tinystories | 1 | full-swiglu-fused | 3.888 | 557.5 | 594.0 |
| tinystories | 1 | full-swiglu-kernel | 3.846 | 540.4 | 582.0 |
| tinystories | 1 | full-gelu | 3.857 | 539.4 | 610.0 |
| tinystories | 1 | reuse-group | 4.308 | 419.8 | 458.0 |
| tinystories | 1 | reuse-diagonal | 4.328 | 418.2 | 456.0 |
| tinystories | 1 | reuse-fixed | 3.949 | 413.1 | 466.0 |
| tinystories | 1 | reuse-fixed-matched | 4.063 | 414.2 | 466.0 |
| tinystories | 1 | reuse-untied | 4.015 | 404.2 | 464.0 |
| tinystories | 1 | reuse-cached | 4.191 | 531.7 | 586.0 |
| tinystories | 2 | reuse-cached | 4.277 | 531.7 | 586.0 |
| tinystories | 2 | reuse-untied | 4.018 | 404.2 | 464.0 |
| tinystories | 2 | reuse-fixed-matched | 4.203 | 414.2 | 466.0 |
| tinystories | 2 | reuse-fixed | 4.152 | 413.1 | 466.0 |
| tinystories | 2 | reuse-diagonal | 4.678 | 418.2 | 456.0 |
| tinystories | 2 | reuse-group | 4.442 | 419.8 | 458.0 |
| tinystories | 2 | full-gelu | 4.190 | 539.4 | 610.0 |
| tinystories | 2 | full-swiglu-kernel | 4.599 | 540.4 | 582.0 |
| tinystories | 2 | full-swiglu-fused | 4.122 | 560.7 | 602.0 |
| tinystories | 3 | full-swiglu-fused | 4.010 | 560.7 | 602.0 |
| tinystories | 3 | full-swiglu-kernel | 3.875 | 540.4 | 582.0 |
| tinystories | 3 | full-gelu | 3.862 | 539.4 | 610.0 |
| tinystories | 3 | reuse-group | 4.450 | 419.8 | 458.0 |
| tinystories | 3 | reuse-diagonal | 4.548 | 418.2 | 456.0 |
| tinystories | 3 | reuse-fixed | 4.189 | 413.1 | 466.0 |
| tinystories | 3 | reuse-fixed-matched | 4.091 | 414.2 | 466.0 |
| tinystories | 3 | reuse-untied | 5.052 | 404.2 | 464.0 |
| tinystories | 3 | reuse-cached | 3.980 | 531.7 | 586.0 |
| wikitext | 1 | full-swiglu-fused | 3.965 | 560.7 | 602.0 |
| wikitext | 1 | full-swiglu-kernel | 4.134 | 540.4 | 582.0 |
| wikitext | 1 | full-gelu | 4.037 | 539.4 | 610.0 |
| wikitext | 1 | reuse-group | 4.756 | 419.8 | 458.0 |
| wikitext | 1 | reuse-diagonal | 4.797 | 418.2 | 456.0 |
| wikitext | 1 | reuse-fixed | 4.280 | 413.1 | 466.0 |
| wikitext | 1 | reuse-fixed-matched | 4.393 | 414.2 | 466.0 |
| wikitext | 1 | reuse-untied | 4.291 | 404.2 | 464.0 |
| wikitext | 1 | reuse-cached | 4.382 | 531.7 | 586.0 |
| wikitext | 2 | reuse-cached | 4.465 | 531.7 | 586.0 |
| wikitext | 2 | reuse-untied | 4.376 | 404.2 | 464.0 |
| wikitext | 2 | reuse-fixed-matched | 4.262 | 414.2 | 466.0 |
| wikitext | 2 | reuse-fixed | 4.249 | 413.1 | 466.0 |
| wikitext | 2 | reuse-diagonal | 4.936 | 418.2 | 456.0 |
| wikitext | 2 | reuse-group | 4.678 | 419.8 | 458.0 |
| wikitext | 2 | full-gelu | 4.064 | 539.4 | 610.0 |
| wikitext | 2 | full-swiglu-kernel | 4.118 | 540.4 | 582.0 |
| wikitext | 2 | full-swiglu-fused | 4.047 | 560.7 | 602.0 |
| wikitext | 3 | full-swiglu-fused | 4.201 | 560.7 | 602.0 |
| wikitext | 3 | full-swiglu-kernel | 4.173 | 540.4 | 582.0 |
| wikitext | 3 | full-gelu | 4.096 | 539.4 | 610.0 |
| wikitext | 3 | reuse-group | 4.657 | 419.8 | 458.0 |
| wikitext | 3 | reuse-diagonal | 4.759 | 418.2 | 456.0 |
| wikitext | 3 | reuse-fixed | 4.021 | 413.1 | 466.0 |
| wikitext | 3 | reuse-fixed-matched | 3.995 | 414.2 | 466.0 |
| wikitext | 3 | reuse-untied | 4.126 | 404.2 | 464.0 |
| wikitext | 3 | reuse-cached | 4.180 | 531.7 | 586.0 |

tinystories: resource passes False.

Against full-swiglu-fused: throughput 0.903x; memory reduction 25.12%; passes False.

Against full-swiglu-kernel: throughput 0.872x; memory reduction 22.32%; passes False.

Against full-gelu: throughput 0.869x; memory reduction 22.17%; passes False.


wikitext: resource passes False.

Against full-swiglu-fused: throughput 0.865x; memory reduction 25.12%; passes False.

Against full-swiglu-kernel: throughput 0.884x; memory reduction 22.32%; passes False.

Against full-gelu: throughput 0.869x; memory reduction 22.17%; passes False.

Evidence: records/readout-reuse-v5-{cpu-checks,cuda-checks,profiles}.json; frozen sources/protocol and all profiles under dump/readout-reuse-v5. Controls are not post-hoc candidates. Sustained resources and language quality remain unconfirmed.
