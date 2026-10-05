# Paired readout v1: resource results

Retire execution v1 before language: a resource gate fails.

[Hypothesis](paired_readout_hypothesis.md). Completed 48 profiles: three alternating rounds, 20 warmup/100 measured updates, same full-attention Transformer and real train windows on both corpora. Primary 2,359,296 FFN weights (72.09% fewer than full SwiGLU), reused P readout and fixed signed pair exchange. Saved projected features and mixed BF16 responses, scalar-forward scatter fusion and native adjoints; native BF16 projections and FP32 response/group operations, TF32 disabled. Independent CPU equations/all gradients/finite differences/recovery and 120 CUDA fixtures plus 20 complete models passed. Language quality is unknown; no new primitive claimed.

| Corpus | Round | Variant | Seconds | Allocated MiB | Reserved MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| tinystories | 1 | full-swiglu-fused | 3.658 | 557.5 | 594.0 |
| tinystories | 1 | full-swiglu-kernel | 3.937 | 540.4 | 582.0 |
| tinystories | 1 | full-gelu | 4.127 | 539.4 | 610.0 |
| tinystories | 1 | pair-skew | 4.131 | 432.6 | 486.0 |
| tinystories | 1 | pair-symmetric | 4.092 | 432.6 | 486.0 |
| tinystories | 1 | pair-diagonal | 4.138 | 432.6 | 486.0 |
| tinystories | 1 | pair-recompute | 4.297 | 412.5 | 466.0 |
| tinystories | 1 | pair-untied | 4.079 | 410.3 | 464.0 |
| tinystories | 2 | pair-untied | 4.089 | 410.3 | 464.0 |
| tinystories | 2 | pair-recompute | 4.204 | 412.5 | 466.0 |
| tinystories | 2 | pair-diagonal | 4.271 | 432.6 | 486.0 |
| tinystories | 2 | pair-symmetric | 4.255 | 432.6 | 486.0 |
| tinystories | 2 | pair-skew | 4.358 | 432.6 | 486.0 |
| tinystories | 2 | full-gelu | 4.121 | 539.4 | 610.0 |
| tinystories | 2 | full-swiglu-kernel | 4.454 | 540.4 | 582.0 |
| tinystories | 2 | full-swiglu-fused | 4.319 | 560.7 | 602.0 |
| tinystories | 3 | full-swiglu-fused | 4.601 | 560.7 | 602.0 |
| tinystories | 3 | full-swiglu-kernel | 4.591 | 540.4 | 582.0 |
| tinystories | 3 | full-gelu | 4.434 | 539.4 | 610.0 |
| tinystories | 3 | pair-skew | 4.760 | 432.6 | 486.0 |
| tinystories | 3 | pair-symmetric | 4.758 | 432.6 | 486.0 |
| tinystories | 3 | pair-diagonal | 4.526 | 432.6 | 486.0 |
| tinystories | 3 | pair-recompute | 4.682 | 412.5 | 466.0 |
| tinystories | 3 | pair-untied | 4.342 | 410.3 | 464.0 |
| wikitext | 1 | full-swiglu-fused | 4.237 | 560.7 | 602.0 |
| wikitext | 1 | full-swiglu-kernel | 4.469 | 540.4 | 582.0 |
| wikitext | 1 | full-gelu | 4.313 | 539.4 | 610.0 |
| wikitext | 1 | pair-skew | 4.356 | 432.6 | 486.0 |
| wikitext | 1 | pair-symmetric | 4.410 | 432.6 | 486.0 |
| wikitext | 1 | pair-diagonal | 4.389 | 432.6 | 486.0 |
| wikitext | 1 | pair-recompute | 4.405 | 412.5 | 466.0 |
| wikitext | 1 | pair-untied | 4.681 | 410.3 | 464.0 |
| wikitext | 2 | pair-untied | 4.614 | 410.3 | 464.0 |
| wikitext | 2 | pair-recompute | 4.518 | 412.5 | 466.0 |
| wikitext | 2 | pair-diagonal | 4.409 | 432.6 | 486.0 |
| wikitext | 2 | pair-symmetric | 4.423 | 432.6 | 486.0 |
| wikitext | 2 | pair-skew | 4.477 | 432.6 | 486.0 |
| wikitext | 2 | full-gelu | 4.163 | 539.4 | 610.0 |
| wikitext | 2 | full-swiglu-kernel | 4.416 | 540.4 | 582.0 |
| wikitext | 2 | full-swiglu-fused | 4.292 | 560.7 | 602.0 |
| wikitext | 3 | full-swiglu-fused | 4.407 | 560.7 | 602.0 |
| wikitext | 3 | full-swiglu-kernel | 4.864 | 540.4 | 582.0 |
| wikitext | 3 | full-gelu | 4.852 | 539.4 | 610.0 |
| wikitext | 3 | pair-skew | 4.612 | 432.6 | 486.0 |
| wikitext | 3 | pair-symmetric | 4.730 | 432.6 | 486.0 |
| wikitext | 3 | pair-diagonal | 4.651 | 432.6 | 486.0 |
| wikitext | 3 | pair-recompute | 4.654 | 412.5 | 466.0 |
| wikitext | 3 | pair-untied | 4.601 | 410.3 | 464.0 |

tinystories: resource passes False.

Against full-swiglu-fused: throughput 0.991x; memory reduction 22.85%; passes True.

Against full-swiglu-kernel: throughput 1.022x; memory reduction 19.96%; passes False.

Against full-gelu: throughput 0.947x; memory reduction 19.80%; passes False.


wikitext: resource passes False.

Against full-swiglu-fused: throughput 0.959x; memory reduction 22.85%; passes True.

Against full-swiglu-kernel: throughput 0.998x; memory reduction 19.96%; passes False.

Against full-gelu: throughput 0.963x; memory reduction 19.80%; passes False.

Evidence: records/paired-readout-v1-{cpu-checks,cuda-checks,profiles}.json; frozen sources/protocol and all profiles under dump/paired-readout-v1. Controls are not post-hoc candidates. Sustained resources and language quality remain unconfirmed.
