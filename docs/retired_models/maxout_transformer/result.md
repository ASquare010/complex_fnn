# Maxout and compact FFNs: language screen v1

Status: 22 / 22 runs complete. This is single-seed development.
[Registered hypothesis and acceptance contract](../../language_screen_v1.md).

## Decision

No candidate meets both language margins on both corpora. Retire these fixed recipes and register a revision before further training.

tinystories: strongest full `full-swiglu`; strongest compact `narrow-swiglu-fused`. maxout: full loss cost 3.93%, compact loss gain -0.56%. curve-only: full loss cost 6.25%, compact loss gain -2.81%. blockshuffle: full loss cost 2.55%, compact loss gain 0.77%.

wikitext: strongest full `full-swiglu`; strongest compact `narrow-swiglu`. maxout: full loss cost 3.57%, compact loss gain -1.11%. curve-only: full loss cost 4.53%, compact loss gain -2.05%. blockshuffle: full loss cost 2.53%, compact loss gain -0.09%.

## Quality and complete-run resource measurements

Same width-192, four-layer, six-head Transformer; context 256, vocabulary 4096,
batch eight, BF16 CUDA, seed 101, AdamW rate 0.0006, 200 warmup and 2,000 updates.
Every completed run trained on 4,096,000 supervised language targets. Validation
covers all complete nonoverlapping windows, discarding the same short tail for
all models. Test losses remain unscored. Compare NLL within each corpus only.

| Corpus | Variant | Validation NLL | Targets | Update seconds | Allocated / reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| tinystories | full-swiglu | 2.883340 | 199,936 | 68.34 | 252.95 / 294.00 |
| tinystories | full-swiglu-fused | 2.884440 | 199,936 | 69.85 | 249.95 / 292.00 |
| tinystories | full-gelu | 2.979904 | 199,936 | 69.55 | 241.20 / 290.00 |
| tinystories | narrow-swiglu | 2.980331 | 199,936 | 60.01 | 218.00 / 252.00 |
| tinystories | narrow-swiglu-fused | 2.979796 | 199,936 | 66.18 | 216.56 / 264.00 |
| tinystories | narrow-gelu | 3.003410 | 199,936 | 66.01 | 212.62 / 250.00 |
| tinystories | plain-silu | 3.067467 | 199,936 | 67.27 | 212.41 / 250.00 |
| tinystories | curve-only | 3.063614 | 199,936 | 73.63 | 222.93 / 260.00 |
| tinystories | blockshuffle | 2.956800 | 199,936 | 98.13 | 284.67 / 336.00 |
| tinystories | maxout | 2.996539 | 199,936 | 68.03 | 215.11 / 262.00 |
| tinystories | meanout | 3.121888 | 199,936 | 50.51 | 208.11 / 242.00 |
| wikitext | full-swiglu | 4.523139 | 322,560 | 71.89 | 253.87 / 296.00 |
| wikitext | full-swiglu-fused | 4.523150 | 322,560 | 75.25 | 250.87 / 292.00 |
| wikitext | full-gelu | 4.641271 | 322,560 | 73.00 | 242.12 / 292.00 |
| wikitext | narrow-swiglu | 4.633178 | 322,560 | 78.83 | 218.92 / 254.00 |
| wikitext | narrow-swiglu-fused | 4.633594 | 322,560 | 45.92 | 217.48 / 264.00 |
| wikitext | narrow-gelu | 4.677101 | 322,560 | 69.12 | 213.55 / 250.00 |
| wikitext | plain-silu | 4.740294 | 322,560 | 89.27 | 213.34 / 250.00 |
| wikitext | curve-only | 4.728093 | 322,560 | 84.00 | 223.86 / 260.00 |
| wikitext | blockshuffle | 4.637398 | 322,560 | 123.26 | 284.67 / 336.00 |
| wikitext | maxout | 4.684398 | 322,560 | 79.02 | 215.11 / 262.00 |
| wikitext | meanout | 4.819802 | 322,560 | 76.73 | 208.11 / 242.00 |

Update times include batching/startup and are single measurements. Allocated
memory is the peak during training; reserved and device-wide usage differ.

## Repeated hardware screen

Actual TinyStories batches; 20 warmup + 50 timed updates; three rounds with order
reversed in round two. All losses and gradients were finite and fit under 6 GiB
reserved VRAM. These short profiles do not qualify sustained speed on both corpora.

| Variant | Median targets/s [min, max] | Peak allocated / reserved MiB |
| --- | ---: | ---: |
| full-swiglu | 66,903 [59,690, 68,668] | 252.95 / 294.00 |
| full-swiglu-fused | 67,432 [65,216, 69,928] | 249.95 / 292.00 |
| full-gelu | 69,307 [62,138, 73,886] | 241.20 / 290.00 |
| narrow-swiglu | 60,551 [58,405, 68,320] | 218.00 / 252.00 |
| narrow-swiglu-fused | 69,769 [67,642, 69,920] | 216.56 / 264.00 |
| narrow-gelu | 69,535 [68,351, 69,818] | 212.62 / 250.00 |
| plain-silu | 65,493 [63,379, 71,440] | 212.41 / 250.00 |
| curve-only | 54,463 [48,463, 58,006] | 222.93 / 260.00 |
| blockshuffle | 43,910 [42,334, 44,519] | 282.75 / 336.00 |
| maxout | 66,901 [62,873, 68,594] | 214.18 / 260.00 |
| meanout | 65,221 [63,906, 71,703] | 207.18 / 240.00 |

Maxout has 345,408 FFN weights (70.72% fewer than full SwiGLU), 1,723,392 total
weights, and 688,128 projection FLOPs/token across four layers. Projection counts
exclude bias/selection work. Fused SwiGLU packs up/gate into one projection and
uses exactly the ordinary variant's initial weights, so comparison includes a
standard engineering optimization rather than a weakened dense implementation.

## Removal experiments

* tinystories/maxout: trained NLL 2.996539; frozen-weight removal 6.115512.
* tinystories/curve-only: trained NLL 3.063614; frozen-weight removal 3.063602.
* wikitext/maxout: trained NLL 4.684398; frozen-weight removal 6.518464.
* wikitext/curve-only: trained NLL 4.728093; frozen-weight removal 4.728086.


Maxout removal replaces max by mean with weights frozen. Meanout is also trained
independently with identical shapes/initialization. This changes nonlinearity
and statistics as well as selection, so it is not complete mechanistic attribution.
Curve removal tests whether its final trained weights need the curve at inference.

## Evidence and limitations

[Numerical/data checks](../../../records/language-v1-checks.json),
[profiles](../../../records/language-v1-profiles.json) and
[compact comparison](../../../records/language-v1-comparison.json).
Individual `records/language-v1-*.json` retain configurations and measurements;
`dump/language-v1/` retains protocol, generated configs and run mapping. Each run
stores frozen source, metrics and checkpoints. No across-seed uncertainty or
independent confirmation is available from this screen.

The three-branch activation is established [Maxout](https://arxiv.org/abs/1302.4389).
This study makes no novelty claim. Numerical checks cover the independent max
equation and gradients, causality, token locality, counts, fused SwiGLU equivalence,
dataset integrity and disjoint normalized document hashes. Near duplicates are
not guaranteed absent. The active research goal remains unfulfilled.
