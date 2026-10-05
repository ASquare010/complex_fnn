# Coupled channel curves: results

## Current bounded refinement

Self-Curve's wider-initialization variation (`curve-wide`) advanced to paired-seed
confirmation and is now the frozen overall research choice. Seed101 full validation improves from 2.598287 to 2.591014 on
TinyStories and from 4.288112 to 4.272187 on WikiText, with unchanged 2,121,728 FFN
weights. It ranks first in the six-recipe development comparison. Restoring its
own initial curves gives WikiText 4.272974; this intervention alone does not
identify the cause of the improvement. All removal evidence is retained in
`dump/compact-refinement-v1/`. [Round results](../../docs/compact_refinement_result.md)
and [confirmation protocol](../../docs/compact_refinement_confirmation.md).

Confirmation seeds211/307/401 improve over the parent in all six paired runs.
Mean TinyStories NLL is 2.592300 versus 2.605145; WikiText is 4.272862 versus
4.282949. The observed resource medians meet the parent-relative limits, but
variable full-baseline timings do not establish a speedup. The independent
seed509/8000-update comparison and held-out tests are complete and audited.
Curve-Wide beats its parent and compact SwiGLU h384 on validation and test in both
corpora. Held-out NLL is 2.020291 on TinyStories and 3.966762 on WikiText, versus
full SwiGLU 1.974934 and 4.022206. It is the final selected compact recipe; the
full model retains the TinyStories quality advantage. See the
[final summary](../../docs/compact_refinement_summary.md).
[Frozen choice and limitations](../../docs/compact_refinement_final_selection.md).

## Historical stricter-gate decision

Retire this fixed recipe: a required language or mechanism comparison fails. Completed 18/18 runs.

[Hypothesis](../../docs/channel_curve_hypothesis.md) / [Fused execution](../../docs/channel_curve_v2_hypothesis.md) / [Hardware result](../../docs/channel_curve_v2_result.md)

Two full 512-by-512 mixing matrices surround three learned activation/gate branches per coordinate. Fixed feature shifts [1,17,129] mix features within a token. Each branch has learned slope, offset, gate gain and offset; branch outputs share a readout direction. No imposed rank bottleneck, but no information-retention or expressivity guarantee. Attention, embeddings and the residual backbone remain common.

2,121,728 FFN weights, 74.90% fewer than full SwiGLU. Forward projection work 4,194,304 FLOPs/token; total training projection work 12,582,912 plus scalar activation/reconstruction/reduction work. Fused input backward reconstructs additional nonlinearities. Allocated short-screen memory about 401 MiB; version 2 passed against every full control on both corpora. Sustained resource benefit remains unconfirmed. Windows bundled NVRTC/CUDA Driver API backend tested on RTX 4070 Laptop.

Nine variants per corpus: three full dense, compact kernel SwiGLU 346 and GELU 518, candidate, self-gate curves, fixed activation shapes, plain SiLU 512. GELU exactly matches candidate weights; compact SwiGLU has 0.193% more. Compare strongest ordinary compact including plain. Self-gate/fixed-shape are mechanism controls; candidate must beat both. Controls retain the registered separate initial mean/variance calibration, which does not match all distributions or logits.

Width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight, BF16, seed 101, 2000 updates / 4,096,000 targets, rate .0006, warmup 200, decay .1. Identical sampled windows and full eligible validation; tokenizers differ by corpus, so compare NLL within each. Test loss unopened. One common architecture must pass <=1% loss cost against strongest full and >=1% gain over strongest ordinary compact on both corpora, plus mechanism controls.

| Corpus | Variant | Validation NLL | Allocated MiB | Update / wall seconds |
| --- | --- | ---: | ---: | ---: |
| tinystories | channel-curve | 2.626786 | 401.2 | 83.65 / 94.27 |
| tinystories | fixed-shape | 2.628363 | 401.1 | 84.12 / 94.80 |
| tinystories | full-gelu | 2.592983 | 539.4 | 89.71 / 102.94 |
| tinystories | full-swiglu-fused | 2.538486 | 560.7 | 88.49 / 101.29 |
| tinystories | full-swiglu-kernel | 2.544240 | 542.5 | 86.36 / 98.96 |
| tinystories | narrow-gelu | 2.622748 | 404.1 | 79.98 / 90.35 |
| tinystories | narrow-swiglu-kernel | 2.613739 | 405.8 | 83.84 / 94.28 |
| tinystories | plain-silu | 2.673912 | 400.9 | 79.47 / 89.86 |
| tinystories | self-curve | 2.598287 | 401.2 | 87.13 / 97.90 |
| wikitext | channel-curve | 4.298236 | 401.2 | 83.49 / 94.04 |
| wikitext | fixed-shape | 4.304249 | 401.1 | 86.10 / 96.87 |
| wikitext | full-gelu | 4.269168 | 539.4 | 88.66 / 101.49 |
| wikitext | full-swiglu-fused | 4.237625 | 560.7 | 89.12 / 101.99 |
| wikitext | full-swiglu-kernel | 4.239026 | 540.5 | 88.44 / 101.40 |
| wikitext | narrow-gelu | 4.321492 | 404.1 | 81.67 / 92.22 |
| wikitext | narrow-swiglu-kernel | 4.308013 | 405.8 | 85.07 / 95.86 |
| wikitext | plain-silu | 4.369832 | 400.9 | 80.96 / 91.51 |
| wikitext | self-curve | 4.288112 | 401.2 | 89.17 / 100.09 |

tinystories: full loss cost 3.48%; compact gain -0.50%; beats self/fixed-shape False.

wikitext: full loss cost 1.43%; compact gain 0.23%; beats self/fixed-shape False.

tinystories-channel-curve-removal: trained 2.626786; even branch-summed/hidden coordinates removed 3.141276; training-mean FFN output 8.161526. Self gates 7.045825; restored initial activation shapes 2.628235; original calibration retained.

tinystories-fixed-shape-removal: trained 2.628363; even branch-summed/hidden coordinates removed 3.163655; training-mean FFN output 8.584618. Self gates 6.797146; restored initial activation shapes 2.628363; original calibration retained.

tinystories-full-gelu-removal: trained 2.592983; even branch-summed/hidden coordinates removed 2.905189; training-mean FFN output 7.583977.

tinystories-full-swiglu-fused-removal: trained 2.538486; even branch-summed/hidden coordinates removed 3.030469; training-mean FFN output 9.228604.

tinystories-full-swiglu-kernel-removal: trained 2.544240; even branch-summed/hidden coordinates removed 3.015803; training-mean FFN output 9.090187.

tinystories-narrow-gelu-removal: trained 2.622748; even branch-summed/hidden coordinates removed 3.077481; training-mean FFN output 8.655324.

tinystories-narrow-swiglu-kernel-removal: trained 2.613739; even branch-summed/hidden coordinates removed 3.382612; training-mean FFN output 8.536375.

tinystories-plain-silu-removal: trained 2.673912; even branch-summed/hidden coordinates removed 3.035531; training-mean FFN output 8.151718.

tinystories-self-curve-removal: trained 2.598287; even branch-summed/hidden coordinates removed 3.435816; training-mean FFN output 8.247643. Self gates 2.598287; restored initial activation shapes 2.599811; original calibration retained.

wikitext-channel-curve-removal: trained 4.298236; even branch-summed/hidden coordinates removed 4.847349; training-mean FFN output 10.406254. Self gates 7.883546; restored initial activation shapes 4.299586; original calibration retained.

wikitext-fixed-shape-removal: trained 4.304249; even branch-summed/hidden coordinates removed 4.843604; training-mean FFN output 10.754345. Self gates 7.920284; restored initial activation shapes 4.304249; original calibration retained.

wikitext-full-gelu-removal: trained 4.269168; even branch-summed/hidden coordinates removed 4.625870; training-mean FFN output 10.499083.

wikitext-full-swiglu-fused-removal: trained 4.237625; even branch-summed/hidden coordinates removed 4.771781; training-mean FFN output 12.236745.

wikitext-full-swiglu-kernel-removal: trained 4.239026; even branch-summed/hidden coordinates removed 4.758692; training-mean FFN output 11.917761.

wikitext-narrow-gelu-removal: trained 4.321492; even branch-summed/hidden coordinates removed 4.713471; training-mean FFN output 10.017462.

wikitext-narrow-swiglu-kernel-removal: trained 4.308013; even branch-summed/hidden coordinates removed 5.005625; training-mean FFN output 9.796193.

wikitext-plain-silu-removal: trained 4.369832; even branch-summed/hidden coordinates removed 4.739931; training-mean FFN output 9.129613.

wikitext-self-curve-removal: trained 4.288112; even branch-summed/hidden coordinates removed 5.007891; training-mean FFN output 9.488027. Self gates 4.288112; restored initial activation shapes 4.289028; original calibration retained.

Frozen-weight interventions change statistics and do not by themselves prove semantic roles or improved training. Retrained controls are necessary. CPU FP64 independent equations/gradients/finite differences, CUDA references including incomplete tiles and BF16 rounding passed. Production CPU/CUDA outputs/gradients match prototypes exactly in recorded checks; shared-Trainer CPU recovery is exact. All error magnitudes/tolerances are recorded.

Evidence: `records/channel-curve-v2-*.json`, `records/channel-curve-language-v2-comparison.json`, exported `records/channel-curve-v2-{corpus}-{variant}.json`; frozen sources/protocols under `dump/channel-curve-*`. Learned activations/gating/recomputation are prior art; originality and repeated-seed generalization remain unverified.
# Current refinement phase

The strong self-gated control is now a parent in the
[bounded refinement comparison](../../docs/compact_refinement_result.md).
Its first variation broadens initial activation shapes without changing kernel
code. The historical coupled-model decision below is preserved.
