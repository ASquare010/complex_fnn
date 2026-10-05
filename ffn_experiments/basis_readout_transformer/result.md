# Expanded nonlinear basis-readout FFN: results

## Current bounded refinement

Smaller basis groups reduce FFN weights from 2,490,368 to 2,293,760 and improve
TinyStories NLL from 2.604161 to 2.598418, but worsen WikiText from 4.283474 to
4.293402 in seed101. This variation does not advance under the registered
two-corpus ranking. All trained models and frozen removals remain recorded;
this is a measured dataset tradeoff, not a universal rejection of smaller groups.
[Complete round results](../../docs/compact_refinement_result.md).

## Historical stricter-gate decision

Retire this fixed recipe: a required language or tied-readout comparison fails. Completed 18/18 runs.

[Hypothesis](../../docs/basis_readout_hypothesis.md) / [Hardware](../../docs/basis_readout_result.md) / [Conditional confirmation](../../docs/basis_readout_confirmation.md)

Full 512-by-512 P/Q matrices surround six scalar responses per coordinate: three shifted SiLU functions and their products with that coordinate. A learned readout mixes all responses within each 32-coordinate group before Q. Initial readout uses only local product responses; training can learn linear responses and different/grouped readout directions. Fixed templates are the hardware survivor; learned templates failed the slowdown limit and receive no language run. The design does not create input information or guarantee its retention. Attention, embeddings and all backbone settings are common.

Fixed/learned FFN weights 2,490,368/2,502,656, cuts 70.54%/70.40%. Tied control 2,109,440 weights, honestly smaller; it assembles a dense grouped matrix from actual local coefficients and uses the same physical operation. Both candidates use 4,980,736 forward projection FLOPs/token and 14,942,208 training projection FLOPs/token plus basis/reconstruction/reduction work. P/Q use BF16 AMP, basis and grouped GEMMs FP32, TF32 disabled. Recompute fixed candidate about 405.5 MiB in short screen; cached execution 607.5 MiB. Hardware survivor barely meets the slowdown limit; sustained confirmation is required.

Nine registered variants per corpus here: three full dense, compact kernel SwiGLU 408, GELU 608/611, fixed-basis, cached-basis and tied-basis. Compact GELU 608 exactly matches fixed weights; GELU 611 matches learned weights; SwiGLU has 0.658% more than fixed. Select strongest across all three compact references. Cached is the same mathematical fixed-basis function with ordinary autograd; no NLL superiority over identical cached execution is required. Candidate must beat tied on both corpora; its larger readout budget remains a possible explanation, which is why equally sized dense comparisons matter.

Width 512, four layers, heads 16, context 256, vocabulary 4096, batch eight, BF16, threads four, seed 101, 2000 updates / 4,096,000 targets, rate .0006, warmup 200, decay .1. Identical sampled windows and full eligible validation. Compare within each corpus because tokenizers differ. Test loss unopened. Required <=1% loss cost against strongest full and >=1% gain over strongest ordinary compact on both corpora, plus tied comparison. One-seed development cannot qualify a replacement.

| Corpus | Variant | Validation NLL | Allocated MiB | Update / wall seconds |
| --- | --- | ---: | ---: | ---: |
| tinystories | cached-basis | 2.610949 | 607.5 | 97.46 / 109.19 |
| tinystories | fixed-basis | 2.604161 | 405.5 | 93.11 / 104.72 |
| tinystories | full-gelu | 2.592983 | 539.4 | 88.36 / 101.25 |
| tinystories | full-swiglu-fused | 2.538486 | 560.7 | 88.90 / 101.68 |
| tinystories | full-swiglu-kernel | 2.544240 | 542.5 | 89.54 / 102.37 |
| tinystories | narrow-gelu | 2.616391 | 408.9 | 83.56 / 94.47 |
| tinystories | narrow-gelu-learned-budget | 2.629668 | 409.1 | 81.57 / 92.18 |
| tinystories | narrow-swiglu-kernel | 2.606365 | 407.6 | 85.24 / 96.05 |
| tinystories | tied-basis | 2.596579 | 402.6 | 99.99 / 111.70 |
| wikitext | cached-basis | 4.276874 | 607.5 | 97.54 / 109.29 |
| wikitext | fixed-basis | 4.283474 | 405.5 | 92.13 / 103.56 |
| wikitext | full-gelu | 4.269168 | 539.4 | 89.20 / 102.39 |
| wikitext | full-swiglu-fused | 4.237625 | 560.7 | 91.78 / 104.87 |
| wikitext | full-swiglu-kernel | 4.239026 | 540.5 | 90.58 / 103.70 |
| wikitext | narrow-gelu | 4.317237 | 408.9 | 83.97 / 94.85 |
| wikitext | narrow-gelu-learned-budget | 4.316168 | 409.1 | 81.68 / 92.52 |
| wikitext | narrow-swiglu-kernel | 4.288883 | 407.6 | 86.36 / 97.51 |
| wikitext | tied-basis | 4.298645 | 402.6 | 99.49 / 111.50 |

tinystories/fixed-basis: full loss cost 2.59%; compact gain 0.08%; beats tied False.

wikitext/fixed-basis: full loss cost 1.08%; compact gain 0.13%; beats tied True.

tinystories-cached-basis-removal: trained 2.610949; half features removed 3.563156; training-mean FFN output 8.278398. Initial readout 2.746157; product only 2.646040; linear only 6.673640; initial shapes 2.610949. Basis removals occur before B, retaining original calibration.

tinystories-fixed-basis-removal: trained 2.604161; half features removed 3.504561; training-mean FFN output 8.283653. Initial readout 2.746682; product only 2.642003; linear only 6.874086; initial shapes 2.604161. Basis removals occur before B, retaining original calibration.

tinystories-full-gelu-removal: trained 2.592983; half features removed 2.905189; training-mean FFN output 7.583977.

tinystories-full-swiglu-fused-removal: trained 2.538486; half features removed 3.030469; training-mean FFN output 9.228604.

tinystories-full-swiglu-kernel-removal: trained 2.544240; half features removed 3.015803; training-mean FFN output 9.090187.

tinystories-narrow-gelu-learned-budget-removal: trained 2.629668; half features removed 3.004531; training-mean FFN output 7.847036.

tinystories-narrow-gelu-removal: trained 2.616391; half features removed 2.996419; training-mean FFN output 9.082068.

tinystories-narrow-swiglu-kernel-removal: trained 2.606365; half features removed 3.289909; training-mean FFN output 7.945631.

tinystories-tied-basis-removal: trained 2.596579; half features removed 3.729862; training-mean FFN output 8.648421. Initial readout 2.599018; product only 2.596845; linear only 13.059984; initial shapes 2.596579. Basis removals occur before B, retaining original calibration.

wikitext-cached-basis-removal: trained 4.276874; half features removed 5.132156; training-mean FFN output 9.825506. Initial readout 4.629725; product only 4.351540; linear only 8.978522; initial shapes 4.276874. Basis removals occur before B, retaining original calibration.

wikitext-fixed-basis-removal: trained 4.283474; half features removed 5.100576; training-mean FFN output 9.866708. Initial readout 4.656733; product only 4.365289; linear only 9.067687; initial shapes 4.283474. Basis removals occur before B, retaining original calibration.

wikitext-full-gelu-removal: trained 4.269168; half features removed 4.625870; training-mean FFN output 10.499083.

wikitext-full-swiglu-fused-removal: trained 4.237625; half features removed 4.771781; training-mean FFN output 12.236745.

wikitext-full-swiglu-kernel-removal: trained 4.239026; half features removed 4.758692; training-mean FFN output 11.917761.

wikitext-narrow-gelu-learned-budget-removal: trained 4.316168; half features removed 4.744500; training-mean FFN output 10.470019.

wikitext-narrow-gelu-removal: trained 4.317237; half features removed 4.759945; training-mean FFN output 10.860102.

wikitext-narrow-swiglu-kernel-removal: trained 4.288883; half features removed 4.924743; training-mean FFN output 9.910267.

wikitext-tied-basis-removal: trained 4.298645; half features removed 5.349600; training-mean FFN output 9.726684. Initial readout 4.303431; product only 4.299969; linear only 11.337420; initial shapes 4.298645. Basis removals occur before B, retaining original calibration.

Frozen interventions change statistics and do not prove semantic roles. CPU FP64 independent equations/all gradients/finite differences, CUDA FP32/BF16 adjoints and rounding, complete-model causality/locality/save-load/CPU recovery passed. Production weights/counts/outputs/gradients match prototypes exactly in checks; shared-Trainer CPU recovery is exact. Preparation process failure without diagnostics and successful rerun are preserved.

Evidence: `records/basis-readout-v1-*.json`, `records/basis-readout-language-v1-comparison.json`, exported `records/basis-readout-v1-{corpus}-{variant}.json`; frozen protocols/sources under `dump/basis-readout-*`. Basis functions, grouped matrices and recomputation are established; originality and repeated-seed generalization remain unverified.
# Current refinement phase

Fixed basis readout is now a parent in the
[bounded refinement comparison](../../docs/compact_refinement_result.md),
with smaller readout groups as its first variation. The historical screen below
retains its original decision and thresholds.
