# Nonlinear feature-flow FFN: results

Retire this fixed recipe: a required quality or mechanism comparison fails. Completed 22/22 runs.

[Hypothesis](../../feature_flow_hypothesis.md) / [Resources](../../feature_flow_v4_result.md) / [Conditional confirmation](../../feature_flow_confirmation.md)

Full-width P/Q run once. Three small grouped updates mix bounded nonlinear responses, then Q reads the final nonlinear reaction. M is learned and shared across those three internal steps within each layer, independent across layers. No token state or attention/embedding change. All mechanism controls start at the same function with M zero. Group locality has no established semantic meaning.

2,162,688 FFN weights, 74.42% fewer than full SwiGLU; forward projection work 4,587,520 FLOPs/token and training projection work including reconstruction 14,155,776 plus scalar/reduction work. Revision 4 reconstructs reactions/derivatives in lane registers and reuses them during exact discrete backpropagation. BF16 P/Q, FP32 internal operations, native final casts and TF32 disabled. Short resource screen: 401.7 MiB allocated, 25.54–28.36% reduction and throughput ratios .984–1.029 against full dense; sustained resources remain unconfirmed. Timings varied between rounds; all are preserved.

Eleven variants per corpus: three full dense, exactly sized compact kernel SwiGLU 352 and GELU 528, variance-calibrated compact kernel SwiGLU 352, primary, one-step, actual diagonal, no-update and ordinary cached controls. Calibrated SwiGLU multiplies down initialization by sqrt(1376/352), keeping every other weight/count/computation unchanged. Strongest ordinary compact is selected across all three. Primary must beat one-step and diagonal on both corpora. Cached is the same mathematical function and has no required NLL superiority comparison.

Width 512, four layers, heads 16, context 256, vocab 4096, batch eight, BF16, threads four, seed 101, 2000 updates / 4,096,000 targets, learning rate .0006, warmup 200 and weight decay .1. Same sampled windows and full eligible validation. Test loss unopened. Required <=1% cost against strongest full and >=1% gain over strongest ordinary compact on both corpora. One development seed cannot qualify a replacement.

| Corpus | Variant | Validation NLL | Allocated MiB | Update / wall seconds |
| --- | --- | ---: | ---: | ---: |
| tinystories | feature-cached | 2.606072 | 617.7 | 121.46 / 134.37 |
| tinystories | feature-diagonal | 2.604415 | 403.2 | 91.88 / 103.24 |
| tinystories | feature-flow | 2.606496 | 403.7 | 89.16 / 100.31 |
| tinystories | feature-once | 2.607807 | 403.7 | 88.35 / 99.44 |
| tinystories | feature-plain | 2.604789 | 403.2 | 85.50 / 96.50 |
| tinystories | full-gelu | 2.592983 | 539.4 | 88.78 / 104.34 |
| tinystories | full-swiglu-fused | 2.538486 | 560.7 | 89.97 / 103.23 |
| tinystories | full-swiglu-kernel | 2.544240 | 542.5 | 94.45 / 108.08 |
| tinystories | narrow-gelu | 2.622865 | 404.4 | 81.10 / 91.70 |
| tinystories | narrow-swiglu-calibrated | 2.614896 | 406.2 | 84.55 / 95.47 |
| tinystories | narrow-swiglu-kernel | 2.622897 | 406.2 | 85.36 / 96.07 |
| wikitext | feature-cached | 4.291654 | 617.7 | 122.03 / 135.45 |
| wikitext | feature-diagonal | 4.293004 | 403.2 | 90.08 / 101.54 |
| wikitext | feature-flow | 4.300017 | 403.7 | 87.65 / 98.54 |
| wikitext | feature-once | 4.296649 | 403.7 | 89.74 / 100.66 |
| wikitext | feature-plain | 4.296419 | 403.2 | 88.24 / 99.47 |
| wikitext | full-gelu | 4.269168 | 539.4 | 89.12 / 102.47 |
| wikitext | full-swiglu-fused | 4.237625 | 560.7 | 92.89 / 106.54 |
| wikitext | full-swiglu-kernel | 4.239026 | 540.5 | 89.56 / 102.66 |
| wikitext | narrow-gelu | 4.329683 | 404.4 | 81.19 / 91.92 |
| wikitext | narrow-swiglu-calibrated | 4.309997 | 406.2 | 86.09 / 96.94 |
| wikitext | narrow-swiglu-kernel | 4.300213 | 406.2 | 84.58 / 95.34 |

tinystories: full loss cost 2.68%; compact gain 0.32%; beats one-step/diagonal False.

wikitext: full loss cost 1.47%; compact gain 0.00%; beats one-step/diagonal False.

tinystories-feature-cached-removal: trained 2.606072; half responses removed 2.606949; training-mean FFN 8.423438. Zero M 2.608531; diagonal M 2.607941; single step 2.606086; original calibration retained.

tinystories-feature-diagonal-removal: trained 2.604415; half responses removed 2.604420; training-mean FFN 8.729918. Zero M 2.604500; diagonal M 2.604415; single step 2.604435; original calibration retained.

tinystories-feature-flow-removal: trained 2.606496; half responses removed 2.607348; training-mean FFN 8.603021. Zero M 2.609012; diagonal M 2.608527; single step 2.606477; original calibration retained.

tinystories-feature-once-removal: trained 2.607807; half responses removed 2.608654; training-mean FFN 8.909471. Zero M 2.610399; diagonal M 2.609838; single step 2.607807; original calibration retained.

tinystories-feature-plain-removal: trained 2.604789; half responses removed 2.604789; training-mean FFN 8.668289. Zero M 2.604789; diagonal M 2.604789; single step 2.604789; original calibration retained.

tinystories-full-gelu-removal: trained 2.592983; half responses removed 2.905189; training-mean FFN 7.583977.

tinystories-full-swiglu-fused-removal: trained 2.538486; half responses removed 3.030469; training-mean FFN 9.228604.

tinystories-full-swiglu-kernel-removal: trained 2.544240; half responses removed 3.015803; training-mean FFN 9.090187.

tinystories-narrow-gelu-removal: trained 2.622865; half responses removed 3.007012; training-mean FFN 8.029525.

tinystories-narrow-swiglu-calibrated-removal: trained 2.614896; half responses removed 3.401988; training-mean FFN 8.349486.

tinystories-narrow-swiglu-kernel-removal: trained 2.622897; half responses removed 3.332685; training-mean FFN 8.568975.

wikitext-feature-cached-removal: trained 4.291654; half responses removed 4.292872; training-mean FFN 9.203911. Zero M 4.294403; diagonal M 4.293416; single step 4.291655; original calibration retained.

wikitext-feature-diagonal-removal: trained 4.293004; half responses removed 4.293142; training-mean FFN 9.440181. Zero M 4.293214; diagonal M 4.293004; single step 4.293013; original calibration retained.

wikitext-feature-flow-removal: trained 4.300017; half responses removed 4.301398; training-mean FFN 9.804952. Zero M 4.303129; diagonal M 4.301878; single step 4.300059; original calibration retained.

wikitext-feature-once-removal: trained 4.296649; half responses removed 4.298104; training-mean FFN 9.787180. Zero M 4.300022; diagonal M 4.298794; single step 4.296649; original calibration retained.

wikitext-feature-plain-removal: trained 4.296419; half responses removed 4.296419; training-mean FFN 9.176048. Zero M 4.296419; diagonal M 4.296419; single step 4.296419; original calibration retained.

wikitext-full-gelu-removal: trained 4.269168; half responses removed 4.625870; training-mean FFN 10.499083.

wikitext-full-swiglu-fused-removal: trained 4.237625; half responses removed 4.771781; training-mean FFN 12.236745.

wikitext-full-swiglu-kernel-removal: trained 4.239026; half responses removed 4.758692; training-mean FFN 11.917761.

wikitext-narrow-gelu-removal: trained 4.329683; half responses removed 4.772864; training-mean FFN 11.082306.

wikitext-narrow-swiglu-calibrated-removal: trained 4.309997; half responses removed 5.049349; training-mean FFN 10.373492.

wikitext-narrow-swiglu-kernel-removal: trained 4.300213; half responses removed 4.986005; training-mean FFN 9.499984.

Frozen removals change statistics and do not prove semantic roles. CPU FP64 independent equations/adjoints/finite differences, CUDA full-batch gradients and cast edgecases passed. Production initialized weights/counts/outputs/gradients match staged prototypes exactly, including the actual full-width shape. Shared-Trainer CPU recovery is exact for all controls. Recurrence, scalar gates and grouped matrices have prior art; originality remains unverified.

Evidence: `records/feature-flow-v*-*.json`, `records/feature-flow-language-v1-comparison.json`, exported per-run records and frozen sources/protocols under `dump/feature-flow-*`.
