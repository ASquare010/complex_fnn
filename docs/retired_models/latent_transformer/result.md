# Latent FFN: results



Retire this fixed wide recipe: it misses a required language or allocation comparison. Completed 18/18 runs.



[Hypothesis](../../latent_ffn_hypothesis.md) Â· [Hardware screen](../../latent_ffn_result.md)



FFN(x) = D B (SiLU(A_feature P x) * A_gate P x). Input/output rank 128, hidden 1280. All matrices trained from scratch. Attention and embeddings remain the common backbone. Thin controls use hidden 512/input rank 320/output rank 128, with identical weight/projection budgets; one predeclared initialization control scales D by sqrt(2.5).



2,490,368 FFN weights: 70.54% fewer than full SwiGLU. Forward projection FLOPs 4,980,736/token; recomputation adds 2,621,440/token backward. Total training projection work 17,563,648/token, excluding other costs. Short hardware screen passed all full controls on both corpora at about 396 MiB allocated. Sustained resource benefit remains unconfirmed.



Nine variants per corpus; width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight, BF16, seed 101, 2000 updates / 4,096,000 targets, rate .0006, warmup 200, decay .1. Identical windows and full eligible validation; test loss unopened. Select strongest full and ordinary compact per corpus, require within 1% full NLL and >=1% compact improvement on both, plus beating both thin allocation controls. Cached wide is an execution control; no required NLL superiority over it.



| Corpus | Variant | Validation NLL | Allocated MiB | Updates / wall seconds |

| --- | --- | ---: | ---: | ---: |

| tinystories | full-gelu | 2.592983 | 539.4 | 87.82 / 100.72 |

| tinystories | full-swiglu-fused | 2.538486 | 560.7 | 88.10 / 101.17 |

| tinystories | full-swiglu-kernel | 2.544240 | 542.5 | 91.52 / 104.83 |

| tinystories | narrow-gelu | 2.616391 | 408.9 | 83.60 / 94.44 |

| tinystories | narrow-swiglu-kernel | 2.606365 | 407.6 | 86.13 / 97.30 |

| tinystories | thin-calibrated | 2.701581 | 396.0 | 105.96 / 119.87 |

| tinystories | thin-recompute | 2.725223 | 396.0 | 93.23 / 104.40 |

| tinystories | wide-cached | 2.726152 | 458.8 | 91.27 / 102.49 |

| tinystories | wide-recompute | 2.726152 | 396.0 | 90.68 / 101.83 |

| wikitext | full-gelu | 4.269168 | 539.4 | 88.63 / 102.21 |

| wikitext | full-swiglu-fused | 4.237625 | 560.7 | 91.40 / 104.54 |

| wikitext | full-swiglu-kernel | 4.239026 | 540.5 | 109.15 / 124.38 |

| wikitext | narrow-gelu | 4.317237 | 408.9 | 82.73 / 93.38 |

| wikitext | narrow-swiglu-kernel | 4.288883 | 407.6 | 88.63 / 99.93 |

| wikitext | thin-calibrated | 4.406500 | 396.0 | 91.06 / 102.22 |

| wikitext | thin-recompute | 4.438517 | 396.0 | 90.82 / 102.06 |

| wikitext | wide-cached | 4.400809 | 458.8 | 90.96 / 102.31 |

| wikitext | wide-recompute | 4.400809 | 396.0 | 91.72 / 103.09 |



tinystories: full loss cost 7.39%; compact gain -4.60%; beats both thin controls: False.



wikitext: full loss cost 3.85%; compact gain -2.61%; beats both thin controls: True.



tinystories-full-gelu-removal: trained 2.592983; half detector features 2.905189; training-mean FFN output 7.583977.



tinystories-full-swiglu-fused-removal: trained 2.538486; half detector features 3.030469; training-mean FFN output 9.228604.



tinystories-full-swiglu-kernel-removal: trained 2.544240; half detector features 3.015803; training-mean FFN output 9.090187.



tinystories-narrow-gelu-removal: trained 2.616391; half detector features 2.996419; training-mean FFN output 9.082068.



tinystories-narrow-swiglu-kernel-removal: trained 2.606365; half detector features 3.289909; training-mean FFN output 7.945631.



tinystories-thin-calibrated-removal: trained 2.701581; half detector features 3.259990; training-mean FFN output 7.946677. Unit expanded gates 8.334068.



tinystories-thin-recompute-removal: trained 2.725223; half detector features 3.269762; training-mean FFN output 8.343011. Unit expanded gates 8.872181.



tinystories-wide-cached-removal: trained 2.726152; half detector features 3.153861; training-mean FFN output 7.796377. Unit expanded gates 8.857284.



tinystories-wide-recompute-removal: trained 2.726152; half detector features 3.153861; training-mean FFN output 7.796377. Unit expanded gates 8.857284.



wikitext-full-gelu-removal: trained 4.269168; half detector features 4.625870; training-mean FFN output 10.499083.



wikitext-full-swiglu-fused-removal: trained 4.237625; half detector features 4.771781; training-mean FFN output 12.236745.



wikitext-full-swiglu-kernel-removal: trained 4.239026; half detector features 4.758692; training-mean FFN output 11.917761.



wikitext-narrow-gelu-removal: trained 4.317237; half detector features 4.759945; training-mean FFN output 10.860102.



wikitext-narrow-swiglu-kernel-removal: trained 4.288883; half detector features 4.924743; training-mean FFN output 9.910267.



wikitext-thin-calibrated-removal: trained 4.406500; half detector features 4.908704; training-mean FFN output 10.283481. Unit expanded gates 8.298545.



wikitext-thin-recompute-removal: trained 4.438517; half detector features 4.914194; training-mean FFN output 9.782496. Unit expanded gates 8.585577.



wikitext-wide-cached-removal: trained 4.400809; half detector features 4.848904; training-mean FFN output 10.529606. Unit expanded gates 9.343072.



wikitext-wide-recompute-removal: trained 4.400809; half detector features 4.848904; training-mean FFN output 10.529606. Unit expanded gates 9.343072.



Frozen-weight removals change statistics and do not by themselves prove semantic roles or improved training. Detector removal occurs before B, not in the rank-128 readout. Production CPU/CUDA weights/outputs/gradients match the staged implementation exactly; shared-Trainer CPU resume is exact. Independent inner equation/gradient checks use recorded tolerances.



Evidence: `records/latent-v1-*.json`, `records/latent-language-v1-comparison.json`, per-run `records/latent-v1-{corpus}-{variant}.json` and frozen sources/protocols under `dump/latent-*`. Factorization, GLUs and recomputation are established; originality and generalization are unverified.
