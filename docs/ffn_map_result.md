# Trained full FFN activation-map diagnosis

Completed CPU-only diagnostic. No language gain or replacement qualification claim.

[Registered protocol](ffn_map_diagnostic.md). Full native SwiGLU, seed 101, 2000 updates; both corpora, four layers, two reversed training halves, two fit sample sizes. No validation/test activations. Frozen model CPU FP32 differs from BF16 training; regression/SVD FP64.

| Corpus | Fit seed | Layer | Fit pairs | Effective input rank | Affine held-out R2 | Rank-128 held-out R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tinystories | 6001 | 0 | 4096 | 512 | 0.5497 | 0.5380 |
| tinystories | 6001 | 0 | 8192 | 512 | 0.6096 | 0.5908 |
| tinystories | 6001 | 1 | 4096 | 512 | 0.5191 | 0.5094 |
| tinystories | 6001 | 1 | 8192 | 512 | 0.5763 | 0.5620 |
| tinystories | 6001 | 2 | 4096 | 512 | 0.4087 | 0.4045 |
| tinystories | 6001 | 2 | 8192 | 512 | 0.4710 | 0.4614 |
| tinystories | 6001 | 3 | 4096 | 512 | 0.3881 | 0.3867 |
| tinystories | 6001 | 3 | 8192 | 512 | 0.4485 | 0.4423 |
| tinystories | 6007 | 0 | 4096 | 512 | 0.5535 | 0.5421 |
| tinystories | 6007 | 0 | 8192 | 512 | 0.6175 | 0.5972 |
| tinystories | 6007 | 1 | 4096 | 512 | 0.5222 | 0.5128 |
| tinystories | 6007 | 1 | 8192 | 512 | 0.5847 | 0.5704 |
| tinystories | 6007 | 2 | 4096 | 512 | 0.4019 | 0.3980 |
| tinystories | 6007 | 2 | 8192 | 512 | 0.4683 | 0.4585 |
| tinystories | 6007 | 3 | 4096 | 512 | 0.3894 | 0.3880 |
| tinystories | 6007 | 3 | 8192 | 512 | 0.4568 | 0.4502 |
| wikitext | 6001 | 0 | 4096 | 512 | 0.4469 | 0.4319 |
| wikitext | 6001 | 0 | 8192 | 512 | 0.5220 | 0.4954 |
| wikitext | 6001 | 1 | 4096 | 512 | 0.3145 | 0.3073 |
| wikitext | 6001 | 1 | 8192 | 512 | 0.4066 | 0.3864 |
| wikitext | 6001 | 2 | 4096 | 512 | 0.3537 | 0.3466 |
| wikitext | 6001 | 2 | 8192 | 512 | 0.4293 | 0.4119 |
| wikitext | 6001 | 3 | 4096 | 512 | 0.2396 | 0.2432 |
| wikitext | 6001 | 3 | 8192 | 512 | 0.3249 | 0.3206 |
| wikitext | 6007 | 0 | 4096 | 512 | 0.4514 | 0.4372 |
| wikitext | 6007 | 0 | 8192 | 512 | 0.5259 | 0.4984 |
| wikitext | 6007 | 1 | 4096 | 512 | 0.3317 | 0.3267 |
| wikitext | 6007 | 1 | 8192 | 512 | 0.4205 | 0.4018 |
| wikitext | 6007 | 2 | 4096 | 512 | 0.3500 | 0.3441 |
| wikitext | 6007 | 2 | 8192 | 512 | 0.4306 | 0.4129 |
| wikitext | 6007 | 3 | 4096 | 512 | 0.2398 | 0.2440 |
| wikitext | 6007 | 3 | 8192 | 512 | 0.3206 | 0.3164 |

The table shows the registered zero-ridge fits; all four ridge sensitivities, output ranks, per-output scores/variation, sample sizes, partitions and position hashes are retained in the diagnostic JSON. High fit is not evidence of preserved predictions or permission to discard input directions. No architecture was tuned with this held-out activation measurement. Synthetic affine/rank-128/quadratic checks passed before research checkpoints were read.

Evidence: `records/ffn-map-v1-diagnostic.json`; registered protocol and source/data/checkpoint hashes retained.

Compact export retains every aggregate comparison plus worst/highest-variation output coordinates. Full per-output arrays are in `dump/ffn-map-v1/diagnostic.json`, whose hash is retained in the compact record. Source and registration are frozen under `dump/ffn-map-v1/source/`.
