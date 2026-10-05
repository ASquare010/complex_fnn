# Full FFN weight-rank diagnostic: results

[Registered method](ffn_rank_diagnostic.md). CPU, one thread, FP64 matrix spectra.
Both completed full native SwiGLU controls receive the same 2,000-update budget.
Their archived dense model and initialization components match current code.

| Corpus | Layer | Phase | Input energy at 64 / 128 | Readout energy at 64 / 128 |
| --- | ---: | --- | ---: | ---: |
| tinystories | 1 | initial | 22.10% / 40.13% | 26.56% / 46.83% |
| tinystories | 1 | trained | 28.93% / 48.09% | 45.68% / 67.82% |
| tinystories | 2 | initial | 22.06% / 40.09% | 26.60% / 46.87% |
| tinystories | 2 | trained | 33.64% / 51.59% | 48.03% / 69.04% |
| tinystories | 3 | initial | 22.07% / 40.09% | 26.59% / 46.86% |
| tinystories | 3 | trained | 39.18% / 55.76% | 49.80% / 69.49% |
| tinystories | 4 | initial | 22.08% / 40.12% | 26.64% / 46.90% |
| tinystories | 4 | trained | 47.39% / 62.83% | 55.24% / 74.19% |
| wikitext | 1 | initial | 22.10% / 40.13% | 26.56% / 46.83% |
| wikitext | 1 | trained | 25.46% / 44.34% | 37.66% / 61.11% |
| wikitext | 2 | initial | 22.06% / 40.09% | 26.60% / 46.87% |
| wikitext | 2 | trained | 31.95% / 51.61% | 40.04% / 62.97% |
| wikitext | 3 | initial | 22.07% / 40.09% | 26.59% / 46.86% |
| wikitext | 3 | trained | 33.29% / 51.56% | 41.96% / 63.37% |
| wikitext | 4 | initial | 22.08% / 40.12% | 26.64% / 46.90% |
| wikitext | 4 | trained | 41.61% / 58.99% | 48.17% / 68.43% |

Energy is the retained fraction of squared singular values, not retained semantic
information. Optimal Frobenius errors and ranks 16/32/256 are in the JSON record.
These are trained weight matrices, not activation covariances or compression-loss
measurements. No rank, architecture or quality benefit is selected/established
from these spectra. Do not reuse trained weights without counting their budget.
Evidence: `records/ffn-rank-v1-diagnostic.json`. No new validation/test scoring or
GPU computation was performed; the signed-context campaign remains unchanged.
