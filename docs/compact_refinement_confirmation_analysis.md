# Paired-seed language comparison

All 60 runs audited: full last-checkpoint validation, identical budgets and frozen recipes.
Means cover seeds 211, 307 and 401. Lower loss is better. Three seeds do not establish statistical significance.
Resource measurements are complete and the overall choice is frozen; independent
longer-budget evaluation and held-out testing are now complete. See the
[final summary](compact_refinement_summary.md), [selection](compact_refinement_final_selection.md)
and [resource timing limitations](compact_refinement_resource_analysis.md).

| Recipe | FFN weights | TinyStories mean NLL | WikiText mean NLL |
| --- | ---: | ---: | ---: |
| curve-wide | 2,121,728 | 2.592300 | 4.272862 |
| signed-fine | 2,359,296 | 2.607223 | 4.271804 |
| curve-parent | 2,121,728 | 2.605145 | 4.282949 |
| compact-swiglu-kernel-h346 | 2,125,824 | 2.614868 | 4.287453 |
| compact-gelu-h518 | 2,121,728 | 2.625608 | 4.314461 |
| signed-parent | 2,490,368 | 2.600803 | 4.271632 |
| compact-swiglu-kernel-h384 | 2,359,296 | 2.607348 | 4.282197 |
| compact-gelu-h576 | 2,359,296 | 2.617236 | 4.301414 |
| full-swiglu | 8,454,144 | 2.542837 | 4.220415 |
| full-gelu | 8,454,144 | 2.589529 | 4.254606 |

| Variation | Corpus | Mean NLL change vs parent | Seeds improved |
| --- | --- | ---: | ---: |
| curve-wide | tinystories | -0.012845 | 3/3 |
| curve-wide | wikitext | -0.010087 | 3/3 |
| signed-fine | tinystories | +0.006420 | 0/3 |
| signed-fine | wikitext | +0.000172 | 1/3 |

[Every seed and evidence](compact_refinement_confirmation_result.md).
[Audit evidence](../records/compact-confirmation-v1-language-audit.json).
