# Compact FFN refinement: round 1 results

Three measured parents and three preregistered variations. Full last-checkpoint validation; seed101, 2000 updates/4,096,000 targets. No test loss in this development screen. Training times are observations, not isolated speed benchmarks. Subsequent confirmation, final selection and held-out testing are complete; see the [final summary](compact_refinement_summary.md).

| Corpus | Recipe | NLL | FFN weights | Allocated MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tinystories | signed-parent | 2.603935 | 2,490,368 | 414.3 | 71.64 | [record](../records/compact-refinement-v1-tinystories-signed-parent.json) |
| tinystories | signed-fine | 2.602645 | 2,359,296 | 410.0 | 80.31 | [record](../records/compact-refinement-v1-tinystories-signed-fine.json) |
| tinystories | curve-parent | 2.598287 | 2,121,728 | 401.2 | 78.80 | [record](../records/compact-refinement-v1-tinystories-curve-parent.json) |
| tinystories | curve-wide | 2.591014 | 2,121,728 | 401.2 | 81.35 | [record](../records/compact-refinement-v1-tinystories-curve-wide.json) |
| tinystories | basis-parent | 2.604161 | 2,490,368 | 405.5 | 90.30 | [record](../records/compact-refinement-v1-tinystories-basis-parent.json) |
| tinystories | basis-small-groups | 2.598418 | 2,293,760 | 403.2 | 90.72 | [record](../records/compact-refinement-v1-tinystories-basis-small-groups.json) |
| wikitext | signed-parent | 4.277294 | 2,490,368 | 415.9 | 82.67 | [record](../records/compact-refinement-v1-wikitext-signed-parent.json) |
| wikitext | signed-fine | 4.276089 | 2,359,296 | 410.0 | 68.37 | [record](../records/compact-refinement-v1-wikitext-signed-fine.json) |
| wikitext | curve-parent | 4.288112 | 2,121,728 | 401.2 | 85.97 | [record](../records/compact-refinement-v1-wikitext-curve-parent.json) |
| wikitext | curve-wide | 4.272187 | 2,121,728 | 401.2 | 67.95 | [record](../records/compact-refinement-v1-wikitext-curve-wide.json) |
| wikitext | basis-parent | 4.283474 | 2,490,368 | 405.5 | 86.77 | [record](../records/compact-refinement-v1-wikitext-basis-parent.json) |
| wikitext | basis-small-groups | 4.293402 | 2,293,760 | 403.2 | 90.94 | [record](../records/compact-refinement-v1-wikitext-basis-small-groups.json) |

tinystories, signed-fine: relative loss change versus parent -0.050% (negative is better).

tinystories, curve-wide: relative loss change versus parent -0.280% (negative is better).

tinystories, basis-small-groups: relative loss change versus parent -0.221% (negative is better).

wikitext, signed-fine: relative loss change versus parent -0.028% (negative is better).

wikitext, curve-wide: relative loss change versus parent -0.371% (negative is better).

wikitext, basis-small-groups: relative loss change versus parent +0.232% (negative is better).
