# Dense baseline and controls: results

## Current compact FFN confirmation

The language comparison at width512/four layers completed all paired seeds
211/307/401, with 2000 updates and full last-checkpoint validation on both corpora.
These are language NLL results; the historical synthetic task screen below is a
separate comparison and does not enter the language ranking.

| Dense recipe | FFN weights | TinyStories mean NLL | WikiText mean NLL |
| --- | ---: | ---: | ---: |
| Full native SwiGLU h1376 | 8,454,144 | 2.542837 | 4.220415 |
| Full GELU h2064 | 8,454,144 | 2.589529 | 4.254606 |
| Compact SwiGLU h384 | 2,359,296 | 2.607348 | 4.282197 |
| Compact SwiGLU h346 | 2,125,824 | 2.614868 | 4.287453 |
| Compact GELU h576 | 2,359,296 | 2.617236 | 4.301414 |
| Compact GELU h518 | 2,121,728 | 2.625608 | 4.314461 |

Full SwiGLU has the best mean language loss in this 2000-update comparison.
Compact SwiGLU h384 is the strongest
ordinary compact control in the registered combined comparison. Curve-Wide has
lower mean loss than every compact dense control on both corpora, with fewer FFN
weights than h384. The independent seed509/8000-update evaluation is complete:
full SwiGLU test NLL is 1.974934/4.022206 on TinyStories/WikiText, compact h384
is 2.024020/3.971140 and Curve-Wide is 2.020291/3.966762. Full SwiGLU retains
the TinyStories lead but trails both compact recipes on WikiText at this longer
single-seed budget. Variable resource timings do not support a reliable speedup claim.

[Every confirmation run](../../../docs/compact_refinement_confirmation_result.md) /
[paired analysis](../../../docs/compact_refinement_confirmation_analysis.md) /
[independent evaluation](../../../docs/compact_refinement_independent_result.md) /
[resource limitations](../../../docs/compact_refinement_resource_analysis.md).
