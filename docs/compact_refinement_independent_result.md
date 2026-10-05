# Independent compact FFN evaluation

Frozen research choice: **curve-wide**. Seed509, 8000 updates per run.
The choice predates these results. Test scores do not select the recipe.

Completed and [audited](../records/compact-refinement-final-audit.json): all eight
full validation and held-out test evaluations. Curve-Wide beats its parent and
compact SwiGLU on both splits and corpora. Full SwiGLU wins TinyStories;
Curve-Wide wins WikiText at this single-seed longer budget.

**Timing note:** WikiText Curve-Wide's raw 36970.15 seconds includes verified
Windows standby/hibernation and is unsuitable for speed comparisons.
[Event evidence](../records/compact-final-v1-timing-review.json). No corrected
compute time is inferred. Training MiB means peak allocated GPU memory.

| Corpus | Recipe | Validation NLL | Test NLL | FFN weights | Training MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| tinystories | curve-wide | 2.122985 | 2.020291 | 2,121,728 | 401.2 | 375.17 | [record](../records/compact-final-v1-tinystories-curve-wide-final.json) |
| tinystories | curve-parent | 2.125866 | 2.020990 | 2,121,728 | 403.2 | 374.00 | [record](../records/compact-final-v1-tinystories-curve-parent-final.json) |
| tinystories | compact-swiglu-kernel-h384 | 2.127756 | 2.024020 | 2,359,296 | 407.5 | 365.92 | [record](../records/compact-final-v1-tinystories-compact-swiglu-kernel-h384-final.json) |
| tinystories | full-swiglu | 2.078244 | 1.974934 | 8,454,144 | 560.7 | 394.55 | [record](../records/compact-final-v1-tinystories-full-swiglu-final.json) |
| wikitext | curve-wide | 3.942322 | 3.966762 | 2,121,728 | 403.2 | 36970.15 | [record](../records/compact-final-v1-wikitext-curve-wide-final.json) |
| wikitext | curve-parent | 3.962206 | 3.986061 | 2,121,728 | 403.2 | 359.32 | [record](../records/compact-final-v1-wikitext-curve-parent-final.json) |
| wikitext | compact-swiglu-kernel-h384 | 3.946195 | 3.971140 | 2,359,296 | 407.5 | 329.36 | [record](../records/compact-final-v1-wikitext-compact-swiglu-kernel-h384-final.json) |
| wikitext | full-swiglu | 3.991952 | 4.022206 | 8,454,144 | 560.7 | 415.16 | [record](../records/compact-final-v1-wikitext-full-swiglu-final.json) |
