# Signed-context FFN: results

## Current bounded refinement

The smaller, finer-grouped `signed-fine` advanced to paired-seed confirmation.
Seed101 full validation improves from 2.603935 to 2.602645 on TinyStories and
from 4.277294 to 4.276089 on WikiText. FFN weights fall from 2,490,368 to 2,359,296.
These small gains require repeat-seed confirmation; they are not evidence that
grouping alone caused the improvement because width also changed. All frozen
removals remain in `dump/compact-refinement-v1/`.
[Round results](../../docs/compact_refinement_result.md) and
[confirmation protocol](../../docs/compact_refinement_confirmation.md).

The fresh-seed gains did not hold: mean TinyStories NLL is 2.607223 versus parent
2.600803, with losses in all three pairs. WikiText is 4.271804 versus parent
4.271632, with one win in three pairs. Signed-Fine is not selected. The signed
parent remains a strong compact alternative and has slightly better WikiText
mean loss than Curve-Wide, but ranks behind it on the registered combined score.
[Complete paired analysis](../../docs/compact_refinement_confirmation_analysis.md).
