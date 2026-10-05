# Coupled channel curves: results

## Current bounded refinement

Self-Curve's wider-initialization variation (`curve-wide`) advanced to paired-seed
confirmation and is now the frozen overall research choice. Seed101 full validation improves from 2.598287 to 2.591014 on
TinyStories and from 4.288112 to 4.272187 on WikiText, with unchanged 2,121,728 FFN
weights. It ranks first in the six-recipe development comparison. Restoring its
own initial curves gives WikiText 4.272974; this intervention alone does not
identify the cause of the improvement. All removal evidence is retained in
`dump/compact-refinement-v1/`. [Round results](../../docs/compact_refinement_result.md)
and [confirmation protocol](../../docs/compact_refinement_confirmation.md).

Confirmation seeds211/307/401 improve over the parent in all six paired runs.
Mean TinyStories NLL is 2.592300 versus 2.605145; WikiText is 4.272862 versus
4.282949. The observed resource medians meet the parent-relative limits, but
variable full-baseline timings do not establish a speedup. The independent
seed509/8000-update comparison and held-out tests are complete and audited.
Curve-Wide beats its parent and compact SwiGLU h384 on validation and test in both
corpora. Held-out NLL is 2.020291 on TinyStories and 3.966762 on WikiText, versus
full SwiGLU 1.974934 and 4.022206. It is the final selected compact recipe; the
full model retains the TinyStories quality advantage. See the
[final summary](../../docs/compact_refinement_summary.md).
[Frozen choice and limitations](../../docs/compact_refinement_final_selection.md).
