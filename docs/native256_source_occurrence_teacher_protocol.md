# Conditional source-pointer diagnosis at the fixed1,000 checkpoints

18 October 2026, before this analysis. No optimizer or student TEST scoring.
Compare the completed ordinary pooled-key1,000 reader and the new source-aligned
1,000 reader on ALL348 original development memory dialogues. Check each fixed
checkpoint/training receipt/source hash and original dev token/mask/metadata.
Use unchanged public packed inference, capturing the probabilities already
computed by each read unit; source coordinates are only scoring targets.

Report correct SOURCE-occurrence probability/top1 separately from probability
summed over all occurrences of the same word ID. Separate first substantive
name piece, all name pieces, first color piece, all value pieces and first
missing-answer token. Count285 first-name/color decisions and63 missing decisions.
Also record final LM full-vocabulary top1, generator target probability, copy
mixture mass and source-copy versus generator contributions at those positions.

These are TEACHER-FORCED conditional statistics. Later name pieces and colors
have correct earlier answer tokens, including the name, in context. High accuracy
there does not prove that the original question selected the right record or
that a free answer will be correct. First-name measures expose that confound;
full free replies, changed queries and all-four generation remain decisive.
This read-only diagnosis can identify whether source selection or final mixture
is a remaining bottleneck, but cannot establish causal sufficiency, convergence,
general capacity limits or final two-epoch/broad quality parity.
