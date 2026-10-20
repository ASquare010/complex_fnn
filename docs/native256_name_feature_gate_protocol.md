# Name-key linear feature gate

Prospective component diagnostic,2026-10-18. No LM, encoder, codec, production
source, old checkpoint or cache is edited or trained. Completed pilots stay
immutable. The23M-reader/two-original-epoch/64:1 objective remains unmet.

The current address controller scores a query against125 linear unpack features
at each source position. Test whether these features permit entity-name matching
before running a new full LM curriculum. Compare with the trained nonlinear
one-position decoder's256-dimensional features. This is NOT question extraction
or free QA: query entity class is supplied as an ORACLE to the diagnostic.

Reuse genuine frozen source codes from the completed address pilot. Regex and
tokenizer offsets independently identify FIRST entity-name BPE positions. Seven
training names have distinct first BPE IDs. All other positions are class OTHER.
Build125-feature and trained256-feature banks for COMPONENT training only; never
feed expanded positions to LM attention or store them in a production cache.

Fit fresh seed47 linear8-class probes, per-channel affine standardization from
training ONLY, identical balanced512-point batches (64/class) for1024updates,
AdamW LR.02, zero weight decay. Log at128/256/512/1024. This is a convex linear
diagnostic with repeated feature exposures, not the final two-epoch budget or a
capacity-impossibility proof. Decoder/encoder/probe feature banks stay frozen.

For an oracle known query name, score each source position by its name-class
logit minus OTHER logit. Choose maximal margin, or NULL if max<=0. This remains
linear in unstandardized features, with a query-dependent constant: representable
by the current unshifted address scoring/gate when given the correct query class.
Evaluate name-anchor accuracy, both swapped memories and all-four units on actual
training and existing familiar/values/wording/positions/missing cohorts. Do not
claim unseen-name transfer because those names have no probe class.

Tests verify exact affine reparameterization and feature extraction against the
public trained unpack/reader. Independent evaluation rederives source labels,
all saved classifier predictions and oracle addresses from saved weights. All
old protected files and completed pilot files remain unchanged. A successful
probe only justifies the next LM name-anchor curriculum; it does not satisfy QA,
language, reasoning, cache/speed or final full-budget requirements.
