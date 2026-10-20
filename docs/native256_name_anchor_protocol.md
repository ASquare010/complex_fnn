# Same-template entity-anchor curriculum512

Prospective protocol2026-10-18. No completed arm is resumed or extended. Two NEW
arms use identical copies of the shared FRESH seed47 initial state from the
completed selective-address pilot, fresh Adam and the unchanged empty-history
compatibility model. All23,209,216 reader parameters/one position per64 old tokens/
original-code V cache/inference operations are unchanged. Encoder is separate and
frozen, including packing. Frozen pretrained word decoder is counted in reader.
Native256/original corpus/caches/completed models and sources remain immutable.

The verified feature gate shows that the linear unpack keys can distinguish
ordinary entity names when the query class is supplied as an ORACLE. It does not
show the LM can extract that class or generalize broadly. This pilot teaches the
unshifted address score to find the queried entity's FIRST BPE before value offset.
Source positions enter only loss targets, never model inputs or inference.

Both arms use identical data/order/labels/LR: value-only control versus name-anchor
curriculum. First128updates: .75 original LMCE, with curriculum arm additionally
1.0 name-anchorCE. Remaining384: both .75LMCE+.25answerCE+.25value-addressCE;
curriculum adds.25name-anchorCE. Control name weight0. The warmup intentionally
precedes answer/value supervision in BOTH arms. Record weighted-exposure counts
separately: source/target labels seen during warmup are not answer-trained labels.
Name loss is normalized over FIRST query decisions (including missing=NULL),
not EOS tokens. No all-four interaction loss, decoder features or other new loss.

16 original language segments+16QA examples/update, micro8, FP32 AdamW.01, clip1,
warm30, LR3e-4 cosine to3e-5 over512.512updates/two NEW4096-example source passes;
NOTtwo original language corpus epochs. Both receive the same8192 original LM
segment exposures. Main inference is unchanged, so do not infer a speed win.

Training uses the previous4096 genuine-source examples/codes, with both entity
questions in every quartet changed to the SAME template and equal raw phase.
What/Give is balanced by unit, never correlated with question/entity role.
Source entity order/value swaps/positions and128 balanced positive values remain.
6.25% examples missing. Source name anchor is regex/token-offset derived.
This is NEW loss/data curriculum from fresh weights, not a trained continuation.

Fresh held BLOCKS are disjoint from all previous address-pilot source blocks. Evaluate
familiar/128QA-held IDs/names/wording/positions/missing/composed, BOTHquestions/
BOTHmemories/all-four and both record orders, query changes, actual positive and
missing training fit. Held IDs were known to frozen codec and ordinary replay.
New-name tests remain essential: seven learned names alone cannot establish
arbitrary-entity binding. Complete answer+EOS and first/name/value-address scores
are distinct. Blocked/shuffle controls, full original LM dev and historical312.

Before training: independent source/query/label/balance/disjoint/order audit;
CPU actualobjective/microbatch/exact serialized Adam resume/zero-name-coefficient
parity; full-size zero-update preflight. Pause after actual first update, audit,
then resume that same checkpoint. Never use original/native trainers. All guarded
sources remain frozen while live. Independent training/results/provenance audits
required. No adoption/automatic extension; broad quality, true all-raw reference,
fresh two-original-epoch comparison and end-to-end efficiency remain unmet.
