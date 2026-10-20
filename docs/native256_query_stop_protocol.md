# Independent local STOP prototype

18 October2026. Motivation is the independently verified transfer decomposition:
with gold access value+period supplied, generator EOS wins212/224 endpoints but
the final mixture wins only10/224;202 correct generator stop decisions are
overridden. Old access copy EOS mass is zero. Notebook copy EOS mass averages
.991615 and stops256/256. This is teacher-forced evidence, not free QA success.

Implement a NEW model, leaving V2/native/original checkpoints/sources unchanged.
Keep two local readers, width256, raw64, one global memory/KV slot per64 old tokens,
FP32 word-bit format and the existing cache class/arrays. Reader budget remains
exactly23,209,216, encoder separate. A512-parameter final two-way STOP head replaces
one FFN hidden unit (total31368 to31367). This is a new fresh architecture, not
tensor-identical to V2 and not a trained checkpoint migration.

Let s=softmax(Wstop*h+[-4,0])[STOP], initially sigmoid(-4). Keep the ordinary
copy/generator mixture and internalNULL mass unchanged. Redistribute ONLY the
nonNULL copy mass: multiply old lexical-copy probabilities by(1-s) and put
s*(1-Pnull) on EOS. This is a local output-vocabulary action, NOT an attention
position or stored KV entry. Existing old EOS copies remain allowed. With empty
or blocked memory (Pnull=1), the output equals the old generator fallback and the
new head contributes zero. It does not read answer labels at inference.

Main training remains original assistant CE. A prospective training-only auxiliary
teaches this head from ORIGINAL assistant EOS/nonEOS labels, with detached hidden
states: .02 times the mean of the globally normalized EOS and nonEOS class means.
Class0 means STOP/EOS, class1 means continue. Normalize across effective updates,
not per microbatch; labels -100 excluded. No auxiliary core/embedding/old-reader
gradients, new fact text or teacher decoded history. This objective is prepared
only; no new training run is authorized by a passing zero-update test alone.

Before any training: independent normalized-distribution/gradient and NULL-fallback
checks, exact parameter count, active public/cached/chunked/rollover equality and
unchanged cache bytes/attributes, causal future/padding isolation, actual objective
microbatch/serialized Adam tests and full-size GPU gradient/capacity/timing gates.
Any later pilot must use a new run/format and matched control, preserving V2.

This addresses one measured stopping mechanism. It does NOT fix initial value-only
format selection, missing answers or general reasoning by construction. Future
claims require free full answers and all previous counterfactual, missing, language
and transfer checks; final goal still requires fresh full two-original-epoch parity.
