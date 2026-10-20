# Preserve raw audit batching without training a second model

18October2026, before new v2 preparation. Span full-v1 preparation saved correct
fresh weights/RNG/original orders and ZERO updates. The data auditor intentionally
compares compressed and raw packets, then audits initial states only for trained
arms. v1's single-arm tuple incorrectly rejected its raw packet. Preserve that
zero-update export and failure receipt; do not edit its guarded sources.

New run dump/native256-lexical-span-unified-v2, unique format
native256-lexical-span-full-original-v2. All original span model/gate/kernel,
training math/objective/orders/initialization/budget and frozen encoder unchanged.
Only a Data subclass scopes the accepted data-arm tuple to both formats inside
each original batch call, restoring it on every exit. The model factory and
prepared initial-state set still permit ONLY compressed64; raw1 cannot train.
No duplicate raw checkpoint, extra data, epochs or altered label normalization.

Regression tests use the real heaviest original train row: identical raw inputs/
labels/masks, exactly64x source positions/visibility and identical recovered IDs,
then single-arm restoration and rejection of a raw model. A nested driver failure
must restore every global callback/format/root/Data reference. New manifest guards
the entire failed v1 export with collision checks; no stored hash is overridden.

Run the same prospective full data audit, GPU microbatch8 preflight, first actual
update/audit and999-update continuation as the v1 protocol using
tools/run_native256_lexical_span_full_v2.py. All model/data/evaluation/acceptance
requirements in native256_lexical_span_full_protocol.md still apply. No real span
training has begun. Raw fixed-checkpoint GPU evaluation continues independently;
only CPU preparation/audits may overlap it, never another GPU worker.
