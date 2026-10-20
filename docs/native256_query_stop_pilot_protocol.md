# Fresh512-update paired STOP supervision pilot

18October2026. Full goal ACTIVE/UNMET; previous turn isolated copy stopping and
added a separate prototype. Six model CPU tests/three combined-objective tests
and full-size GPU preflight passed. This is a NEW experiment, never a resume or
extension of the completed V2/native runs.

Both arms use the SAME new23,209,216-parameter STOP reader, independent identical
fresh47 initial tensors/RNG and empty Adam, original batches/orders/labels and
original48,762-update/two-epoch LR horizon. Encoder separate/frozen; raw64 and
one global old memory/KV position per64 tokens. No new training examples, decoder
features or decoded old history. The added literal-ID metadata limitation remains.

Ordinary arm: original global assistant CE+.05 query+.02 critical-source CE.
Supervised arm: same objective+.02 balanced original EOS/nonEOS STOP CE, detached
selected final hidden states. Global EOS/nonEOS denominators span effective32;
microbatch8. Both still train STOP weights through ordinary CE. This isolates the
additional loss; it does NOT isolate STOP architecture versus old V2 at step512.
Shared fresh architecture differs from V2 by512 weights/one FFN hidden unit.

Use new roots `dump/native256-query-stop-pilot-20261018/{ordinary,supervised}`,
each unique format `native256-query-stop-pilot-CONDITION-v1`. Each contains the
existing `compressed64` arm layout so immutable evaluator/data/auditor helpers
can be reused through explicit reversible process-local scopes. Never resume
through any old trainer. Final comparison requires both initial tensors/RNG exact,
same all actual labels/orders/LR/cursors and complete independent verification.

Export both, independently audit initial/data/schedule/source/protected guards.
Run one actual update per arm, save, independently check all Adam1 counters,
targets/query/source/EOS/other denominators and recipe/cursor; only then resume
that new arm for511 additional updates. No automatic extension beyond512.
FP32/noTF32, AdamW.01/clip1, original warm200/cos3e-4 to3e-5. Record every batch,
loss component/denominator and source snapshot; all source files frozen while live.

Evaluation after full training audit: full7,831DEV/1,859,388labels; identical
natural positives/missing/changed-values/changed-questions/blocked; all seven
legacy access groups/controls; historical312/numeric53; proper notebook both
questions/both swapped memories/full and completion/all-four/blocked controls;
transfer original32cap/value-free-prefix16cap/blocked/donor cases from the completed
diagnosis. Preserve original strict EOS scoring and separately label partial
value prefixes. Independent aggregate/source/output/public CPU replays required.
Record fixed-checkpoint document/unit intervals, not training-seed uncertainty.
This bounded pilot is not two original epochs completed or final quality success.

One GPU worker; independent CPU audits may overlap. Original/native/V2/checkpoints/
data/caches and reserved student TEST/novel literals remain untouched. Existing
completed control receipts remain immutable; do not restart failed prior arms.
No adoption based on EOS/auxiliary loss alone; require free facts, missing answers,
counterfactuals, language and efficiency together. Full goal remains broad fresh
two-original-epoch parity against a strong matched1:1 reference.
