# Fresh original-corpus comparison of pooled memory keys

Prospective18October2026. Only change relative to lexical64 v2: nine key slots
pool causal offsets0,1,2,3,4,5..8,9..12,13..20,21..32. Exactly23,209,216 fully
trainable reader parameters, identical actual saved fresh47 tensors/RNG, separate
frozen9,426,432-parameter encoder, raw64, two read units and unchanged global KV.
The FP32 literal-ID metadata is still added information; this does not prove the
original semantic encoder lossless. Local pooling adds compute/temporary memory.

New run: dump/native256-lexical-span-unified-v1. Unique checkpoint format prevents
original trainers from resuming it. Use frozen existing data/language loss/
orders/optimizer/LR code through process-local callbacks restored on every exit.
Only compressed64 is trained; do not duplicate raw1 or extend old lexical64.
Existing native256, original cache/data and prior checkpoints remain untouched.
Reuse completed lexical64 v2 step1,000 for the one-feature quality comparison;
reuse conventional raw1 read-only at matched steps for compression comparison.

Prepare requires the completed independent zero-update span-v2 GPU audit. The
manifest archives all model/helper/protocol sources and hashes original inputs.
Both original orders files must match byte-for-byte, and every saved fresh weight
and RNG must equal actual lexical64 v2 initialization, with empty Adam. Run the
all-row/all-code/all48,762-original-order data audit and selected microbatch8
full-size preflight before training. First actual update pauses at1; independently
audit all Adam states/order/labels/cursor and guards. Then resume999 NEW updates
to1,000. No implicit extension: trainer requires explicit --stop-after.

This is NOT an extra QA curriculum. Original780,186 train segments, effective32,
FP32 AdamW.01,clip1,lr3e-4/warm200/cos3e-5 and full48,762-update two-ORIGINAL-epoch
schedule remain. First1,000 updates contain7,416,257 original assistant labels.
No packing/E training or new examples/loss weights. No full-budget quality claim
at1,000; any later continuation stays inside the same original two passes.

At1,000: frozen full original development and free factual/missing/value+query
pairs, legacy unfamiliar values/names/positions/wording, historical and numeric
probe; then same controlled notebook both-queries/both-memories full/completion
and blocked invariance. Data's OLD preparation sources are independently hashed
when reused, rather than falsely relabeled as new training sources. Result files
bind NEW model sources/checkpoint. Reaggregate and public CPU replay independently.
Require actual same original labels/orders/LR and paired document/unit comparison
to both controls. Lower loss/partial teacher tokens alone are not binding success.
Eventual adoption also requires full two-original-epoch performance, original
TEST and same-format genuinely held literal-value tests, with strong mixed512
reference retained separately. Zero all-four at1,000 is not a universal capacity
impossibility claim; favorable early scores alone do not justify adoption.

Prepared commands, each only after the preceding gate passes:

```powershell
& .\.venv-cuda64\Scripts\python.exe .\tools\run_native256_lexical_span_full.py prepare
& .\.venv-cuda64\Scripts\python.exe .\tools\run_native256_lexical_span_full.py audit --audit data
& .\.venv-cuda64\Scripts\python.exe .\tools\run_native256_lexical_span_full.py preflight
& .\.venv-cuda64\Scripts\python.exe .\tools\run_native256_lexical_span_full.py train --microbatch 8 --stop-after 1 --skip-initial-eval
& .\.venv-cuda64\Scripts\python.exe .\tools\run_native256_lexical_span_full.py audit --audit first
& .\.venv-cuda64\Scripts\python.exe .\tools\run_native256_lexical_span_full.py train --resume --microbatch 8 --stop-after 999
```

One GPU worker at a time; CPU preparation/audits may use another core while an
unrelated fixed-checkpoint GPU evaluation reads immutable sources. No speed
benchmark may overlap other work. Current raw-v2 evaluation must finish before
span preflight or training acquires the GPU. This protocol is unexecuted at draft.
