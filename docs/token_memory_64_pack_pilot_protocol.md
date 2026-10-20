# Native256 frozen versus trainable packing pilot (2026-10-17)

User authorizes a new, separate 512-update-per-arm experiment. Original native256
source/run/checkpoints and all older sources/caches remain immutable. No native
trainer was active at launch (Windows process inspection); its last saved
checkpoint is incomplete at step8750 (last log8954). Both arms initialize from an identical immutable copy
of that last checkpoint, recording its actual step/hash. This is a diagnostic
continuation, not a fresh two-epoch comparison or a completed-native result.

Both use the same native256 reader (23,209,216 parameters, 21 layers), raw64
mask/legacy positions and one 256-number code per completed 64 tokens. The
frozen arm trains only the reader. The packing arm also trains the existing
encoder compress.token and compress.span_map weights; no new layers or slots.
The encoder prefix and original reconstruction decoder stay frozen. Extra
trainable packing weights/Adam memory are disclosed separately from reader count.

Training features are FP32 frozen encoder outputs just before compression,
captured from genuine complete blocks. Both arms run the same packing operation
on these features. Caching them is valid because the upstream encoder is fixed;
codes themselves are recomputed from the arm's current packing weights. Encoder
features never enter the language reader, only the one final code per block.
Language examples also use current packing weights, avoiding stale-code training.

512 updates, each 16 original language segments plus 16 new reading examples
(four four-example units), microbatch8 after zero-update capacity/parity checks. First128 updates use one-record units;
remaining384 use two-record units. Histories and values swap, both queries asked;
20% units include missing queries. Names, wordings and two-word access values
vary; test separately holds out combinations, names, wording, answer words, and
composed changes. Full source prompts and fact-bearing blocks are split-disjoint.
All facts are entirely inside the first completed64 block; whole question and
complete answer/EOS fit the active raw64 window. No decoder-success filtering.

Loss .75 original assistant-token CE + .25 full-vocabulary answer CE + .05
original frozen-decoder reconstruction CE on four old fact blocks/update. Last
term has gradients only in the packing arm. Decoder expansion is training-only;
no reconstructed tokens/features or extra KV enter production. Global label
normalization across microbatches; FP32 AdamW .01, clip joint norm1, reader LR
3e-5 cosine to3e-6/warm30, packing LR one tenth of reader LR. Same example order,
labels, targets, schedule and sources in both arms. Unique checkpoint format.

Before training: independently audit serialized cases/answers/positions/partitions,
feature-versus-live-code parity, microbatch gradients and serialized resume, exact
initial weights and zero-update GPU capacity. Check actual first saved update
before continuing. Do not modify guarded sources while a worker runs.

After512 each, evaluate parent/frozen/packing with free complete answers+EOS,
both-question/both-memory success, missing values, blocked/shuffled fact-code
controls, teacher loss (not factual success), actual training-fit and all held-out
groups; full original dev through each arm's current codes and historical cases.
Evaluate reconstruction fidelity and code drift. Recompute results independently
from saved predictions/logs and recheck all protected hashes. No automatic
extension/adoption. Preserve64:1 position compression; it is not64x total speed
or a global losslessness guarantee. One seed and partial native parent limit claims.
