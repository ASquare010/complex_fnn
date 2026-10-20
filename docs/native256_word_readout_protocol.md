# Broad oracle-position word readout: prospective component gate

Previous turn made progress: a matched position curriculum improved language
slightly but failed robust unseen-position/value/binding retrieval. The full
23M/two-original-epoch/64:1 goal remains unmet. This NEW component experiment
tests a necessary readout before constructing a query-conditioned reader.
No native/old model/source/cache changes, no growing families, no subagents.
Do not restart or extend any completed pilot.

Use genuine frozen original encoder codes from the independently verified
4096/512/512-block reconstruction dataset. Every block has64 tokens and one
256-dimensional code. Reuse the trained decoder's position-specific unpack map
to extract ONLY an oracle-requested256-dimensional feature. Never supply the
target word, other decoded positions or raw history to code arms. The correct
position is supplied deliberately: this is NOT question answering, binding,
free generation, production inference or proof of final quality.

Train four NEW shared word readouts: selected feature; selected feature plus
original code; original code plus fixed sinusoidal position; and a raw-token
embedding oracle control. The raw control sees the actual source token embedding
at the given position, explicitly a1:1 capacity control. All use the SAME fresh
seed47 residual MLP512→512→256, RMS norms, frozen normalized codec word prototypes
and learned logit scale. The selected-only/raw arms zero the second256 input
coordinates. No per-position learned readout or trainable token table. The frozen
codec prototype basis avoids conflating word recovery with an unrelated LM
embedding basis; eventual integration must account for these weights.

Hold the exact BPE IDs of light/dark/sun/star/book/house/school/road OUT of readout
training targets in all arms. Their occurrences may remain inside source codes,
and the pretrained codec/prototypes already know them. Thus this tests held-out
READOUT labels, not words unseen by every pretrained resource. Count all train,
dev/test and fact-probe occurrences; retain all failures without decoder filtering.

Exactly TWO passes of all eligible training(block,position) pairs, shared
permutations seed4702100+epoch, batch1024/micro256, tail included once per pass.
AdamW.01, clip1, FP32, warm10, LR3e-4 cosine3e-5 over the resulting update count.
Freeze encoder, unpack map, decoder and word prototypes. Only the~0.4M readout
trains. No original language-model weights change, so replay/full-language reruns
are not needed for this COMPONENT test. These two small-data passes are NOT two
original corpus epochs or a23M-language-model comparison.

Before training: CPU selected-feature equivalence, target isolation, actual
microbatch/global gradients, frozen prototypes, serialized two-epoch Adam/cursor
resume; independent code/data/split/holdout/order audit; zero-update fullsize
GPU preflight and component latency/memory benchmark. Also measure the original
decoder run on a SINGLE selected feature (no other positions) as a zero-update
reference. A single-position attention block has weight1, so its attention value/
output maps can be evaluated without Q/K scores; verify algebra against the
actual block. This does not establish an integrated KV/cache implementation.

Pause after actual selected-arm update1, independently check Adam/order/cursor/
labels/frozen weights, then resume the SAME new run to the exact two-pass limit.
No adaptive extension. Freeze these helper/protocol sources before preparation.

Evaluate every dev/test position and actual128-block train fit. Report all-token,
non-PAD/BOS/EOS, held-ID, never-supervised-ID, per-position and exact-block metrics;
these are oracle word accuracy, not factual/free-answer accuracy. Code controls
zero or substitute another block while keeping requested positions. Evaluate all
2048 prior selective-reading fact cases with oracle positions, including balanced
held-word values, zero/shuffled codes and donor replay. Keep EOS outside this
single-word component metric. Original full-D references retain codec errors.

Independent training/result audits must derive labels and positions from source,
check all actual optimizer steps/orders/label masks, recompute saved token scores,
compare reference predictions and verify protected old assets. Full hashes before
the pipeline and after evaluation plus an independent final pass; relevant
source/data/codec hashes at each intermediate stage. No additional GPU worker.

If the trained local readout works broadly, proceed separately to learned
question-to-address selection and fit the whole reader/cache/parameter/speed
budget. If it fails even with the oracle, do not call it a retrieval solution or
launch full-budget LM training from an unsupported feature bridge. A~3.5M
component (unpack/prototypes/readout) is not automatically free in a23M model.
Final total-versus-reader budget clarification remains pending. No raw-history,
decoded-token or extra memory-position cache may be added in production.
