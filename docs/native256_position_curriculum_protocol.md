# Position-varied curriculum: prospective matched diagnostic

The previous read-only diagnosis was progress: the 512-step fixed-layout readers
can recover familiar words, but shifted facts and entity selection fail. The full
23M/two-original-epoch/64:1 quality goal remains unmet. This new experiment isolates
position training; it is not a new architecture or a final quality comparison.

Train ONE new frozen-packing arm for exactly512 updates. Reuse the immutable
completed balanced fixed-position frozen arm as the matched control; do not
restart or extend any prior run. Both arms initialize from the SAME copied native
step8750 reader state with fresh Adam and original frozen packing weights. Native
source/run/cache/checkpoints stay unchanged. No subagents or growing models.

Keep exactly the control's1024 four-case units, two reading passes, batch order,
within-batch unit order, source word bags, raw queries/targets, language segments,
learning-rate schedule, optimizer and objective. Change only where the fact is
placed within its completed64-token block. Train at offsets0/8/16/24 from the old
position, with the same offset for all16 reading cases in a batch. Choose offset
index `(original_batch_index + reading_epoch) % 4`; each position receives64
batches per pass, all eight values occur twice per batch, and each unit sees two
different positions. Facts remain complete in the first64 tokens, history body
length47, raw-query body position77, answer one value token plus EOS.

Reader23,209,216 parameters; frozen original encoder/packing/decoder. Exactly one
attention/KV position per64 old tokens, raw64, no decoder features or raw-history
side channel. .75LMCE+.25answerCE+.05D-reconstruction (no packing gradient in this
arm),16LM+16reading per update, micro8, FP32 AdamW.01, clip1, warm30, reader LR3e-5
cos3e-6. Reuse cached prefix features for old blocks; encode only new shifted
blocks. Two passes of this auxiliary data are NOT two original corpus epochs.

Fresh evaluation:64 aligned units/256 cases at every offset0/4/8/12/16/20/24/28/32/36,
so trained positions, interpolation and extrapolation are separate. Same fresh
source word bags and raw queries across offsets. Fresh names, wording, answer
words and missing-entity groups each at0 and12, plus composed changes at12. All
fresh evaluation source blocks disjoint from old data and this pilot's training.
No decoder-success filtering. Retain the eight already-inspected selection
diagnosis groups as regressions. Evaluate the fixed control and new reader on
the SAME fresh cases; reuse verified fixed outputs for unchanged regressions.

Measure unrestricted greedy full value+EOS, first-value teacher decisions, both
changed values correct/all-four, blocked/shuffled controls on first64 balanced
cases per group, absent-fact hallucination, novel values and two-record selection.
Measure full original7831-segment language dev for the new reader and compare to
the fixed arm's saved same-mode arrays, including paired-document confidence
intervals (fixed checkpoints, not seed uncertainty). Historical312 checks and
original-D reconstruction on every unique evaluation block remain required.
No broad QA, reasoning, adoption or full-budget parity claim from a toy gain.

Before training: CPU position/order/source and objective/frozen/resume checks;
independent all-case serialization, source-derived answers, word bags, cache and
balanced exposure audit; fullsize zero-update gradient/causality/memory preflight;
pause after actual update1 and independently audit Adam/cursor/labels before
resuming the SAME new run to512. Unique checkpoint format prevents native resumes.
Freeze helper sources before preparing; never edit guarded source while live.
Independent final training and result audits, protected hashes and unchanged
packing must pass. Stop at512 regardless of outcome, no automatic extension.

If layout transfer improves, the next separate test can add same-word-bag entity
swaps. If it fails, use the evidence to investigate reader conditioning/curriculum
before launching full-budget training. Neither outcome proves universal capacity
limits. Full adoption still requires fresh matched two-original-epoch language,
unseen-value, binding and reasoning evaluation at the requested compression and
parameter budget. The historical61.013% reference includes raw512 PLUS older
memory; a pure all-raw1:1 reference is still missing.
