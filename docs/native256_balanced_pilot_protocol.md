# Balanced single-token fixed-position pilot (2026-10-17)

User authorizes a NEW 512-update-per-arm experiment, with language replay and
64 old tokens → one256 code → one attention position. Native256 and both completed
packing pilots remain immutable. No growing families or subagents.

Both arms start from the same copied partial native step8750 state used by the
previous pilots, with fresh Adam and original packing weights. This is diagnostic
continuation, not fresh two-epoch language training or continuation of a failed
pilot. Reader23,209,216 parameters; trainable-packing arm additionally updates
the existing2,080,000 packing weights. No added production operations or KV.

Training has1024 new four-case units (4096unique examples), two shuffled passes,
8192 reading exposures. A unit has two matched memories containing different
values, each asked with two paraphrases. Values are exactly eight one-token words:
red, green, white, black, water, fire, earth, air. Every16-example reading batch
contains EACH value twice; batches, not individual units, are shuffled. Thus a
constant answer scores12.5%, and cannot get both changed values correct.

Facts always begin at global token15, the value is token24, and the whole fact
ends at26 (zero-based indices, end exclusive). Names have two BPE tokens. The source history begins with the fact and
36 one-token filler words; there is no target-valued noise in filler. A paired
history differs at EXACTLY token24, preserving all raw input, lengths, masks and
positions. The whole query begins at77, after the completed64-token block;
the correct value+EOS target has EXACTLY2 tokens and fits raw64. No unknown QA
targets, no positional variation in this stage, no decoder-success filtering.

Fresh balanced development/test groups cover ID, new names, new wording, eight
held single-token answer words, composed transfer, and absent-entity cases. Each
is256 cases; missing cases are evaluation only. A128-case actual training subset
is selected as complete balanced batches from early and late training. Retain
the previous six single-fact test groups plus mixed missing and six two-record
test groups as already-inspected regressions, with no claims of a new blind test.
Fresh fact blocks are disjoint from ALL previous cases and other fresh splits.

The language segments/order match the prior8192 replay segments exactly. Each
update has16 LM+16 reading examples, micro8; same .75LMCE+.25answerCE+.05frozen-D
reconstruction of four fact blocks. FP32 AdamW.01, clip1, LR3e-5cos3e-6/warm30,
packing LR.1 reader. Prefix/decoder stay frozen; cache their old features read-only
and encode only new blocks. Recompute final codes through each arm's current
packing weights for reading AND language. Decoder expansion is training-only.

Before training: meaningful CPU objective/microbatch/frozen/resume and balancing/
position tests; independent all-case source/token/label/cache/order audit; fullsize
zero-update GPU gradient/causality/memory preflight; actual first-update audit.
Stop exactly512 each, no adaptive extension. During live work do not edit guarded
helpers or original assets. Unique checkpoint format prevents native resumes.

Evaluate free full-value+EOS, teacher first-value decision (separate from EOS),
constant/chance baseline, both-swapped-values and all-four success, blocked and
shuffled codes, actual train fit, held words/names/wording, absent-fact hallucination,
all retained regressions, full7831-segment original dev, historical312, original-D
fact reconstruction and code drift. Independently rederive saved metrics and
verify all protected hashes. Better token loss, output changes, or isolated hits
near12.5% do not establish reading. Language gain over partial parent is confounded
with extra replay without an LM-only control. One seed and synthetic tasks limit
inference. No adoption or further stage automatically; broader goal stays unmet
unless robust held-out reading and language preservation are demonstrated.

Prespecified value-association check: compare exact/EOS predictions against true
labels and the opposite memory's value label for the same query. Flip labels for
whole four-case units, preserving eight-way balance and shared query priors;
10,000 fixed-seed flips provide a one-sided conditional check. Constant answers
have zero association. This concerns fixed checkpoints, not training-seed
uncertainty, and cannot turn a narrow toy result into broad factual retrieval.
