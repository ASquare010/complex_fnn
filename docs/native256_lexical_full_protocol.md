# Fresh original-corpus two-epoch lexical64 versus true1:1 comparison

Prospective18October2026. The reference and exact-kernel GPU/cache/gradient gates
completed and passed independent audits. No quality improvement is established.
Old/native/growing sources, datasets, vector caches and checkpoints stay immutable.
This NEW run is the full-budget comparison, not continuation of any diagnostic.

Two NEW fresh seed47 readers,23209216 trainable parameters each, FP32:
`compressed64` uses the verified optimized lexical LM (21x256,559616 reader
weights, FFN total31374, two read units). `raw1` is a conventional21x256 native
core (FFN total32467), no lexical reader, one ordinary embedding/global KV per
old word. The frozen encoder is separate and only used by compressed64; old
semantic vectors are read from the original immutable FP32 cache. Each old64-token
vector gains exact literal IDs in its low bits during batch preparation. This
is added lexical metadata, not recovery learned by the original encoder. Input
data/caches use a new explicit format contract, never silent old-cache reuse.

Use EXACT original unified-chat-v4 segments, Python `random.Random(47+epoch)`
orders, effective32 batches, every original assistant label,2 complete original
passes=48762 updates=361357104 assistant labels. No new QA training examples,
Name/position/gold-span auxiliary losses or extra answer weighting. Original
AdamW.01, clip1, LR3e-4, warm200, cosine to3e-5 over the full48762 schedule. Same
global assistant-label CE normalization before clipping in both arms. Microbatch
may differ to fit hardware; allbatch gradients and resume equivalence must pass.
Default verified compressed micro8; raw micro1 requires separate real full-size
backward/Adam-capacity preflight. No raw baseline history truncation/detached old
KV gradients. Both see identical old words/current windows/labels; the1:1 control
gets all old words as ordinary transformer positions, not encoded summaries.

Old raw64 packing/BOS and original512-segment label boundaries are shared. Raw1
expands each old block to64 physical word slots and multiplies visibility by64;
the compressed main transformer still has ONE global slot per64 old words. No
decoded old-word bank is retained on its cache. Metadata and unique-vocabulary
tables are temporary. On-the-fly CPU bit packing avoids another5GB cache or
recomputing the frozen encoder during training. The raw transport wrapper parses
IDs before embedding; it must not use semantic code values in its forward.

Before training: source/data/cache identities+hashes; independent all-case labels,
source-block/metadata/mask derivation; independent fresh initialization; full-size
gradient and actual raw-reference capacity gate; CPU global objective/microbatch/
actual serialized optimizer/RNG/sampler resume across epoch boundary. Save initial
states and empty optimizers. Audit the first ACTUAL update before resuming that
same checkpoint. Unique full-budget checkpoint format/arm/source/recipe/cursor
guards; never resume through any original/pilot trainer. Safe SIGINT/stop-after
pauses only after a complete update. All actual orders/LR/label counts/Adam steps
must be independently checked. Do not extend completed older pilots.

Evaluate at disclosed checkpoints with full original dev and source breakdowns,
held original memory free answers and source/query swaps, blocked/shuffled
controls, short conditional completion and value-only formats, prior four-way
access suites, new names/values/positions/wording/missing/multiword/multiblock and
reasoning cases, plus historical312. Legacy synthetic suites were NOT training
data for these new models; do not call them current training-fit or claimed
in-distribution reading. Both models use identical inference prompts/scoring.
Match full-budget weights when claiming parity; partial1000-update diagnostics
are not final two-epoch evidence. Fixed-checkpoint paired-document intervals do
not estimate seed uncertainty. Exact value/EOS, paired/all-four, missing answers,
language preservation and reasoning remain separate requirements.

The current speed evidence is mixed: fast64 is~2.9x faster batch4 loaded forward
and~25x smaller total allocated cache at2496 old words, but~20% slower single-token
decode than conventional raw1 in the measured PyTorch setup. Rollover includes E.
Report this limitation and complete actual training throughput/peaks; no universal
64x compute/latency claim. Full success remains broad comparable quality under
the original23M-reader/two-epoch/64:1 constraints, with strong true1:1 comparison.
