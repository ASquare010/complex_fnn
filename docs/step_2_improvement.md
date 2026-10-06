# Step 2 improvement: direct recovery through an ordered bottleneck

Status: first 2,000-update revision complete, 2026-10-05. Full validation: 911/1,000 exact paragraphs and 99.8854% token accuracy. The aim remains reliable
near-lossless recovery from compact, reusable memories. A working training loop
or lower teacher-forced loss is not sufficient completion evidence.

## Evidence motivating the revision

The original 2,000-update pilot recovered 0/1,000 held-out paragraphs exactly.
The user's notebook checkpoint also has 2,000 completed updates, not 5,000;
its saved configuration is four encoder/two decoder blocks, microbatch six with
four-step accumulation. Its dataset manifest still has 50,000 training paragraphs.
Changing notebook constants does not retroactively run more updates or rebuild
an existing dataset. Preserve both checkpoints and the user's notebook settings.

Positions were already present in the original model. Its span pooling averages
encoded positions, and training supplies correct previous output tokens. These
are plausible sources of difficulty, not proven exclusive causes of failure.
Cross-entropy already gives a differentiable reward for each correct target;
an argmax exact-match count is not a useful additional gradient. Start the new
architecture with ordinary CE before testing auxiliary loss terms.

## First controlled candidate

`position_compressor`: width 256, four encoder blocks, one decoder block,
Curve-Wide FFNs, eight-token spans, 256-token input limit and vocabulary 4096.
Within each span, concatenate encoded features in position order and learn a
projection from 8Ãƒâ€”256 values into one 256-value latent. Expand each latent back
into eight ordered feature positions, refine them with the small decoder and
predict all original tokens in parallel. There are no U-Net skip connections,
source-token decoder inputs, teacher forcing or target embeddings in decoding.

Exact input token length is explicit int64 metadata. This is an additional
advantage over the old EOS-predicting model, and its eight bytes per paragraph
are included in payload measurements. Do not attribute any improvement solely
to depth, pooling or loss: several architectural factors changed together.

The first run uses the unchanged pinned FineWeb-Edu 50k/1k/1k split, seed 17,
2,000 updates, microbatch four with four-step accumulation, BF16 autocast,
AdamW 3e-4, weight decay 0.01, 100 warmup updates and clipping at one. Full
validation is recorded every 500 steps. The reserved test split stays untouched.
Measure token accuracy, paragraph exact match, edit distances, correct/zeroed/
shuffled memories, parameters, runtime, vector counts, feature counts and bytes.
Checkpoint and source identities are recorded under a new run name.

Each model window retains a 256-token bound. A verified codec supports consecutive
windows for longer strings and encoder-only export. The new position compressor
notebook was executed through saved-memory recovery and encoder export. Chunking
is not semantic streaming. Independent challenge evaluation and auxiliary loss
comparisons remain outstanding; the near-lossless gate is not met.

## Interpretation and next gates

### Bounded continuation and diagnostic protocol

Continue the preserved 2,000-update checkpoint to 3,000 total updates using
ordinary cross-entropy, the same optimizer and sampling state, and a cosine
learning rate from 3e-4 to 3e-5 over these 1,000 additional updates. Keep the
architecture, data and effective batch unchanged. This tests additional training
with a declining learning rate, not a clean loss-function comparison. Save under
`dump/position-continuation-3000-v1/ce`; preserve the parent. Full validation at
2,500 and 3,000 updates; use the final checkpoint for the reported result.

Ten fixed diagnostics cover names, numbers, negation, repetition, whitespace,
Unicode, literal control-marker strings, code, a short input, and a 992-token
repeated passage. The parent passed seven, but failed whitespace, Unicode and
code. These CPU/FP32 checks are separate from CUDA/BF16 corpus validation and
are not representative accuracy estimates. Evidence:
`records/position-challenge-2000-v1.json`. This exposes why the existing result
cannot support recovery of arbitrary text. The diagnostic strings are not
training data. Reserved test evaluation follows the validation gate.

Coverage audit of the unchanged 50k training paragraphs found zero newline and
tab token occurrences, and zero occurrences of one byte used by the emoji probe.
Several Arabic/CJK byte tokens occur only tens of times. This provides a concrete
data-coverage explanation for some diagnostic failures, not proof that coverage
alone solves them. See `records/position-coverage-audit-v1.json`. A later general
text experiment needs explicit character/token coverage and held-out combinations;
do not inject these diagnostic strings into training or silently alter this split.

A vector-count reduction is not raw-string byte compression. Report L tokens,
N=ceil(L/8) vectors, NÃƒâ€”256 features, dtype, payload and length metadata separately.
Preserving ordered features does not establish that slots are semantic concepts
or that another LLM can use them without training.

### Token coverage experiment

From the preserved 3,000-update checkpoint, run 1,000 additional CE updates with
the same architecture, optimizer state and continuation schedule. Mix 50,000
original training paragraphs with 50,000 deterministic synthetic rows (seed 117),
each 32â€“128 uniformly sampled token IDs from 3â€“4095. Verify every ordinary token
is represented; control IDs are excluded. Sample rows equally from this combined
pool. Evaluate 256 independent synthetic rows (seed 118), checking no exact row
overlap, plus the full original development set and unchanged diagnostic prompts.
Do not reopen the previously used test set for selection. Save the new run under
`dump/position-coverage-4000-v1` and preserve the notebook's current model until
results justify changing it.

Random token sequences may not represent valid UTF-8. Their metric is exact token
recovery, not decoded-string equality or language quality. This experiment tests
coverage/general copying; it does not establish semantic concepts or replace
the eventual need for independently held-out real multilingual and formatted text.

### Repetition and concentrated-alphabet coverage

The 4,000-update model reaches 100% recovery on the development paragraphs and
uniform-token validation, but only 73/128 dense Unicode/formatting stress inputs.
Full precision also fails, and shrinking inference windows to 8, 32 or 64 tokens
reduces exact recovery to 44, 56 or 63 of 128. Keep the architecture and 256-token
windows unchanged. Evidence: `records/position-context-diagnostic-v1.json`.

Register a 1,000-update continuation to step 5,000: 50k natural rows, 25k uniform
token rows and 25k synthetic pattern rows. Pattern rows have 1â€“256 tokens drawn
from a per-row random alphabet of 1â€“64 ordinary token IDs; half repeat a random
1â€“16-token motif. Seeds 217/218 generate training/development patterns; remove
exact overlaps. Seeds 219/220 generate uniform training/development sequences.
Use the same CE, effective batch and cosine continuation schedule. Freeze source
snapshots and preserve all parents. Evaluate full natural development, independent
patterns and uniform sequences, and the existing Unicode development probes.
These pattern rows contain no diagnostic text. They test broader copying patterns,
not a new language-model objective. Prior test results remain tied to old weights.

Use Ã¢â€°Â¥99.9% token accuracy and Ã¢â€°Â¥95% exact paragraph recovery on full validation as
a provisional near-lossless research gate, then verify on reserved data and a
separate challenge set. This is an operational test threshold, not a guarantee
for every possible string. Do not weaken it after observing results. If it is
missed, investigate error concentration, slot budget and learning progress.

## Final evidence and scope

The frozen 5,000-update model passes the predeclared near-lossless operational
threshold on development and fresh natural-language confirmation (both 1,000/1,000
exact with 100% token accuracy), as well as independent repeated-pattern recovery
(511/512 exact; 99.9569% token accuracy). Fresh synthetic Unicode confirmation is
256/256 exact. This supports the requested improved reconstruction component;
it does not prove universal exactness. See `step_2_results.md` and the model report.

The smaller decoder, ordered bottleneck, broader copying data and ordinary CE
achieved this result without an auxiliary Poly-1 term. No loss-function superiority
claim is made. Full model 4.147M parameters; exported encoder 3.226M. Compression
counts and bytes are reported separately. Future LLM training on these vectors,
semantic streaming memory and optimization remain later work.
