# H118 — conditional diagnosis of training variability

The previous goal turn made **progress**: H117 completed a three-seed/two-corpus
replication, established a scoped TinyStories memory component and rejected the
fixed two-corpus quality claim. The broader VRAM/quality/runtime and architectural
parameter-efficiency goals remain unmet. H118 investigates the failure before
allocating another changed recipe. It cannot retrospectively qualify H117.

## Question and selected scope

On H117's failed WikiText seed101, does the FP32 chunk/native quality gap repeat
beyond the observed variation between executions of the identical policy?
This seed is deliberately selected because it failed: this is a **conditional
diagnostic**, not an independent seed sample, population test or replication
across datasets. H117's original native/chunk runs are repeat0; retain both.

Allocate exactly **four additional fresh 800-update executions**: two native FP32
and two chunked FP32. Repeat1 order is native then chunks; repeat2 reverses it.
Each loads the same hashed H117 CPU step-zero model and empty Adam state, uses
seed101 and sampling seed10101, and uses the identical frozen H117 `run_case`
function. Only output-root/fixture labels change. Reuse the function directly;
do not copy or alter the training loop. Record the temporary output-root binding
explicitly and restore it after execution.

Keep width384/hidden456/eight layers/six heads/vocabulary4096/batch8/context512,
9,099,648 total and2,801,664 FFN parameters. Keep default SDPA, block checkpointing,
FP32 decoder/classifier/parameters/Adam moments, TF32 off, four CPU threads,
LR0.0006, AdamW(0.9,0.95), epsilon1e-8, existing decay groups and clip1. Do not
change deterministic-algorithm flags, workspace policy, optimizer or tolerances.
No new model/default, data, tokenizer, BF16 training arm or activation is added.

## Budget and measurements

New allocation: **3,200 optimizer updates / 13,107,200 training target presentations**,
plus four initial probes =3,204 main backwards. Reuse H117's frozen successful
24-backward CPU-double qualification of the unchanged execution adapter; no new
qualification backward is required. Independent audit adds four FP32 initial
backwards and zero updates: **3,208 new backwards** total. Original H117 updates
remain separately counted. No retries or extra repeats based on outcomes.

Use the unchanged full common streamed BF16 scorer at0/200/400/800 and save
model/Adam/sampler states at200/400/800 plus initial gradient tensors. Preserve
all3,200 update records, gradient/activation diagnostics, all6,444 memory intervals,
20-update warmup and all780 remaining timings per run. Record per-job allocated
and reserved peaks without subtracting anything; driver/context memory remains
outside PyTorch allocation. Record elapsed time and five zero-allocator study
boundaries. Audit resources are separate from the training measurements.

The full800-update endpoint is essential: H117's local gradient fidelity and
50-update H116 screen did not predict it. Four repetitions are a bounded
diagnostic allocation, not a new architecture sweep.

## Verification

Before execution, verify the H117 receipt, its compact/raw evidence, all78 tensor
hashes,61 maintained hashes and required frozen source chain. Preserve the seven
current navigation documents before later updates. Freeze new sources/plan,
data, source checkpoints, original two trajectories and all compared tensors.

After all four executions finish, freeze their results/tensors before independent
audit. Reuse H117's audit functions for each new run: source/initial-gradient
hashes, all3,200 batches, all12 trained states, complete native scoring of those
12 states, four initial FP32 gradient replays. Regenerate the one shared initial
state and score it natively once: **13 native scores** total. Original repeat0
audit is reused from its frozen receipt, not counted as newly executed.

Keep replay thresholds1e-5 global/1e-4 per tensor and loss/score relative error1e-6.
Compare native/chunk initial gradients at each repetition against H117's unchanged
1e-6 loss/0.002 global/0.02 per-tensor limits. Retain failures rather than changing
limits. Source/state/nonfinite/runtime failure stops and preserves evidence.

For all six trajectories, report first differing recorded training loss within
each policy, checkpoint equality and model/Adam-moment relative distances at
200/400/800. Compare all15 unordered pairs at each saved step, plus all15 initial
gradient pairs. Use symmetric relative L2:

`distance(a,b) = ||a-b||2 / max((||a||2 + ||b||2)/2, 1e-12)`.

These descriptive distances cannot establish a unique CUDA-kernel cause.

## Fixed interpretation

Let N and C be the three native and three chunked final NLLs (including original
repeat0), D=median(C)−median(N), and W=max(range(N),range(C)). Report all endpoints,
paired ratios per repetition, means, medians, sample variances, ranges and D/W
(undefined if W=0). Do not treat these three executions as three random seeds.

- **REPEATED MATERIAL DISADVANTAGE** if `min(C) > 1.01 * max(N)` and all audits pass.
  This is deliberately stricter than a positive mean: every observed chunked
  endpoint is more than1% worse than every native endpoint in this selected case.
- **OVERLAPPING REPEAT VARIATION** if the two observed NLL intervals overlap and
  all audits pass. This supports investigating repeat variability as a confound;
  it does not prove equivalence or explain the original gap uniquely.
- Otherwise **INCONCLUSIVE**. Report separately whether W covers the absolute
  original pair gap and whether the two new paired ratios exceed1.01.
- Any failed numerical/state audit makes interpretation **UNQUALIFIED**, while
  preserving observed metrics. No favorable subset can replace the full result.

None of these outcomes rescues H117's failed quality gate or proves a memory
component generally safe. Keep its numerical/resource evidence and limited
TinyStories qualification intact. A mechanism-specific follow-up needs a new
prospective protocol; this plan allocates no altered precision/optimizer/kernel.

## Prior work and scope of explanation

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
explains why mathematically equivalent operations need not be bitwise identical;
[reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) separates
seeding from other nondeterminism. These are general constraints, not evidence
for the cause or size of this particular endpoint difference. H118 tests that
empirically. No new theorem, activation or research-priority claim is made.
