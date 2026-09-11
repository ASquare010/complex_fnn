# H110: does the qualified memory saving survive 800 language-training updates?

Previous goal turn: progress. H109 provided audited negative evidence about
derivative-aware initialization followed by value learning. That recipe stays
closed. Return to the strongest retained actual-VRAM lead: H102's narrow-GELU
loss chunking saved 21-32% allocation with close NLL, but only after 12 updates.
The full VRAM/quality goal and separate >=70%-parameter architectural target are
unchanged. This extension can validate an execution result, not a new activation.

## Hypothesis and established mechanism

At identical model, initialization, optimizer and training tokens, checkpointed
classifier-loss chunks preserve useful language learning while reducing peak
training tensor allocation over 800 updates, across three seeds and two shapes.
Floating-point gradient accumulation can alter trajectories despite exact
real-arithmetic loss identities; longer learning may falsify H102's short result.

Use the unchanged H101/H102 execution adapter and maintained helper. For N token
rows and vocabulary V, chunking by C bounds each logit workspace by O(CV), while
checkpointing recomputes rather than retaining it. Model weights, gradients,
Adam states, attention and O(Nd) representations remain. Summed chunk CE divided
by the total number of valid targets is the same mathematical objective. This
does not guarantee floating-point trajectory equality or total memory savings.

[PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html) and
[Cut Your Losses](https://arxiv.org/abs/2411.09009) establish recomputation and
classifier-memory reduction precedents. This is native PyTorch checkpointed
token chunking, not CCE or a new kernel, and no gradient terms are dropped.

## Fixed comparisons

Narrow GELU only: d384, eight layers, six heads, h456, V4096. Counts stay exactly
9,099,648 total / 2,801,664 FFN parameters for both executions. Policies:

1. Whole-block checkpointing, unchunked CE (memory-aware reference).
2. The same whole-block checkpointing plus loss chunks of 512 token rows.

Shapes B16/T128 and B8/T512; seeds17/29/43. Twelve sequential fresh worker
processes, 800 updates each, 9,600 updates total. Rotate policy order by seed.
No full GELU/SwiGLU or FFN-chunk retry: those fixed H101/H102 branches remain
unqualified. No new architecture, width, chunk-size or optimizer sweep.

Each fresh model uses H102's exact named-tensor initialization and uncalibrated
width defaults, constant LR0.0006, matrix decay0.1, AdamW betas(0.9,0.95),
epsilon1e-8 and global clip1. This directly extends the short constant-rate
recipe, not the separate calibrated narrow architectural baseline. Both policies
receive identical treatment. No schedule, width initialization or gradient
precision changes are introduced. BF16 autocast, FP32 weights/Adam, TF32 off,
four CPU threads, UV-managed Python3.12.9 and recorded malloc allocator settings.

The unchanged hashed WikiText-2 cache stays on CUDA and counts toward memory.
Training sampler seed is model_seed+10000, with identical stream and initial
state hashes checked per pair. No test split access. Validation uses the same
unchunked forward/CE for both policies, irrespective of training execution.
This is heavily reused development validation, not an independent broad-data
confirmation or a convergence-optimal recipe.

## Qualification, measurements and preserved endpoints

Before long training, verify tiny CPU-double and CUDA-BF16 loss/parameter-gradient
agreement for both shapes (including an uneven final chunk), exact parameter
identities/state keys and default execution restoration. CPU gradient relative
error<=1e-9, CUDA<=2%, relative loss error<=0.1%; all finite. Existing 14 helper
tests and H102 full-model qualification remain prior evidence, not new test counts.

Record loss, preclip gradient norm, forward/backward/optimizer/update timing and
allocated/reserved training peak at every update. The first 20 updates are timing
warmup. Report medians, means, sample variances, early/late time windows and
throughput. Synchronize timing; exclude sampler, validation, diagnostics and
checkpoint I/O from measured update time, but include them in wall time.

At steps0/100/200/400/800, evaluate 16 fixed unchunked validation batches and
record layer activation/gradient summaries outside the timed training interval.
Preserve raw checkpoints at100/400/800 for independent rescoring; preserve final
model, optimizer and sampler state. At800 additionally score the complete valid
split with the same context/batch convention, excluding only the incomplete last
context window. Compare policies within each shape, not NLL across tokenizations
or context lengths. Record exact validation target counts/order hashes.

Per-update peak resets occur after sampling/zero-grad and before model execution;
the full persistent CUDA dataset/model/Adam allocation still counts. Final results
report the maximum of all training steps, validation separately, and the maximum
for the whole measured job. This exposes whether unchunked evaluation erases a
training-only saving. Keep parameter, gradient and optimizer bytes distinct.
No inference benefit or parameter reduction from chunking is claimed.

## Frozen acceptance and stop rules

For every seed at a shape, candidate must use >=15% less **training allocation**
and no more than25% greater median update time than block checkpointing. Final
complete-validation NLL must differ by at most1% relative (absolute difference),
and the fixed validation subset at100/200/400/800 must also stay within1%.
Report seed means/medians/sample variances and every individual gate. No selecting
the best checkpoint, seed or timing interval. Whole-job allocation must be shown
separately; a training-only qualification is not a whole-job VRAM claim.

Stop allocating further seeds/shapes if a completed pair violates any NLL fidelity
gate or produces nonfinite training: preserve both endpoints and independently
rescore before deciding whether the recipe is closed. If only timing/memory
fails, finish the already fixed grid to report device variation. A source/runtime
failure pauses execution for explicit diagnosis; never silently rerun a trial.

If all gates pass, retain this as an 800-update, three-seed execution result at
that shape. Longer convergence, larger scale, a broader corpus, total-job memory
and the architectural gold target remain separate requirements. If fidelity
fails, close this duration recipe; do not alter precision, chunk size, rate,
schedule or tolerances to rescue it. No universal numeric-equivalence theorem.

Independent audit must reconstruct all initial states and sampling streams,
load all saved endpoints and rescore unchunked subset/full validation, check
finite optimizer state/step counts, aggregate all recorded resources and recompute
the frozen gates. Preserve sources, plans, successful trials and failures.
