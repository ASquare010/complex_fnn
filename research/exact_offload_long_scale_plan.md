# H163: sustained qualification of exact memory helpers at 22.2M parameters

H162 passed host/GPU/short-training checks; its30-update result cannot establish
sustained quality. H163 runs six fresh800-update cases on WikiText2, seeds401,
409,419, order ordinary/helper, helper/ordinary, ordinary/helper. Budget4800
optimizer updates and six initial gradient backwards; no new datasets or search.
Use UV's existing Python and the RTX4070. No result-dependent early selection,
replacement of completed cases, or gate changes. Stop on execution failure.

Keep H162's architecture, initialization, optimizer, batches and FP32 policy:
22,163,968 parameters, width512/hidden608,12 layers/eight heads, batch16/context512,
GELU, ordinary attention, TF32off, four CPU threads, whole-block recomputation,
AdamW lr0.0006/beta(.9,.95)/eps1e-8, matrix decay.1, vector decay0, clip norm1.
Native CE versus unchanged buffered CE and offload of the final four block inputs.
Both arms use identical initialization and batch sequences. No LR schedule.

Retain explicit FP32 chunked full validation initially and at200/400/800. Save
initial gradients, model weights at200/400/800 and a rolling complete resume
checkpoint (model/Adam/sampler/config/step) at those endpoints. The rolling
checkpoint supersedes its earlier state; immutable weight snapshots and metrics
remain. Expected output below6GiB. Do not keep redundant resume copies.

Measure whole-job peak allocated/reserved GPU through construction, full GPU data,
initial probe, all validation, optimizer steps and checkpoint serialization. Record
per-update full CUDA/forward/backward/optimizer and wall times, loss and gradient
norm. Windows counters from H162 capture current/peak working set and private
commit before imports, before training, and after worker completion. OS lifetime
peaks include CPU serialization and imports. Record passive200msGPU telemetry.
The proven cleanup runs only after measured work; GPU boundary must return to0.

Every seed must pass GPU allocation ratio<=.90, median complete CUDA/wall ratios
<=1.15, final NLL ratio<=1.01, pinned allocation<=128MiB; incremental host peak
working set and private commit each<=256MiB, both arms' host peaks<=8GiB. Discard
20 timing warmups, then compare three260-update wall-time block means: max/min
<=1.15. Also require max timed wall/median<=5. These retain H162's numerical
limits, extending its stability windows prospectively for800 updates. No clock
normalization, silent interruption trimming or aggregate rescue.

Freeze new sources/plan and all inherited dependency hashes before execution.
Independently regenerate each initial model and sampled GPU batch sequence;
verify full gradient arrays using NumPy FP64 with global relativeL2<=1e-5,
max tensor<=1e-4, relative initial loss<=1e-6. Native full-classifier audit scores
all saved model endpoints (relative NLL<=1e-6). Inspect rolling optimizer moments
for finiteness and800step counters, compare its model/sampler hashes with saved
endpoints and regenerated sampler state. This does not independently replay all
4800 optimizer steps. No extra training is performed by the audit.

Report all seeds and mean/median/sample variance, convergence curves, GPU/host
peaks and any failed gates. Pass qualifies only800updates on this one corpus and
model scale; not terminal convergence, unrelated-task quality, activation novelty,
parameter reduction or the broad research goal. Keep all prior failures intact.
