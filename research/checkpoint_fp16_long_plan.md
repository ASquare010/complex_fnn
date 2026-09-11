# H160: FP16 checkpoint storage from initialization

Previous goal turn: progress. H159 passed all six paired short-run timing,
resource, stability, gradient and short-NLL gates. This study tests whether the
approximate backward changes fresh learning. Preserve all earlier results.

Use H156's six H117 step-zero model states with empty Adam state and independently
regenerable initialization. Two corpora, seeds101/113/127, fresh ordinary native
controls and buffered loss/eight FP16 GPU checkpoint inputs. Alternate which arm
runs first by fixture. Twelve isolated processes, one GPU model at a time, each
800 continuous updates; no completed reruns. Reuse the long-training loop unchanged:
batch16, context512, validation batch16, FP32/TF32 off, four CPU threads, ordinary
attention,8.125MiB cuBLAS workspace, LR0.0006, AdamW0.9/0.95, original decay,
clipping1. Save model/Adam/sampler at200/400/800 and initial full gradients.

Before training, run native and FP16 initial backwards on all six fixtures, with
sequential zero GPU boundaries. Save all12 gradient arrays and independently
calculate NumPy FP64 errors. FP16 global relative L2<=.002, max tensor<=.02,
cosine>=.99999, relative loss error<=1e-6, all gradients finite, exactly8 input
saves/unpacks. Failure stops long training. All source/batch hashes must match.
This is an initial-scale check, not proof of all later approximate gradients.

Long-run gates remain H156's per-seed limits: peak whole-job allocated GPU memory
ratio<=.90, median complete CUDA and wall update ratios<=1.15, final validation
NLL ratio<=1.01, pinned-host allocated peak<=128MiB, both runs' three260-update
mean wall-time blocks (after20 warmups) max/min<=1.15, telemetry coverage. No
seed removal, timing correction, best-run selection or relaxed gates. Report
means/medians/sample variances, intermediate validation curves and all failures.

Independent audit regenerates all initial states bitwise, native scores at0,
200/400/800, all9,600 training batches and sampler/checkpoint/moment hashes,
finite states and step counters. Replay12 native initial backwards independently.
For native controls retain exact replay limits1e-5 global/1e-4 tensor/1e-6 loss.
For FP16 keep the old exact replay result as a diagnostic and apply separately
recorded approximate caps. Capture NumPy gradient error with native as denominator
and cosine while leaving the old scoring/state audit unchanged. No approximate
pass is presented as exact-gradient equivalence. The unchanged H157 codec must
save/unpack6,408 block inputs per run (801 backwards x8).

Budget:9,600 optimizer updates,78,643,200 training targets;9,612 training/probe
backwards plus12 initial preflight and12 native audit backwards =9,636 total.
48 study and42 independent native validation scores;60 tensor artifacts including
12 preflight gradients. Require8GiB free; expected artifacts below6GiB. Passive
200ms telemetry, no clock/power changes. All sources and gates frozen before GPU.

Pass qualifies only this model, workloads, precision and800-update duration.
It does not prove terminal convergence, multiple unrelated task domains, fewer
parameters, novel activation geometry or SOTA. Compression is established prior
art; H157/H158/H159 establish the mechanism and narrower prerequisites. The broad
research goal remains open even if this long-run qualification passes.
