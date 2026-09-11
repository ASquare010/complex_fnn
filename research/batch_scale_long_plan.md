# H156: fresh long training at the qualified batch-16 workload

H155's verified augmentation obstruction was progress, but it provides no actual
VRAM/quality improvement. Return to the strongest measured practical route:
H142's maintained buffer loss plus offload of four checkpoint inputs. Its short
batch-16 run saved about 24% allocation with about 5% update-time overhead.
Fresh long batch-16 convergence is missing. H138's earlier batch-8 failure remains.

## Fixed comparison

Use all six H117 step-zero fixtures, with empty Adam moments: WikiText-2 and
TinyStories, seeds 101/113/127. Model, initialization and tokenizer/data hashes
remain unchanged: 9,099,648 parameters, d384, eight blocks, context512, whole-block
checkpointing. Start both arms from each exact initial state. Increase training
batch from 8 to 16. Validation also uses batch16 in BOTH arms, preserving the
unchanged long loop and native audit; H142 used validation batch8, so no
cross-study numerical/timing ratio will be presented as paired evidence.

Ordinary: native training loss with checkpoint inputs on GPU. Candidate: maintained
src/core/training_memory.py buffer_model_loss and offload_checkpoint_inputs(...,4),
covering blocks 4 through 7. No maintained source is edited. The candidate was
selected before this experiment by H142, not by these fresh results.

Reuse H138's exact 800-update loop and independent audit. FP32, disabled TF32,
ordinary nondeterministic attention, default 8.125 MiB cuBLAS workspace, four CPU
threads, UV-managed Python, AdamW LR0.0006/betas(.9,.95), original parameter-group
decay, global clip1. No clock/power changes, extra tuning or retry of a weak seed.
Alternate ordinary/candidate then candidate/ordinary by fixture index; freeze all
12 isolated runs before execution. Each completes a continuous 800-step trajectory.
Run only one training model at a time and stop the driver on process failure.

## Budget and evidence

12 x 800 = 9,600 updates; 78,643,200 targets. Twelve initial gradient probes plus
12 independent native gradient replays: 9,624 backward passes total. Save every
history and initial gradients plus steps200/400/800 model/Adam/sampler states.
48 study validation scores, 42 native audit scores, 48 tensor artifacts; expected
artifact storage below 8 GiB. Require at least 12 GiB free before launch.
Capture passive 200ms GPU telemetry; report missing sensors as missing.

All six seed/corpus pairs must independently pass: whole-job allocated memory
ratio <=0.90, complete update CUDA and wall median ratios <=1.15, final validation
NLL ratio <=1.01, candidate pinned-host peak <=128 MiB. Each run's three 260-update
wall-mean blocks after20 warmups must have max/min <=1.15. No discarded slow runs,
clock-normalized timings, relaxed gates or aggregate rescue of a failed seed.

Reuse exact initialization regeneration, every-batch/sampler verification,
checkpoint/moment finite checks and native scoring. Initial gradient tolerances
remain global relative L2 <=1e-5 and maximum tensor <=1e-4; loss/score <=1e-6.
Ordinary attention is nondeterministic; no bitwise trajectory claim. Hash every
source/input/library and maintained file. The independent audit uses native loss
and native full classifier evaluation, not candidate buffer/offload operations.

The training timer includes forward/backward/clipping/Adam, transfers and
recomputation, excluding batch sampling, gradient clearing, logging, scoring and
serialization. All those phases remain in whole-job CUDA peak accounting; report
whole-case wall time separately. Driver/context/other processes are excluded from
allocated memory. Pinned-host memory is separate and is not total CPU RSS. This is
a training-memory experiment, not an inference-speed or inference-memory result.

A pass establishes fresh long-training qualification of this existing memory
optimization at batch16 across the two small corpora. It does not establish a
new activation, fewer parameters, SOTA or completion of the broader research goal.
On failure preserve every observation and report exactly which criterion failed.
On a process error inspect the live handle and artifacts before any bounded
recovery; never restart completed or confirmed-live work.
