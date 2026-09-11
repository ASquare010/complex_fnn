# H120 — optimizer implementation and complete-update memory

## Hypothesis and scope

H119 narrows the first-step sensitivity question; its CPU clipping audit fails
and the old language-quality gate stays failed. Return to measured VRAM:
can a native AdamW implementation with fewer temporary tensors reduce the
complete-job allocation peak by at least 10%, without materially changing
short-continuation quality or practical update time?

Compare default AdamW (`foreach=None, fused=None`), per-tensor AdamW
(`foreach=False, fused=False`) and native fused AdamW
(`foreach=None, fused=True`). These implementations already exist. This is a
resource/implementation experiment, not a new optimizer or architecture.
[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
documents that foreach can need roughly a parameter-sized extra tensor list.
That does not imply the optimizer sets this model's whole-job peak.

Let P be FP32 parameter storage, M two-moment storage, D cached data and B all
other live storage at the optimizer phase. Default optimizer temporary storage
can be O(P); per-tensor temporaries are bounded by the largest processed tensor
and fused kernels may avoid that list. Persistent FP32 moments still cost 2P.
The relevant peak is max(forward, backward, clipping, optimizer, evaluation,
construction, diagnostics, serialization), not their sum. A smaller optimizer
peak is useless for this objective if another phase remains the maximum.

## Fixed design and budget

Two H117 trained checkpoints: WikiText-2 and TinyStories seed101, each from its
FP32 chunked run at update800. Within each corpus all arms start from that same
hashed model, Adam moments and sampler state. Use both FP32 native and FP32
chunked classifier loss. Four corpus/loss fixtures × three optimizer modes ×
30 continuation updates = **12 cases / 360 training updates / 1,474,560 targets**.
The first 10 updates are warmup, the final 20 timed. No extra first-gradient
backward or CPU qualification is allocated; the first actual update supplies
the saved raw and clipped gradients. All arms use one common profiling loop.

Retain width384, FFN456, eight layers, six heads, vocabulary4096, B8/T512,
FP32 model/gradients/moments, default SDPA, whole-block checkpointing, fixed
LR0.0006, betas(0.9,0.95), epsilon1e-8, matrix decay0.1/no norm-weight decay,
norm clip1, four CPU threads and TF32 off in the existing UV-managed runtime.
Only optimizer implementation flags differ. Apply flags to a local copy of
the loaded optimizer groups before loading state, so native fused step-counter
placement follows PyTorch's own loader. Original checkpoint bytes stay intact.
Rotate mode order across the four fixtures, avoiding always running one first.

Use the same full streamed BF16 evaluator before and after each continuation
(24 scores), then independently score all 12 final states using native BF16
evaluation (12 audit scores). Total 36 scores. Audit has zero optimizer steps
and zero backward passes. No long fresh training or extra seed is allocated.
These correlated one-seed continuations do not establish long-run quality.

## Measurement

Record construction, full validation, every batch/zeroing, forward, backward,
clipping, optimizer, first-gradient/state serialization, final diagnostics and
final-state serialization before any peak reset. Keep allocated and reserved
bytes separate. Account for all untimed phases and all warmup peaks. Record
zero tensor-allocation/reservation boundaries before/between/after cases.

At phase boundaries count unique CUDA storage owned by parameters, buffers,
gradients, optimizer moments/counters, dataset cache, current batch and loss.
Report allocator bytes minus those known storage bytes as unattributed live
storage, which can include workspaces, retained activations and allocator
rounding. Do not claim that boundary inventories describe transient ownership
inside a phase. Persistent moment bytes must equal 2×FP32 parameter bytes.

Time each numerical phase with CUDA events. Report the sum of numerical-phase
event times and complete instrumented wall time separately, plus mean, median,
sample variance and split-half timing stability. Synchronization/inventory
instrumentation changes scheduling: this is a profiling screen. It cannot
replace an uninstrumented production benchmark.

Save every first raw/clipped gradient map, first updated model/optimizer state,
and final model/optimizer/sampler state: **36 new tensor artifacts**. Keep all
360 batch/target hashes, losses, norms, phase records and group metadata.
Hash inputs, scientific source, installed optimizer code, prior evidence and
dataset files before execution. Preserve every failure; no silent retry.

## Numerical audit and fixed decisions

For the first step of every case, independently use NumPy FP64 with the saved
incoming moments and clipped gradient:

    m' = beta1*m + (1-beta1)*g
    v' = beta2*v + (1-beta2)*g^2
    theta' = (1-lr*decay)*theta
             - lr*(m'/(1-beta1^801))/(sqrt(v'/(1-beta2^801))+epsilon).

Verify first step counters801 and final counters830. Require parameter global
symmetric relative L2 <=1e-6 and max absolute error <=2e-7, each moment global
relative L2 <=1e-6, and clipping relative L2 <=1e-6 against FP64 global norm
clipping. These do not promise bitwise equality. Saved raw gradients of each
alternative must match its default fixture within global relative L2 1e-5 and
maximum tensor relative L2 1e-4. Verify all 360 sampled batches independently,
finite states and all checkpoint/data hashes. Native versus streamed final NLL
relative error must be <=1e-6 with identical target counts/order.

For each alternative, separately in each of four fixtures, require all:

- all its/default numerical checks pass;
- complete-job allocated peak <=90% of default;
- final NLL <=1.01×default;
- warmed median event-sum AND instrumented wall time <=1.10×default;
- split-half median timing ratio <=1.15 in both timing measures for both arms.

All four fixtures must pass to earn a separate uninstrumented resource screen.
A passing average cannot override a failed fixture. An optimizer-only saving
without a full-job saving is **ELIMINATED for this memory gate**, even if faster.
Preserve useful timing or numerical findings separately. No model/optimizer
default is changed automatically; no parameter, new-layer or breakthrough
claim follows. The broad VRAM/quality and architectural objectives stay open.
