# H158: Fresh-process host-allocation qualification

The previous goal turn made progress: H157's gradients and GPU allocation passed,
but its zero-host gate failed. The audit and failure remain immutable. Its shared
process retained offload buffers; even native probes recorded five pinned bytes.
This follow-up changes measurement isolation and explicitly changes the host claim
from absolute zero to no additional allocator footprint relative to native.

Run all six H157 fixtures, each with native0, offload4 and fp16 in separate fresh
processes: 18 backwards, zero optimizer updates. Reuse H157 worker.one unchanged,
including model/Adam/data residency, next saved-sampler batch, buffered candidate
loss, and every gradient/state/input/hook check. FP32, TF32 off, four threads,
ordinary attention and default cuBLAS workspace. Alternate arm order by fixture;
no timing claim. Preserve construction and backward GPU peaks. Record host and
GPU boundaries before and after each probe. Require zero GPU allocated/reserved
at both boundaries and zero initial host allocated bytes. If startup itself owns
pinned bytes, retain the failure; do not subtract them post hoc.

Freeze source/plan and prior evidence hashes before execution. Each child exits
before the next starts. No repeated case or selected subset. Require >=3 GiB disk.

Numerical gates vs the new native0 are unchanged: offload4 global relative gradient
L2 <=1e-5, max tensor <=1e-4, loss relative error <=1e-6. FP16 global <=0.002,
max tensor <=0.02, cosine >=0.99999, loss <=1e-6. Every gradient must be finite.
Independently recompute these with NumPy FP64 from the full saved gradients.
Compare batch/state hashes with both new native and original H157. Eight FP16
inputs/unpacks must each store exactly half the original bytes.

GPU allocation gates: fp16/native <=0.90 and fp16/offload4 <=1.02. New host gates:
FP16 allocated peak <= native allocated peak, FP16 active peak <= native active
peak, and FP16 allocated peak <=1% of offload4 allocated peak. Report absolute
bytes, including incidental allocation, not rounded 'zero'. These are prospective
H158 gates; passing does not turn H157 into a pass. Every fixture must pass.

Pass only earns complete-update timing, followed by fresh quality validation if
speed survives. Approximate checkpoint compression is established prior art (GACT,
linked in H157); no activation novelty, parameter reduction, quality improvement,
inference result or breakthrough follows. Use existing repo layout and UV runtime.
