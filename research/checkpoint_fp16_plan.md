# H157: FP16 checkpoint-input storage preflight

H156 passed fresh batch16 long-training qualification: progress. Test a distinct
approximate memory mechanism before another training allocation. Frozen sources
and existing maintained helpers remain unchanged. Prior art is summarized in
[checkpoint_compression_note.md](checkpoint_compression_note.md), especially
[GACT](https://proceedings.mlr.press/v162/liu22v.html). Saved-tensor compression is
not claimed novel. PyTorch's ordinary hooks contract expects restored content;
lossy restoration intentionally gives approximate gradients, not exact autograd.

## Candidate and correctness scope

Only the FP32 input saved by each whole-block non-reentrant checkpoint is cast
to FP16 on GPU, then restored to FP32 for recomputation. Forward activations,
weights, optimizer moments and classifier buffers remain FP32. Match the input
by pointer/shape/stride/offset inside its block's scope; reject unexpected nonempty
saved tensors. Empty checkpoint sentinels pass through unchanged. Store no reference
to the original input in pack/unpack closures. Restore original methods on exit.

Identity packing is a control. With fixed upstream gradient, the compressed
block VJP should match the native block VJP evaluated at the quantized input,
while the actual forward output remains evaluated at the original input. Verify
this on a small CPU sequence, including input and all block parameter gradients.
This is deliberately not finite-difference agreement with the original forward:
the gradient is approximate. No unbiased-gradient assertion is made.

## Frozen budget

Use all six H156 ordinary final step800 model/Adam/sampler states. Batch16,
context512, FP32, TF32 off, four CPU threads, ordinary attention, default cuBLAS
workspace. Sample the next source-sampler batch identically in every arm. No Adam
updates. Load the existing optimizer moments so the measured live state is realistic.

Six backward probes per fixture, in order: native0; buffered loss/all inputs on
GPU; buffered loss/identity input hooks; buffered loss/four-block CPU offload;
buffered loss/eight FP16 inputs on GPU; native1. Native repeats measure local
nondeterministic noise. 36 GPU backwards, zero optimizer updates. One additional
FP16 diagnostic forward per fixture saves all eight original inputs to CPU,
outside memory profiling. Capture no full inputs during the six memory probes.
The CPU local-VJP check uses two backwards; a cast overflow example is descriptive.
Expected output below 3 GiB, require at least 6 GiB free.

Save every gradient, scalar loss, batch hash, unchanged model/moment hashes,
checkpoint pack/unpack counts, source state hashes and allocation/host peaks.
Reset peak counters after construction but also retain the construction peak;
report their maximum plus any diagnostic/serialization peak for each clean probe.
This covers model/moments/data/forward/backward/gradient-copy phases, **not an Adam
update or isolated inference**. No single cold backward timing is treated as speed.
Zero allocated/reserved GPU boundaries separate probes and diagnostic forwards.

## Prospective gates

For native repeat, buffered, identity and offload controls vs native0:
global relative gradient L2 <=1e-5, max tensor relative L2 <=1e-4, relative scalar
loss error <=1e-6. Identity and FP16 hooks must save/unpack exactly eight inputs;
state dict and optimizer moments stay unchanged. No nonfinite gradients or saved
FP16 inputs. Diagnostic input rounding error is reported by block and verified
independently, including maximum magnitude, zeros/underflow and overflow.

The FP16 path is judged under **new approximate-gradient guardrails**:
global relative gradient L2 <=0.002, max tensor <=0.02, global cosine >=0.99999,
loss error <=1e-6. These do not replace H156's exact-path limits or establish
long-run quality. Every fixture must pass. FP16 peak allocation must be <=1.02
of four-block offload and <=0.90 of ordinary native0; pinned-host peak must be zero.
A pass earns a warmed, temporally paired complete-update timing test, then quality
validation only if resource/speed comparisons survive. It is not a training win.

Before GPU probes, an explicit large-value FP16 cast must demonstrate overflow
is possible. The prototype is only evaluated on diagnosed finite-range inputs;
full forward/backward finiteness checks abort unexpected failures. It is not a
production-safe arbitrary-range compression API.

## Audit

Independently recompute all gradient errors/cosines and per-tensor comparisons
using NumPy FP64. Recompute every diagnostic FP16 cast and error from full saved
CPU inputs using NumPy. Verify source artifacts, pack counts, matching batches,
unchanged states and all clean boundaries. The full numerical control is measured
native FP32 autograd twice; no full-model CPU FP64 equivalence is claimed.
Keep failed runs and original sources. Any recovery requires a documented,
prospective change, never silently replacing results. The broad goal stays open.
