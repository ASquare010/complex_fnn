# H165: BTT contraction-order memory diagnostic

**Draft paused for Windows reinstall (2026-09-14).** Model/check prototypes exist; CPU numerical checks and GPU study have not run. No result is claimed.

H164 made progress: balanced factors improve synthetic fitting over greedy, but
fail wide-model quality and use17.7%more local training VRAM. Do not tune that
failed model or claim a new architecture. H165 tests whether part of the resource
failure comes from execution order, with exactly the same trainable BTT cores.

Rank1 BTT has W[g,i,b,a]=R[g,i,b]*L[b,g,a] after the existing RMS/gain transform.
Direct execution contracts inputs into an N*m1*n0 intermediate then into output.
Materialized execution constructs W (din*dout elements) and computes X@W.
Both maps and their derivatives coincide in real arithmetic; floating-point
association differs. Autograd through W retains gradients to all cores/gains.
W is rebuilt on every forward, never detached, cached across updates or trained
as an extra parameter. This is contraction-order engineering, not novel BTT.
[Established tensorized layers](https://arxiv.org/abs/1509.06569),
[official BTT comparison](btt_official_comparison.md).

Two scales:64->152->64 and512->608->512, rank1 balanced factors. Three seeds
431/443/457. Small scale loads H164's selected teacher-task balanced and wide
states; the larger scale is freshly initialized and has no quality evidence.
Arms: ordinary dense, direct BTT, materialized BTT. Rotate order ABC/BCA/CAB per
seed. Each in a fresh process, FP32/TF32off,4CPUthreads, batch8192 fixedGaussian
input/target seed64000+seed. Same input/targets across all arms. Same BTT initial
state across direct/materialized, no parameter changes. Ordinary is a resource
reference; its function is not assumed equal to the BTT function.

Before GPU runs, require CPU float64 output/input/core/gain VJP equivalence and
gradcheck for materialized projection. Freeze sources/plan/check/source-state
hashes.18cases total. Per case: one initial forward/backward probe saving outputs
and full parameter/input gradients on CPU;12completeAdamW updates (.003,
beta.9/.999,0decay,clip1;same existing core LR scales); one separate forward-only
saved-tensor census.216optimizer updates/234GPUbackwards. Diagnostic census
uses saved_tensors_hooks retaining metadata only and returns each tensor unchanged;
run outside peak/timing measurement. No fitted-quality selection or retries to
replace completed cases. Expected artifacts<300MiB.

Measure whole local job CUDA allocated/reserved peak from construction, resident
batch, initial probe and all12updates; initial CPU snapshots count only their
GPU lifetime. Clear probe references before measured updates. Use9timed updates
after3warmups and record complete CUDA/wall and forward/backward/optimizer time.
Collect passive200ms GPU telemetry. Each child must release all CUDA allocations
and cuBLAS workspaces after all work. Store census saved-tensor shapes, strides,
storagebytes and alias identities. Unique saved storage is not allocator peak:
transient GEMM workspaces/backward allocations are absent from that census.

Independent NumPy FP64 audit compares materialized to direct initial output,
loss,inputgradient and full core/bias/gain gradients: global relativeL2<=1e-5,
maximum parameter-tensor relativeL2<=1e-4, lossrelative<=1e-6. Require matched
input/state hashes, finite states and gradients. GPU execution-order diagnostic
passes at a scale only if every seed passes numerical checks, materialized
allocated peak<=.90direct, complete CUDA/wall medians<=1.05direct, and each timed
run has maximumwall/median<=5 and two timing-blockmeans ratio<=1.15 (4and5updates).
These do not qualify performance against wide dense or rescue H164 learning.
Report materialized/wide resource ratios without promotion claims. All seeds,
mean/median/sample variance and failures remain visible. Audit performs no updates.

Memory intuition: one direct intermediate costs4*N*m1*n0 bytes versus a dense
weight4*din*dout bytes; autograd retention, core normalization, layouts and CUDA
workspaces determine the measured total. Do not infer exact peak savings from
that single-term comparison or claim the direct intermediate causes all overhead.
