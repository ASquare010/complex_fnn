# Coupled curves version 2: fuse branches and coefficient reductions

Registered after all 42 version-1 profiles completed and before implementation
or measurement. Version 1 passed numerical checks and saved memory but missed
the slowdown limit on both corpora; no language loss was measured. Preserve its
[failed result](channel_curve_result.md) and frozen source.

Keep every equation, coefficient, shift, matrix dimension, initialization,
Gaussian calibration, backbone, control, budget and acceptance gate in the
[original hypothesis](channel_curve_hypothesis.md). This is an execution revision,
not a new representational or originality claim.

Replace separate branch kernels, tensor rolls and token-sized coefficient-gradient
arrays with direct feature indexing and fused CUDA operations:

* One forward kernel sums all three branches and applies the fixed mean/scale.
* One input-adjoint kernel combines local and inverse-shifted neighbor paths
  directly, reconstructing scalar branches. No token-sized derivative banks.
* One coefficient-adjoint kernel produces deterministic partial reductions over
  fixed 32-token/32-channel tiles, followed by a reduction of tile partials.
  Avoid atomic accumulation; report different floating-point reduction order.

Use the NVRTC library bundled with the existing Torch installation and the CUDA
Driver API. No external compilation toolchain or added GPU package is required.
Follow [NVIDIA's runtime compilation API](https://docs.nvidia.com/cuda/nvrtc/index.html).
Record cold compilation separately. This may reduce launch/allocation overhead,
but input-adjoint reconstruction evaluates additional scalar nonlinearities;
state that extra work. Tile reductions, register use, indexing or launch costs
may instead make it slower. FP32 arithmetic for BF16 projections, with one final
BF16 input-gradient cast; FP32 master coefficient gradients. No approximate or
straight-through gradients, no higher-order-gradient claim.

Independently validate CPU FP64 equations/finite differences, CUDA FP32/BF16
forward, z and all coefficient gradients, self-gate accumulation and incomplete
tiles. Test width 512 plus non-tile-multiple shapes. Verify BF16 rounding, finite
outputs, complete-model native references, causality/locality/counts/save-load
and exact CPU recovery. Record tolerances and actual errors. Shapes/dtypes must
be explicitly checked before low-level launches; use Torch-owned tensors and
the calling Torch CUDA stream, with no private context or unmanaged allocations.

Then repeat the same 42-profile hardware screen in a fresh identity, three
alternating rounds per corpus, 20 warmup/100 measured updates. Require the fixed
candidate to pass every full-control resource comparison and <6 GiB reserved.
Retire version 2 before language if it fails. Do not substitute a faster control
or relax the slowdown limit. If it passes, promote the complete Transformer,
verify production equivalence/shared-Trainer CPU recovery, and run the original
nine-variant/corpus 2,000-update language screen and registered removals with a
fresh frozen source. Qualification still requires both dense quality margins,
mechanism controls, equal tuning, fresh seeds and independent confirmation.
