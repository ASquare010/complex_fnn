# H149: fused inverse and local gradients

**REJECT tested fused-reconstruction resource recipe.** The new execution backend fuses reconstruction and local derivative
work. It changes no forward equation or parameterization. The decision is based
on actual training allocation and complete update timing against checkpointed
GELU and SwiGLU. Any resource pass earns a separate quality study, not a claim
that the architecture learns equally well. The broad research goal remains open.

## Implementation and qualification

H148's analytical inverse created many full-sized temporary tensors. A fixed
64-row by32-channel Triton tile computes reconstructed z, z-b, input-side dz,
and partial theta/bias gradient sums together. Native PyTorch sums combine the
partials; native matrix products propagate the state and cotangent. The earliest
layer skips z-b because only its input gradient is needed. Four warps, no autotune,
no floating-operation fusion. Forward and fixed dense orthogonal buffers remain
exactly H148's. There are6,144 trainable scalars and4.5MiB of fixed matrices.

The kernel uses sqrt(B*B+4*abs(y)) with stable root branches, rather than hypot.
Qualification covers signed magnitudes1e-12..1e12 plus zero, not all FP32 values.
Shapes(1,1),(7,33),(128,384),(2048,384) check channel/row tails against independent
CPU FP64 formulas for reconstructed preactivation and all local gradients.
Maximum local relative error is1.211e-07; all relative errors<=5e-5 and
scaled elementwise errors<=1e-4. Complete-stack checks at batches128/2048 compare
against H148's PyTorch reconstructed backward: maximum per-family relative error
5.771e-07<=1e-4 and identical forward outputs. These complement H148's
finite-difference checks; they are not an independent finite-difference test of
the compiled kernel. All checks pass before the resource worker is permitted.

Compilation and correctness work took5.85s in the
check process. Generated kernels are cached in this study's triton_cache directory;
cache artifacts are excluded from evidence hashes. Source/compiler hashes and
Triton3.8.0 are recorded. Cached loading and first-use overhead
in the worker may enter its four discarded warmups; setup is not claimed free.
Only contiguous CUDA FP32 first-order use is qualified; no autocast, higher-order,
compiler integration, arbitrary-magnitude or other-hardware guarantee is made.

## Complete GPU training measurements

Eight layers, width384, rho.25. Seeds223/239/251, batches128/2048. Means across
seeds of allocated peak and each run's median eight updates after four warmups.
Inputs require gradients. Gradient reset is outside timing; forward, backward,
clipping and AdamW are inside. Final no-grad scores enter peak but not timing.
QR setup is outside timing. All matrix buffers, gradients and optimizer states
are counted in the CUDA allocator peak; this does not measure total driver VRAM.

| Batch | Arm | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|
| 128 | full_gelu:eager | 62.118 | 9.273 | 9.318 |
| 128 | full_swiglu:eager | 62.137 | 12.396 | 12.444 |
| 128 | full_gelu:checkpoint | 62.118 | 13.022 | 13.063 |
| 128 | full_swiglu:checkpoint | 62.137 | 18.437 | 18.483 |
| 128 | budget_gelu:checkpoint | 50.868 | 13.021 | 13.063 |
| 128 | learned:checkpoint | 24.199 | 17.380 | 17.421 |
| 128 | learned:reconstruct | 24.270 | 15.198 | 15.240 |
| 128 | learned:fused | 22.720 | 7.127 | 7.170 |
| 128 | fixed:reconstruct | 24.246 | 14.271 | 14.312 |
| 128 | affine:reconstruct | 22.912 | 8.426 | 8.467 |
| 2048 | full_gelu:eager | 136.886 | 8.321 | 8.363 |
| 2048 | full_swiglu:eager | 154.585 | 11.468 | 11.512 |
| 2048 | full_gelu:checkpoint | 91.886 | 12.957 | 13.000 |
| 2048 | full_swiglu:checkpoint | 102.460 | 20.831 | 20.876 |
| 2048 | budget_gelu:checkpoint | 84.120 | 13.175 | 13.218 |
| 2048 | learned:checkpoint | 75.324 | 17.520 | 17.562 |
| 2048 | learned:reconstruct | 76.848 | 14.866 | 14.907 |
| 2048 | learned:fused | 51.345 | 6.130 | 6.170 |
| 2048 | fixed:reconstruct | 76.824 | 14.863 | 14.906 |
| 2048 | affine:reconstruct | 57.351 | 8.473 | 8.515 |

The fixed-shape and affine controls retain their native H148 implementation;
this experiment does not establish fused nonlinearity beats a fused affine control.
Their inclusion tests how much runtime/parameter cost is attributable to shape
rather than treating invertibility itself as proof of useful nonlinear capacity.
Same-model native/fused allocation and timing ratios are in native_ratios.json.

## Every primary gate

Fused/control ratios. Required in every fixture: parameters<=.8, peak<=.9,
CUDA/wall<=1.15, candidate and control half-window timing stability<=1.15.
The parameter ratio is about.0026; fixed matrices remain included in memory.
Both full controls receive checkpointing, and their eager results are retained.

| Batch | Seed | Checkpointed control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---:|---:|---:|---|
| 128 | 223 | full_gelu:checkpoint | 0.3658 | 0.592 | 0.593 | none |
| 128 | 223 | full_swiglu:checkpoint | 0.3656 | 0.408 | 0.410 | none |
| 128 | 239 | full_gelu:checkpoint | 0.3658 | 0.533 | 0.535 | none |
| 128 | 239 | full_swiglu:checkpoint | 0.3656 | 0.384 | 0.385 | none |
| 128 | 251 | full_gelu:checkpoint | 0.3658 | 0.518 | 0.520 | none |
| 128 | 251 | full_swiglu:checkpoint | 0.3656 | 0.367 | 0.369 | none |
| 2048 | 223 | full_gelu:checkpoint | 0.5588 | 0.419 | 0.421 | none |
| 2048 | 223 | full_swiglu:checkpoint | 0.5011 | 0.216 | 0.217 | stability |
| 2048 | 239 | full_gelu:checkpoint | 0.5588 | 0.475 | 0.476 | stability |
| 2048 | 239 | full_swiglu:checkpoint | 0.5011 | 0.331 | 0.333 | stability |
| 2048 | 251 | full_gelu:checkpoint | 0.5588 | 0.527 | 0.529 | none |
| 2048 | 251 | full_swiglu:checkpoint | 0.5011 | 0.364 | 0.366 | none |

There are5 timing-unstable cases across all60 arms/fixtures; exact
identities and half-window ratios are retained in unstable_cases.json. No clocks,
power limits, selected reruns or thresholds changed. This brief screen is not
sustained-throughput or long-run convergence evidence. H148's failure is preserved.

## Audit and decision scope

All60 saved states pass independent CPU FP64 explicit-formula scoring with
batch257; max relative discrepancy1.082e-07,
below1e-5. Six datasets and all fixed matrices regenerate bitwise. Optimizer
counters equal12, exact parameter/buffer/optimizer bytes and timing summaries
verify, all states/moments/diagnostic gradients are finite. The audit uses the
forward equation without either custom backward. It is saved-state validation,
not replay of every optimizer step. Means/medians/sample variances are retained.

Fused versus native reconstructed execution starts identically. After12 updates,
maximum parameter relative difference is1.982e-08;
maximum final-loss relative difference is6.012e-08.
These descriptive checks do not guarantee identical later training trajectories.

There were720 optimizer/backward calls and four extra correctness VJPs, plus
four local-kernel qualification cases. Training has61 clean allocated/reserved
CUDA boundaries; the check process records seven more. Audit performs no GPU,
backward or optimizer work. The worker's AST is H148's with only model/root
replacement and mandatory successful-check guards. Four CPU threads, FP32/TF32off,
default8.125MiB cuBLAS workspace, AdamW LR.003 betas(.9,.95), decay0,clip1.
RTX4070 Laptop GPU; PyTorch2.14.0+cu132, CUDA build13.2.
No telemetry was collected; thermal/clock effects are unassessed.

This uses one Gaussian batch and independent orthogonal target per fixture.
Losses are diagnostic only, with no validation, generalization or nonlinear
capacity result. The strict gate controls further allocation. A pass still needs
strong nonlinear tasks, fixed/affine controls, ordinary activation baselines,
multiple seeds and held-out quality at materially reduced resource cost. A fail
requires a distinct implementation or hypothesis before promotion. Maintained
modules/defaults are unchanged and previous receipt/source hashes verify.

Fusion and activation reconstruction are known techniques, not novelty claims:
[Triton fusion tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html),
[RevNets](https://arxiv.org/abs/1707.04585).

[Plan](fused_reconstruction_plan.md), [kernel](../results/fused_reconstruction_v1/kernel.py),
[raw results](../results/fused_reconstruction_v1/result.json),
[audit](../results/fused_reconstruction_v1/audit.json),
[receipt](../results/fused_reconstruction_v1/receipt.json).
