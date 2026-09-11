# H145: separate gate observations fail the eager VRAM screen

**Reject this resource recipe.** Both output-modulation candidates have about
24.8% fewer parameters and25% fewer matrix MACs than the full controls. Neither
achieves the required10% allocated-GPU-memory saving at either batch size.
At batch2048, both use more GPU memory than full GELU. No learning-quality
experiment is promoted from this screen.

## Hypothesis and equations

For dimension384, shared modulation uses z=GELU(Ux+a) with192 features and
F(x)=Bz+c+x*(Cz+e). Split modulation forms288 features in one projection,
feeds the first192 to B and the remaining96 to C. The split candidate and a
288-wide GELU with a learned diagonal input path have exactly222,240 parameters;
the shared candidate has222,144. All four budget arms use221,184 matrix MACs,
versus294,912 for full GELU384 and SwiGLU256. These MAC counts exclude elementwise
operations; lower matrix arithmetic does not guarantee lower runtime or memory.

The split design gives base and gate different observations, but their combined
linear observation still has rank at most288. [H144's bound](modulated_capacity_theory.md)
still applies to that combined observation; removing one skew-target obstruction
is not a universal approximation or learning guarantee. Gate readouts start at
zero; the active-gate FP64 input/all-parameter finite-difference tests pass.
Shared/split initial base weights are identical and outputs agree numerically.
The prospective plan's word "exactly" describes the algebraic function; the
preflight uses floating-point tolerances for outputs from different GEMM shapes.

## Measurements

Means across three seeds. Each timing is a run's median of eight complete
forward/backward/clipping/AdamW updates after four warmups. Inputs require
upstream gradients. Gradient clearing is outside timers. Peak includes model,
resident batch, gradients, optimizer, temporaries and the final diagnostic score.
One model is resident at a time. Reserved allocation is separately retained.

| Batch | Arm | Parameters | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|---:|
| 128 | full_gelu | 295,680 | 22.640 | 2.067 | 2.109 |
| 128 | full_swiglu | 295,808 | 22.643 | 2.559 | 2.603 |
| 128 | budget_gelu | 221,856 | 21.234 | 2.046 | 2.087 |
| 128 | static_diagonal | 222,240 | 21.241 | 2.338 | 2.380 |
| 128 | shared_gate | 222,144 | 21.239 | 2.530 | 2.572 |
| 128 | split_gate | 222,240 | 21.241 | 2.804 | 2.848 |
| 2048 | full_gelu | 295,680 | 47.199 | 2.075 | 2.117 |
| 2048 | full_swiglu | 295,808 | 50.013 | 2.479 | 2.519 |
| 2048 | budget_gelu | 221,856 | 45.340 | 2.079 | 2.119 |
| 2048 | static_diagonal | 222,240 | 48.672 | 2.465 | 2.508 |
| 2048 | shared_gate | 222,144 | 47.577 | 2.579 | 2.622 |
| 2048 | split_gate | 222,240 | 48.471 | 2.760 | 2.799 |

At batch128, both dynamic candidates save only about6.2% versus full GELU.
At batch2048, shared uses about0.8% more and split about2.7% more than full GELU.
The budget GELU control is also below the required10% saving. Parameter savings
are insufficient here because actual training allocation includes much more than
weights. Modulation adds full-width outputs, products and backward dependencies;
this screen measures their aggregate cost, not an allocator-attributed causal
breakdown. No activation-level attribution should be inferred without a trace.

## Every prospective candidate comparison

Ratios are candidate/control; smaller is better. Required: parameter ratio<=.80,
peak<=.90, CUDA/wall<=1.15, candidate half-window stability<=1.15. Controls' stability
is also retained, and all raw timings remain available. No thresholds were tuned.

| Batch | Seed | Candidate | Control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---|---:|---:|---:|---|
| 128 | 53 | shared_gate | full_gelu | 0.9381 | 1.210 | 1.206 | peak, timing |
| 128 | 53 | shared_gate | full_swiglu | 0.9380 | 0.936 | 0.937 | peak |
| 128 | 53 | split_gate | full_gelu | 0.9382 | 1.333 | 1.327 | peak, timing |
| 128 | 53 | split_gate | full_swiglu | 0.9381 | 1.031 | 1.031 | peak |
| 128 | 67 | shared_gate | full_gelu | 0.9381 | 1.207 | 1.203 | peak, timing |
| 128 | 67 | shared_gate | full_swiglu | 0.9380 | 1.304 | 1.304 | peak, timing |
| 128 | 67 | split_gate | full_gelu | 0.9382 | 1.352 | 1.348 | peak, timing |
| 128 | 67 | split_gate | full_swiglu | 0.9381 | 1.460 | 1.461 | peak, timing |
| 128 | 79 | shared_gate | full_gelu | 0.9381 | 1.256 | 1.250 | peak, timing |
| 128 | 79 | shared_gate | full_swiglu | 0.9380 | 0.835 | 0.833 | peak |
| 128 | 79 | split_gate | full_gelu | 0.9382 | 1.385 | 1.376 | peak, timing |
| 128 | 79 | split_gate | full_swiglu | 0.9381 | 0.920 | 0.917 | peak |
| 2048 | 53 | shared_gate | full_gelu | 1.0080 | 1.325 | 1.320 | peak, timing |
| 2048 | 53 | shared_gate | full_swiglu | 0.9513 | 1.253 | 1.253 | peak, timing |
| 2048 | 53 | split_gate | full_gelu | 1.0269 | 1.393 | 1.385 | peak, timing |
| 2048 | 53 | split_gate | full_swiglu | 0.9692 | 1.317 | 1.315 | peak, timing |
| 2048 | 67 | shared_gate | full_gelu | 1.0080 | 1.104 | 1.100 | peak, stability |
| 2048 | 67 | shared_gate | full_swiglu | 0.9513 | 0.910 | 0.910 | peak, stability |
| 2048 | 67 | split_gate | full_gelu | 1.0269 | 1.266 | 1.258 | peak, timing, stability |
| 2048 | 67 | split_gate | full_swiglu | 0.9692 | 1.043 | 1.041 | peak, stability |
| 2048 | 79 | split_gate | full_gelu | 1.0269 | 1.333 | 1.323 | peak, timing, stability |
| 2048 | 79 | split_gate | full_swiglu | 0.9692 | 1.012 | 1.009 | peak, stability |
| 2048 | 79 | shared_gate | full_gelu | 1.0080 | 1.302 | 1.299 | peak, timing |
| 2048 | 79 | shared_gate | full_swiglu | 0.9513 | 0.989 | 0.990 | peak |

The short timing windows are a resource screen, not sustained-throughput proof.
Cases exceeding the15% half-window stability threshold: [(6, 'full_swiglu'), (13, 'static_diagonal'), (24, 'shared_gate'), (25, 'split_gate'), (30, 'split_gate'), (34, 'static_diagonal')].
Timing noise does not rescue either candidate's independently failed memory gate.
Seed-level means, medians and sample variances for peak, CUDA/wall timing and
final diagnostic loss are in [audit.json](../results/split_modulation_v1/audit.json).

## Reproducibility and audit

Frozen recipe: Gaussian input, independent orthogonal linear target, one fixed
batch per seed/size; seeds53/67/79; twelve updates; AdamW LR.003, betas(.9,.95),
zero decay, clip1; FP32, TF32off, four CPU threads, ordinary eager execution,
default8.125MiB cuBLAS workspace. Six-arm execution order rotates by fixture.
NVIDIA GeForce RTX4070 Laptop GPU, driver610.62 (queried after training);
PyTorch 2.14.0+cu132, CUDA build 13.2. No clock/power changes.
Telemetry was not collected, so temperature or clock causality is unassessed.

All36 saved model/optimizer states are finite, optimizer counters equal12,
exact parameter/optimizer byte counts verify, and six datasets regenerate bitwise.
Independent CPU FP64 scoring uses explicit matrices, erf/sigmoid formulas and
batch257, without the model's forward method. All36 final scores agree within
1e-5 relative tolerance; maximum observed relative error is 1.253e-07.
This verifies saved-state scoring, not an independent replay of Adam updates.
First/final input-gradient and post-clipping parameter-gradient norms are finite;
this twelve-step check does not establish long-depth gradient stability.

There were432 optimizer updates/backwards,37 clean GPU allocator boundaries,
36 training-process final scores and36 independent CPU audit scores. The audit
initialized no CUDA and performed no backward calls. No hidden repeats or tuning.
All frozen source and prior receipt hashes verify; maintained code/defaults are
unchanged. Initial/final losses are fixed-batch diagnostics only: no validation,
generalization, expressivity gain or quality retention is claimed.

## Decision

Close these eager shared/split-gate settings before costly fitting. A future
resource hypothesis must change the execution or saved-tensor footprint under
a separately frozen protocol; it cannot silently reuse these timings as evidence
for a fused or recomputed implementation. Existing H141/H142 memory-helper gains
remain scoped evidence; they do not make this architecture successful.

Modulation and feature splitting have prior art, including
[FiLM](https://arxiv.org/abs/1709.07871). This screen claims no novel activation
or breakthrough. The broader VRAM-and-quality research goal remains open.

[Prospective plan](split_modulation_plan.md),
[prototype](../results/split_modulation_v1/model.py),
[raw measurements](../results/split_modulation_v1/result.json),
[evidence receipt](../results/split_modulation_v1/receipt.json).
