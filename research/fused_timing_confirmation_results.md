# H150: longer, mirrored-order resource confirmation

**FAIL mirrored-order resource confirmation.** This is an independent timing protocol using H149's fixed kernel
and saved states. H149's rejected timing screen is preserved. The confirmation
uses longer windows, mirrored execution order and an additional across-repeat
drift gate. It changes no architecture, mathematical operation or threshold.
Resource confirmation does not establish learning quality or a breakthrough.

## Frozen design

For each batch128/2048 and seed223/239/251, restore H149's step12 model and Adam
state for full GELU checkpoint, full SwiGLU checkpoint and learned fused reversible
stack. Rotate the three-arm order by fixture, then reverse it: ABC CBA.
Every segment restores its source state; these are two independent30-update
segments, not60 continuous updates. The exact initial model and optimizer tensors
are checked against saved source tensors before training. Source hashes are frozen.

Discard10 warmups and pool the remaining20 updates from each repeat for40 timing
observations per arm/fixture. Gradient reset is outside timing; forward, backward,
clipping and AdamW are inside. The timed-body AST matches H149 exactly. Input
gradients and all fixed matrix buffers are included. Final no-grad scoring enters
peak but not timing. Model/optimizer loading and QR setup are outside timing;
setup cost is retained per segment (83.5–
149.9ms). Cached kernel code is unchanged.

## All fixture measurements

Peak is the maximum across both repeats. Timing is the pooled40-update median.
Repeat drift is the larger CUDA/wall ratio between repeat medians. Stable requires
both repeat drift and both segments' half-window median ratios<=1.15.

| Batch | Seed | Arm | Peak MiB | CUDA ms | Wall ms | Repeat drift | Stable |
|---:|---:|---|---:|---:|---:|---:|---|
| 128 | 223 | full_gelu:checkpoint | 62.118 | 14.080 | 14.127 | 1.067 | True |
| 128 | 223 | full_swiglu:checkpoint | 62.137 | 18.822 | 18.877 | 1.113 | True |
| 128 | 223 | learned:fused | 22.720 | 6.845 | 6.886 | 1.025 | True |
| 128 | 239 | full_gelu:checkpoint | 62.118 | 13.156 | 13.203 | 1.030 | True |
| 128 | 239 | full_swiglu:checkpoint | 62.137 | 18.671 | 18.715 | 1.016 | True |
| 128 | 239 | learned:fused | 22.720 | 6.726 | 6.768 | 1.025 | True |
| 128 | 251 | full_gelu:checkpoint | 62.118 | 17.292 | 17.353 | 1.083 | True |
| 128 | 251 | full_swiglu:checkpoint | 62.137 | 22.332 | 22.396 | 1.003 | True |
| 128 | 251 | learned:fused | 22.720 | 6.638 | 6.680 | 1.036 | True |
| 2048 | 223 | full_gelu:checkpoint | 91.886 | 13.743 | 13.785 | 1.016 | False |
| 2048 | 223 | full_swiglu:checkpoint | 102.460 | 19.443 | 19.490 | 1.052 | True |
| 2048 | 223 | learned:fused | 51.345 | 6.521 | 6.565 | 1.010 | True |
| 2048 | 239 | full_gelu:checkpoint | 91.886 | 13.234 | 13.277 | 1.067 | True |
| 2048 | 239 | full_swiglu:checkpoint | 102.460 | 18.704 | 18.748 | 1.054 | True |
| 2048 | 239 | learned:fused | 51.345 | 6.604 | 6.646 | 1.002 | True |
| 2048 | 251 | full_gelu:checkpoint | 91.886 | 13.018 | 13.061 | 1.001 | True |
| 2048 | 251 | full_swiglu:checkpoint | 102.460 | 19.219 | 19.260 | 1.023 | False |
| 2048 | 251 | learned:fused | 51.345 | 6.539 | 6.581 | 1.039 | True |

## Every candidate comparison

Ratios are fused/control. Every fixture must satisfy parameter ratio<=.8,
peak<=.9, CUDA/wall<=1.15 and all candidate/control stability checks. Parameter
ratio is about.0026; the fused model's4.5MiB of fixed matrices is still counted
in allocated GPU peak. This is allocator memory, not total driver VRAM.

| Batch | Seed | Checkpointed control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---:|---:|---:|---|
| 128 | 223 | full_gelu:checkpoint | 0.3658 | 0.486 | 0.487 | none |
| 128 | 223 | full_swiglu:checkpoint | 0.3656 | 0.364 | 0.365 | none |
| 128 | 239 | full_gelu:checkpoint | 0.3658 | 0.511 | 0.513 | none |
| 128 | 239 | full_swiglu:checkpoint | 0.3656 | 0.360 | 0.362 | none |
| 128 | 251 | full_gelu:checkpoint | 0.3658 | 0.384 | 0.385 | none |
| 128 | 251 | full_swiglu:checkpoint | 0.3656 | 0.297 | 0.298 | none |
| 2048 | 223 | full_gelu:checkpoint | 0.5588 | 0.474 | 0.476 | stability |
| 2048 | 223 | full_swiglu:checkpoint | 0.5011 | 0.335 | 0.337 | none |
| 2048 | 239 | full_gelu:checkpoint | 0.5588 | 0.499 | 0.501 | none |
| 2048 | 239 | full_swiglu:checkpoint | 0.5011 | 0.353 | 0.354 | none |
| 2048 | 251 | full_gelu:checkpoint | 0.5588 | 0.502 | 0.504 | none |
| 2048 | 251 | full_swiglu:checkpoint | 0.5011 | 0.340 | 0.342 | stability |

3 individual segments fail the15% half-window stability limit;
identities and values are in unstable_segments.json. Across-repeat failures, if
any, appear above. No selected reruns, clock changes, new thresholds or extensions
were used. Means, medians, sample variances and all raw per-update times are saved.
Forty observations per arm are repeated updates, not40 independent model seeds.

## Independent audit and limits

All36 final saved states independently score on CPU in FP64 using explicit forward
formulas and batch257, without the custom backward. Maximum relative score error
is1.590e-07, below1e-5. Six datasets
regenerate bitwise, immutable matrices match regeneration, parameters/moments and
first/final gradient diagnostics are finite. Optimizer counters equal42 and exact
parameter/buffer/optimizer byte counts verify. Source restoration is checked by
the worker, and source hashes/protocol bindings by the audit; no independent replay
of each Adam update is claimed.

Maximum relative final parameter difference between matched repeats is
0.000e+00; maximum
relative diagnostic-loss difference is
0.000e+00. This describes
these30-update windows, not long-trajectory equivalence or convergence.

There were1,080 optimizer/backward calls,37 clean allocated/reserved CUDA
boundaries,36 final fixed-batch scores plus36 CPU audit scores. Audit initialized
no CUDA and performed no backward. FP32/TF32off, four CPU threads, ordinary
8.125MiB cuBLAS workspace, AdamW LR.003 betas(.9,.95), decay0,clip1. Hardware and
kernel are the same local RTX4070 Laptop GPU setup used in H149. No telemetry,
clock/power change or model/default modification. The initial PowerShell process
suffered a CLR crash after file writes and before native formatting; inspection
confirmed no protocol or GPU run existed. Formatting/checks completed through
cmd, then every experiment stage ran once. All frozen and prior receipt hashes verify.

This reuses six previously tested fixtures and18 saved states deliberately; it
is not fresh-seed or new-task generalization. Fixed-shape and affine controls
remain in H149; this confirmation isolates the three primary resource arms.
No loss-based model selection or held-out quality evaluation occurred.

A pass qualifies this implementation for a separately frozen learning study with
unrelated nonlinear targets, matched parameter/compute controls, fixed-shape and
affine ablations, ordinary activations, independent seeds and held-out metrics.
A fail leaves timing/resource eligibility unresolved under these thresholds.
The full research goal remains open in either case.

[Plan](fused_timing_confirmation_plan.md),
[raw segments](../results/fused_timing_confirmation_v1/result.json),
[audit](../results/fused_timing_confirmation_v1/audit.json),
[receipt](../results/fused_timing_confirmation_v1/receipt.json).
