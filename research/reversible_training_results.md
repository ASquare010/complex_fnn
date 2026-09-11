# H148: activation reconstruction inside real training

**REJECT tested reconstruction resource recipe.** The scalar inverse now runs inside a custom autograd backward.
The experiment measures actual complete optimizer updates, including resident
fixed mixing matrices. Numerical correctness and resource eligibility are
separate: all saved states pass independent scoring, but the prospective resource
decision below controls whether this implementation earns a quality study.
No learning-quality or breakthrough claim follows from this fixed-batch screen.

## Mechanism and fair controls

Eight layers compute z=Qx+b and phi(z)=z+a*z/(1+abs(z)), a=.25*tanh(theta).
The reconstruction boundary saves final output, theta, bias and Q. It reconstructs
one layer at a time and returns analytical input/parameter gradients; it does
not retain interior forward activations. Q is immutable dense orthogonal mixing,
created by CPU FP64 QR and cast to FP32. [H147](reversible_softsign_results.md)
derives the inverse, input-Jacobian bounds and contraction-capacity limitation.

At d384/depth8 there are6,144 trainable scalars and1,179,648 fixed matrix entries.
The matrices consume4.5MiB FP32 and are counted in all GPU peaks. Fewer trainable
parameters do not mean the matrices cost no memory, construction or arithmetic.
The learned model has half the forward matrix MACs of the full residual FFN
stacks; analytical inverse and elementwise operations are additional work.

Controls include full GELU/SwiGLU in eager and checkpoint modes, budget GELU
checkpointing, the identical reversible model in eager/checkpoint modes,
frozen-shape reconstruction (3,072 trainable biases), and affine reconstruction.
Affine uses (1+a)*z with the same6,144 parameters and same Q/b initialization.
Fixed-shape and learned nonlinear controls start with identical functions.
Conventional controls are eight residual steps x+FFN(x)/sqrt(8); the reversible
stack is a different architecture whose representational adequacy remains untested.

## Retained-storage trace

CPU hooks at batch2048, deduplicated by storage. Parameter storage is classified
separately in protocol.json. The activation column includes the final output
saved by reconstruction. Original input is separate. Hook-saved buffers do not
exhaust resident buffers: checkpoint closures still hold the model's4.5MiB Q,
although those tensors are not returned by these hooks. All resident buffers are
included in measured GPU peak. Traces exclude loss and optimizer, and hooks are
absent during resource measurement.

| Arm | Saved activation MiB | Saved original input MiB | Hook-saved buffer MiB |
|---|---:|---:|---:|
| full_gelu:eager | 69.000 | 3.000 | 0.000 |
| full_swiglu:eager | 85.000 | 3.000 | 0.000 |
| full_gelu:checkpoint | 21.000 | 3.000 | 0.000 |
| full_swiglu:checkpoint | 21.000 | 3.000 | 0.000 |
| budget_gelu:checkpoint | 21.000 | 3.000 | 0.000 |
| learned:eager | 72.023 | 0.000 | 4.500 |
| learned:checkpoint | 21.000 | 3.000 | 0.000 |
| learned:reconstruct | 3.000 | 0.000 | 4.500 |
| fixed:reconstruct | 3.000 | 0.000 | 4.512 |
| affine:reconstruct | 3.000 | 0.000 | 4.500 |

## Complete GPU training measurements

Means over seeds173/191/211. Each timing is the median of eight updates after
four warmups. Input gradients are enabled; gradient reset is outside timing,
forward/backward/clipping/AdamW inside. A final no-grad diagnostic score is in
job peak but outside timing. QR initialization/setup is outside timing.

| Batch | Arm | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|
| 128 | full_gelu:eager | 62.118 | 9.450 | 9.498 |
| 128 | full_swiglu:eager | 62.137 | 12.375 | 12.420 |
| 128 | full_gelu:checkpoint | 62.118 | 13.031 | 13.074 |
| 128 | full_swiglu:checkpoint | 62.137 | 18.652 | 18.696 |
| 128 | budget_gelu:checkpoint | 50.868 | 13.184 | 13.230 |
| 128 | learned:eager | 26.845 | 9.834 | 9.876 |
| 128 | learned:checkpoint | 24.199 | 17.249 | 17.291 |
| 128 | learned:reconstruct | 24.270 | 14.058 | 14.098 |
| 128 | fixed:reconstruct | 24.246 | 14.881 | 14.923 |
| 128 | affine:reconstruct | 22.912 | 8.298 | 8.340 |
| 2048 | full_gelu:eager | 136.886 | 9.382 | 9.425 |
| 2048 | full_swiglu:eager | 154.585 | 11.486 | 11.531 |
| 2048 | full_gelu:checkpoint | 91.886 | 13.105 | 13.146 |
| 2048 | full_swiglu:checkpoint | 102.460 | 21.179 | 21.227 |
| 2048 | budget_gelu:checkpoint | 84.120 | 12.961 | 13.004 |
| 2048 | learned:eager | 117.347 | 10.317 | 10.360 |
| 2048 | learned:checkpoint | 75.324 | 17.201 | 17.241 |
| 2048 | learned:reconstruct | 76.848 | 14.781 | 14.823 |
| 2048 | fixed:reconstruct | 76.824 | 14.436 | 14.479 |
| 2048 | affine:reconstruct | 57.351 | 8.125 | 8.165 |

The reconstruction trace retains less than checkpointing, yet measured peak can
be higher because backward temporaries also contribute. These peaks establish
the total difference; exact per-operation causality needs an allocation trace.
The cheaper affine control prevents treating reconstruction efficiency as evidence
that learned nonlinear shape itself is useful. No quality comparison is made.

## Every primary comparison

Learned reconstruction divided by each checkpointed full control. Required in
every fixture: parameter ratio<=.8, peak<=.9, CUDA/wall<=1.15, and both candidate
and control half-window stability<=1.15. The parameter ratio is about.0026;
the much larger fixed matrices remain included in the peak column.

| Batch | Seed | Checkpointed control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---:|---|---:|---:|---:|---|
| 128 | 173 | full_gelu:checkpoint | 0.3907 | 1.017 | 1.017 | none |
| 128 | 173 | full_swiglu:checkpoint | 0.3906 | 0.723 | 0.724 | none |
| 128 | 191 | full_gelu:checkpoint | 0.3907 | 1.139 | 1.138 | none |
| 128 | 191 | full_swiglu:checkpoint | 0.3906 | 0.800 | 0.800 | none |
| 128 | 211 | full_gelu:checkpoint | 0.3907 | 1.079 | 1.078 | none |
| 128 | 211 | full_swiglu:checkpoint | 0.3906 | 0.736 | 0.737 | none |
| 2048 | 173 | full_gelu:checkpoint | 0.8363 | 1.173 | 1.172 | timing |
| 2048 | 173 | full_swiglu:checkpoint | 0.7500 | 0.568 | 0.568 | stability |
| 2048 | 191 | full_gelu:checkpoint | 0.8363 | 1.078 | 1.078 | none |
| 2048 | 191 | full_swiglu:checkpoint | 0.7500 | 0.775 | 0.776 | none |
| 2048 | 211 | full_gelu:checkpoint | 0.8363 | 1.135 | 1.134 | none |
| 2048 | 211 | full_swiglu:checkpoint | 0.7500 | 0.806 | 0.807 | none |

5 of60 cases exceed the15% half-window timing-stability threshold;
the exact cases are in unstable_cases.json. Short timing windows are a screen,
not sustained-throughput evidence. No clock/power changes, favorable repeats or
threshold changes were made. Finite runs were retained even when gates failed.

## Verification and limitations

CPU FP64 finite-difference checks cover inputs/theta/bias for nonlinear and affine
custom backward. Learned eager/checkpoint/reconstruction gradients agree within
rtol1e-10/atol1e-12 on exactly FP64-orthogonal small probes. The custom boundary
is first-order only: no higher derivatives, autocast, compiler or stochastic-module
support is claimed. An initial closure-binding lint issue was fixed before any
preflight or frozen experiment ran; no numerical retry was needed.

Sixty saved states independently score in CPU FP64 using explicit matrix and
erf/sigmoid/scalar formulas with batch257. Maximum relative score discrepancy is
1.575e-07, below1e-5. Six datasets
regenerate bitwise, fixed Q and frozen theta match independent regeneration,
optimizer counters equal12, counts/byte totals verify, and model states, moments
and gradient diagnostics are finite. Per-seed means, medians, sample variances,
raw histories, memory and gradient diagnostics are retained. This is saved-state
verification, not independent replay of every Adam update.

For learned checkpoint/reconstruction versus eager from matched initialization,
maximum final parameter relative difference is2.137e-08 and maximum
final loss relative difference is5.997e-08. These are descriptive twelve-step
checks, not bitwise or long-trajectory equivalence guarantees.

There were720 optimizer updates/backwards,61 clean allocated/reserved CUDA
boundaries,60 final fixed-batch scores and60 CPU audit scores. Audit used no
GPU or backward. The worker's AST matches H146's exactly after replacing only
model import and output root. Recipe: FP32/TF32off, four CPU threads, ordinary
kernels/default8.125MiB cuBLAS workspace, AdamW LR.003, betas(.9,.95), decay0,
clip1, twelve steps. The initial plan mentions a diagnostic flag exception, but
no diagnostic logic changed. RTX4070 Laptop GPU; PyTorch2.14.0+cu132,
CUDA build13.2. No hardware telemetry was collected.

The input is one Gaussian batch and its independent orthogonal target. Final
losses are diagnostics, not validation or generalization. No maintained source
or model default changed; prior receipts and frozen hashes verify.

The resource gate alone decides whether the current implementation earns fitting.
A failed recipe requires a new execution hypothesis before promotion; a pass
would still need strong nonlinear targets and conventional activation baselines,
multiple seeds, reduced-width comparisons and held-out quality. The broader
VRAM-and-quality research goal remains open. [RevNets](https://arxiv.org/abs/1707.04585)
established activation reconstruction; this experiment claims no novelty.

[Plan](reversible_training_plan.md), [prototype](../results/reversible_training_v1/model.py),
[results](../results/reversible_training_v1/result.json),
[audit](../results/reversible_training_v1/audit.json),
[receipt](../results/reversible_training_v1/receipt.json).
