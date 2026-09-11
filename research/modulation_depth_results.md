# H146: depth, saved activations and fair recomputation controls

**REJECT tested depth/recomputation modulation recipes.** This study tests actual VRAM and update cost for eight residual
FFNs, with the same recomputation option offered to every architecture. It does
not measure held-out quality or establish a new activation. H145's isolated eager
failure remains unchanged. The broader research goal remains open.

## What changed and why

Each step is x+FFN(x)/sqrt(8), repeated eight times with independently initialized
blocks. Full GELU uses hidden384, full SwiGLU256, budget GELU288, shared gate192,
and split modulation uses base192/gate96. The latter two use about24.8% fewer
parameters and25% fewer matrix MACs than both full controls. The exact module
implementation is H145's. Shared/split parameter totals are1,777,152/1,777,920;
full GELU/SwiGLU totals2,365,440/2,366,464. Budget GELU has1,774,848.

Eight layers test accumulation of saved intermediates. Each architecture runs
both ordinary autograd and non-reentrant checkpointing of each entire residual
step. Thus a checkpointed candidate must beat checkpointed controls. Comparing
only against unoptimized controls would not establish an architecture advantage.
This is an unnormalized residual MLP resource probe, not a Transformer benchmark.

[PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html) is a
known compute-for-memory tradeoff. RNG preservation is disabled here because
these modules contain no stochastic operations. CPU FP64 outputs and all input/
parameter gradients agree at rtol1e-10/atol1e-12 for all five architectures;
the gate weights are perturbed away from zero for these correctness checks.

## Saved-storage evidence

Batch2048, width384, depth8, FP32; CPU hooks deduplicate storage addresses to
avoid counting parameter transposes and activation views multiple times.
Parameter storage is classified separately and available in the raw traces.
The original input is separate from the activation column. This measures tensors
retained for backward after forward, not GPU peak or all live forward outputs.
The loss/output buffer and optimizer temporaries are outside this hook trace.
No hooks run during GPU resource measurements.

| Architecture:execution | Saved activation MiB | Original input MiB |
|---|---:|---:|
| full_gelu:eager | 69.0 | 3.0 |
| full_swiglu:eager | 85.0 | 3.0 |
| budget_gelu:eager | 57.0 | 3.0 |
| shared_gate:eager | 69.0 | 3.0 |
| split_gate:eager | 81.0 | 3.0 |
| full_gelu:checkpoint | 21.0 | 3.0 |
| full_swiglu:checkpoint | 21.0 | 3.0 |
| budget_gelu:checkpoint | 21.0 | 3.0 |
| shared_gate:checkpoint | 21.0 | 3.0 |
| split_gate:checkpoint | 21.0 | 3.0 |

Eager split modulation retains81MiB versus full GELU69MiB; shared retains69MiB.
Both readouts in the shared candidate can alias one feature bank, whereas split
views retain the larger concatenated bank. Checkpointing leaves the same21MiB
of intermediate block inputs for every architecture, plus the original3MiB input.
This equal floor explains why eliminating internal saved tensors alone cannot
establish an architecture-specific memory advantage. GPU peaks additionally
include parameters, gradients, optimizer states and recomputation temporaries.

## GPU measurements

Means across three seeds. Timing entries average each run's median of eight
complete updates after four warmups. Gradient clearing is outside timing;
forward, backward, clipping and AdamW are inside. Input gradients are enabled.
The no-grad final diagnostic score is included in peak but outside timing.

| Batch | Architecture:execution | Allocated peak MiB | CUDA ms | Wall ms |
|---:|---|---:|---:|---:|
| 128 | full_gelu:eager | 62.118 | 9.380 | 9.428 |
| 128 | full_swiglu:eager | 62.137 | 12.144 | 12.189 |
| 128 | budget_gelu:eager | 50.868 | 8.359 | 8.407 |
| 128 | shared_gate:eager | 50.907 | 12.051 | 12.095 |
| 128 | split_gate:eager | 50.926 | 13.220 | 13.265 |
| 128 | full_gelu:checkpoint | 62.118 | 13.015 | 13.058 |
| 128 | full_swiglu:checkpoint | 62.137 | 19.253 | 19.299 |
| 128 | budget_gelu:checkpoint | 50.868 | 13.441 | 13.488 |
| 128 | shared_gate:checkpoint | 50.907 | 19.155 | 19.199 |
| 128 | split_gate:checkpoint | 50.926 | 21.137 | 21.186 |
| 2048 | full_gelu:eager | 136.886 | 8.001 | 8.045 |
| 2048 | full_swiglu:eager | 154.585 | 11.626 | 11.670 |
| 2048 | budget_gelu:eager | 119.745 | 7.896 | 7.941 |
| 2048 | shared_gate:eager | 129.878 | 13.686 | 13.732 |
| 2048 | split_gate:eager | 145.157 | 13.332 | 13.377 |
| 2048 | full_gelu:checkpoint | 91.886 | 13.365 | 13.408 |
| 2048 | full_swiglu:checkpoint | 102.460 | 18.680 | 18.722 |
| 2048 | budget_gelu:checkpoint | 84.120 | 13.126 | 13.169 |
| 2048 | shared_gate:checkpoint | 86.878 | 19.273 | 19.316 |
| 2048 | split_gate:checkpoint | 88.249 | 21.569 | 21.614 |

## Candidate gates against equally optimized full controls

Ranges across seeds, candidate/control. Required: parameters<=.80, peak<=.90,
CUDA and wall<=1.15, candidate and control half-window stability<=1.15 in every
fixture. Each comparison uses the same execution mode. Failures list any gate
failed by at least one seed; individual comparisons are in audit.json.

| Batch | Candidate | Same-mode control | Peak ratio | CUDA ratio | Wall ratio | Failed gates |
|---:|---|---|---:|---:|---:|---|
| 128 | shared_gate:eager | full_gelu | 0.820–0.820 | 1.078–1.491 | 1.077–1.489 | timing |
| 128 | shared_gate:eager | full_swiglu | 0.819–0.819 | 0.850–1.105 | 0.850–1.105 | none |
| 2048 | shared_gate:eager | full_gelu | 0.949–0.949 | 1.481–2.135 | 1.479–2.129 | peak, timing, stability |
| 2048 | shared_gate:eager | full_swiglu | 0.840–0.840 | 1.017–1.492 | 1.017–1.491 | timing, stability |
| 128 | split_gate:eager | full_gelu | 0.820–0.820 | 1.217–1.673 | 1.216–1.670 | timing |
| 128 | split_gate:eager | full_swiglu | 0.820–0.820 | 0.959–1.241 | 0.959–1.240 | timing |
| 2048 | split_gate:eager | full_gelu | 1.060–1.060 | 1.598–1.748 | 1.595–1.744 | peak, timing |
| 2048 | split_gate:eager | full_swiglu | 0.939–0.939 | 1.085–1.222 | 1.085–1.221 | peak, timing |
| 128 | shared_gate:checkpoint | full_gelu | 0.820–0.820 | 1.447–1.518 | 1.446–1.517 | timing |
| 128 | shared_gate:checkpoint | full_swiglu | 0.819–0.819 | 0.947–1.024 | 0.947–1.024 | none |
| 2048 | shared_gate:checkpoint | full_gelu | 0.945–0.945 | 1.422–1.476 | 1.421–1.475 | peak, timing |
| 2048 | shared_gate:checkpoint | full_swiglu | 0.848–0.848 | 1.002–1.061 | 1.002–1.061 | none |
| 128 | split_gate:checkpoint | full_gelu | 0.820–0.820 | 1.612–1.631 | 1.610–1.629 | timing, stability |
| 128 | split_gate:checkpoint | full_swiglu | 0.820–0.820 | 1.016–1.149 | 1.017–1.149 | stability |
| 2048 | split_gate:checkpoint | full_gelu | 0.960–0.960 | 1.540–1.678 | 1.539–1.676 | peak, timing |
| 2048 | split_gate:checkpoint | full_swiglu | 0.861–0.861 | 1.085–1.206 | 1.084–1.206 | timing |

There are2 unstable timing cases; their exact arm/seed/size and
half-window ratios are retained in unstable_cases.json. Eight timed updates
are a screen, not sustained-throughput proof. No favorable repeats were selected.
Memory failures remain valid independently of noisy timing comparisons.

## Checkpoint/eager comparison within each architecture

Means of three paired seed ratios. These are execution effects, not evidence of
novel architecture or preserved quality. Lower peak with slower updates is a
tradeoff and must satisfy the same prospective gate before promotion.

| Batch | Architecture | Peak ratio | CUDA ratio | Wall ratio |
|---:|---|---:|---:|---:|
| 128 | full_gelu | 1.000 | 1.413 | 1.411 |
| 128 | full_swiglu | 1.000 | 1.596 | 1.594 |
| 128 | budget_gelu | 1.000 | 1.609 | 1.605 |
| 128 | shared_gate | 1.000 | 1.590 | 1.588 |
| 128 | split_gate | 1.000 | 1.599 | 1.598 |
| 2048 | full_gelu | 0.671 | 1.670 | 1.667 |
| 2048 | full_swiglu | 0.663 | 1.608 | 1.605 |
| 2048 | budget_gelu | 0.702 | 1.663 | 1.659 |
| 2048 | shared_gate | 0.669 | 1.445 | 1.443 |
| 2048 | split_gate | 0.608 | 1.620 | 1.618 |

## Audit, scope and next decision

Sixty runs,720 optimizer updates/backwards,61 zero allocated/reserved CUDA
boundaries,60 final diagnostic scores plus60 independent CPU audit scores.
All six Gaussian/orthogonal-target datasets regenerate bitwise. Saved states and
Adam moments are finite, step counters equal12, parameter/optimizer counts and
all recorded timing summaries verify. First/final input and post-clipping parameter
gradients are finite; this does not prove freedom from deep gradient pathologies.

Independent CPU FP64 scoring uses explicit matrices, erf/sigmoid nonlinearities,
and batch257 rather than the trained model's forward function. Maximum relative
score error: 8.189e-08, below1e-5.
The audit performs no backward and initializes no CUDA. It verifies saved-state
scoring, not an independent replay of Adam. Mean, median and sample variance
across seeds, raw timings, losses and gradient diagnostics are machine-readable.
Losses are fixed-batch diagnostics: there is no validation/generalization claim.

Seeds83/97/109, batches128/2048, d384, eight residual blocks,12 AdamW updates,
LR.003, betas(.9,.95), zero decay, clip1, FP32/TF32off, four CPU threads,
ordinary eager kernels and8.125MiB default cuBLAS workspace. Arm order rotates
by fixture. RTX4070 Laptop GPU, driver610.62; PyTorch2.14.0+cu132,
CUDA build13.2. Hardware was queried during the run. No clock/power
changes or telemetry collection; clock/thermal causality is unknown.

Only resource-passing candidates may earn a separately frozen quality study.
Failures close these depth/execution settings; a kernel, precision, recomputation
schedule or architecture change requires a distinct hypothesis and protocol.
No maintained implementation/default changed; H145 receipt and source hashes
verify. This work supplies measured elimination evidence, not a breakthrough.

[Plan](modulation_depth_plan.md), [raw results](../results/modulation_depth_v1/result.json),
[audit](../results/modulation_depth_v1/audit.json),
[receipt](../results/modulation_depth_v1/receipt.json).
