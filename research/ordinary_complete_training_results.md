# H135: ordinary-policy complete training

**PASS ordinary short-training gate.** Independent clipping/Adam, native-score and bounded gradient-noise
audit: **True**. Each corpus/control/repeat must pass individually.
This is a fixed-seed short continuation, not a breakthrough or fresh-training claim.

| Corpus | Repeat | Control | GPU allocation saved | Complete CUDA ratio | Wall ratio | NLL ratio | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
| wikitext2 | 0 | ordinary | 24.60% | 1.0425 | 1.0425 | 1.00000039 | none |
| wikitext2 | 0 | native | 14.50% | 0.9883 | 0.9884 | 0.99999902 | none |
| wikitext2 | 1 | ordinary | 24.60% | 1.0335 | 1.0333 | 0.99999951 | none |
| wikitext2 | 1 | native | 14.50% | 0.9975 | 0.9975 | 0.99999715 | none |
| tinystories | 0 | ordinary | 24.98% | 1.1040 | 1.1041 | 1.00001139 | none |
| tinystories | 0 | native | 14.75% | 0.9997 | 0.9999 | 0.99999985 | none |
| tinystories | 1 | ordinary | 24.98% | 1.1050 | 1.1049 | 1.00000024 | none |
| tinystories | 1 | native | 14.75% | 0.9898 | 0.9898 | 0.99999438 | none |

All arms use ordinary/default attention, deterministic algorithms disabled,
no cuBLAS environment override or capacity setter, and the same observed
8.125-MiB workspace. `ordinary`: native loss, checkpoint inputs on GPU.
`native`: native loss with checkpoint-input CPU offload. `reuse`: H128 buffer
classifier with the same CPU offload. All retain 9,099,648 parameters, FP32,
TF32 off, four CPU threads, native RMSNorm, original clipping/Adam schedule,
and the UV-managed Python runtime with PYTHONMALLOC=pymalloc.

## Resource and timing evidence

| Run | Corpus | Arm | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing halves ratio | Telemetry samples |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | wikitext2 | reuse | 306.431 | 64.000 | 70.695 | 70.740 | 4.9633914 | 1.0011 | 7 |
| 1 | wikitext2 | ordinary | 406.399 | 0.000 | 67.810 | 67.854 | 4.9633895 | 1.0006 | 7 |
| 2 | wikitext2 | native | 358.399 | 64.000 | 71.529 | 71.574 | 4.9633963 | 1.0043 | 8 |
| 3 | wikitext2 | native | 358.399 | 64.000 | 71.107 | 71.151 | 4.9633979 | 1.0016 | 7 |
| 4 | wikitext2 | ordinary | 406.399 | 0.000 | 68.627 | 68.683 | 4.9633862 | 1.0108 | 7 |
| 5 | wikitext2 | reuse | 306.431 | 64.000 | 70.926 | 70.970 | 4.9633838 | 1.0014 | 7 |
| 6 | tinystories | reuse | 300.251 | 64.000 | 70.914 | 70.957 | 2.6276998 | 1.0038 | 7 |
| 7 | tinystories | ordinary | 400.220 | 0.000 | 64.236 | 64.264 | 2.6276699 | 1.0015 | 7 |
| 8 | tinystories | native | 352.220 | 64.000 | 70.934 | 70.962 | 2.6277002 | 1.0064 | 7 |
| 9 | tinystories | native | 352.220 | 64.000 | 70.781 | 70.810 | 2.6276845 | 1.0009 | 7 |
| 10 | tinystories | ordinary | 400.220 | 0.000 | 63.399 | 63.430 | 2.6276691 | 1.0015 | 6 |
| 11 | tinystories | reuse | 300.251 | 64.000 | 70.056 | 70.086 | 2.6276697 | 1.0034 | 7 |

Mirrored order per corpus is reuse, ordinary, native, native, ordinary, reuse.
Both repeats must meet allocation ratio <=0.90, complete CUDA/wall <=1.15,
NLL <=1.01, candidate pinned host peak <=128 MiB and timing halves <=1.15.
No failed repeat is averaged away. Repeats share a seed and starting checkpoint.
Twenty measured updates follow ten warmups per run; means, medians, variance,
all steps and reserved allocation are retained in raw machine-readable results.

The unchanged H134 loop times forward, backward, clipping and Adam together,
including recomputation and offload transfers. Gradient clearing, batch sampling,
validation, serialization and inventory are outside timing, but included in
whole-job peak allocation. Allocation excludes driver/context and other processes;
pinned-host measurements are not total CPU RSS. The current matched comparisons
support conclusions within this policy. Cross-study timing differences cannot
prove that a particular deterministic attention kernel caused H134's slowdown.

## Numerical qualification

H128's exact operator qualification remains evidence for the buffer classifier.
H129 showed ordinary full-model gradients are nondeterministic. Accordingly,
this study prospectively uses repeated native controls, not an invalid full-model
bitwise criterion. Symmetric relative L2 uses a 1e-12 norm floor. Limits are:
global min(1e-5,max(1e-6,10*native noise)); tensor min(1e-4,max(1e-5,10*native noise)).
Native-repeat noise must itself remain within the fixed caps. These two repeats
are a noise diagnostic, not a statistical confidence interval. Bitwise match
flags and maximum absolute errors are also retained in the audit JSON.

| Corpus | Control | Gradient | Repeat | Native global noise | Candidate global | Global limit | Native tensor noise | Candidate tensor | Tensor limit | Pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| wikitext2 | ordinary | raw | 0 | 7.259e-08 | 6.765e-08 | 1.000e-06 | 1.666e-07 | 1.666e-07 | 1.000e-05 | True |
| wikitext2 | ordinary | raw | 1 | 7.259e-08 | 3.526e-08 | 1.000e-06 | 1.666e-07 | 8.053e-08 | 1.000e-05 | True |
| wikitext2 | ordinary | clipped | 0 | 7.259e-08 | 6.765e-08 | 1.000e-06 | 1.666e-07 | 1.666e-07 | 1.000e-05 | True |
| wikitext2 | ordinary | clipped | 1 | 7.259e-08 | 3.526e-08 | 1.000e-06 | 1.666e-07 | 8.053e-08 | 1.000e-05 | True |
| wikitext2 | native | raw | 0 | 5.128e-08 | 0.000e+00 | 1.000e-06 | 1.232e-07 | 0.000e+00 | 1.000e-05 | True |
| wikitext2 | native | raw | 1 | 5.128e-08 | 5.128e-08 | 1.000e-06 | 1.232e-07 | 1.232e-07 | 1.000e-05 | True |
| wikitext2 | native | clipped | 0 | 5.128e-08 | 0.000e+00 | 1.000e-06 | 1.232e-07 | 0.000e+00 | 1.000e-05 | True |
| wikitext2 | native | clipped | 1 | 5.128e-08 | 5.128e-08 | 1.000e-06 | 1.232e-07 | 1.232e-07 | 1.000e-05 | True |
| tinystories | ordinary | raw | 0 | 0.000e+00 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 0.000e+00 | 1.000e-05 | True |
| tinystories | ordinary | raw | 1 | 0.000e+00 | 5.162e-08 | 1.000e-06 | 0.000e+00 | 1.067e-07 | 1.000e-05 | True |
| tinystories | ordinary | clipped | 0 | 0.000e+00 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 0.000e+00 | 1.000e-05 | True |
| tinystories | ordinary | clipped | 1 | 0.000e+00 | 5.162e-08 | 1.000e-06 | 0.000e+00 | 1.067e-07 | 1.000e-05 | True |
| tinystories | native | raw | 0 | 0.000e+00 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 0.000e+00 | 1.000e-05 | True |
| tinystories | native | raw | 1 | 0.000e+00 | 5.162e-08 | 1.000e-06 | 0.000e+00 | 1.067e-07 | 1.000e-05 | True |
| tinystories | native | clipped | 0 | 0.000e+00 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 0.000e+00 | 1.000e-05 | True |
| tinystories | native | clipped | 1 | 0.000e+00 | 5.162e-08 | 1.000e-06 | 0.000e+00 | 1.067e-07 | 1.000e-05 | True |

Final model weight distances are descriptive; final native validation and the
original finite/state/optimizer audits decide short-run qualification. This
separates numerical drift from a claim of exact trajectory identity.

| Corpus | Control | Repeat | Final global distance | Max tensor distance | Max absolute difference |
|---|---|---|---:|---:|---:|
| wikitext2 | ordinary | 0 | 6.023e-08 | 1.544e-07 | 2.235e-07 |
| wikitext2 | ordinary | 1 | 5.932e-08 | 1.498e-07 | 2.384e-07 |
| wikitext2 | ordinary | native_noise | 6.011e-08 | 1.529e-07 | 3.055e-07 |
| wikitext2 | native | 0 | 5.908e-08 | 1.490e-07 | 1.416e-07 |
| wikitext2 | native | 1 | 5.960e-08 | 1.504e-07 | 2.161e-07 |
| wikitext2 | native | native_noise | 5.970e-08 | 1.515e-07 | 2.831e-07 |
| tinystories | ordinary | 0 | 7.025e-08 | 2.370e-07 | 2.488e-06 |
| tinystories | ordinary | 1 | 7.026e-08 | 2.416e-07 | 1.101e-06 |
| tinystories | ordinary | native_noise | 6.961e-08 | 2.380e-07 | 1.572e-06 |
| tinystories | native | 0 | 6.644e-08 | 2.373e-07 | 4.843e-07 |
| tinystories | native | 1 | 6.903e-08 | 2.431e-07 | 5.513e-07 |
| tinystories | native | native_noise | 6.828e-08 | 2.363e-07 | 1.103e-06 |

## Reproducibility and limits

360 updates/backwards, 1,474,560 training targets, 24 full study scores and 12
independent native scores. Thirty-six tensor artifacts, 804 memory intervals,
37 zero allocator boundaries; all 61 maintained source/config/test files stay
unchanged. Frozen source, fixture/data and library hashes verify. Independent
NumPy checks clipping and the first Adam update/moments for every run; checkpoint
steps, finite states, saved hashes and all 360 batches are verified separately.

Read-only GPU telemetry samples every 200 ms and each monitor is stopped in
finally. All cases require coverage during measured updates. Clock/power and
temperature ranges, medians and raw samples are retained; missing sensors remain
missing. No GPU clock or power setting changed. Sparse telemetry cannot identify
a causal bottleneck. No historical failed gate is revised by this result.

This tests a memory implementation on existing fixtures, not novel activation
geometry, parameter reduction, multiple independent seeds, fresh convergence,
held-out broad task performance or SOTA superiority. No maintainable default
is changed until the stronger training evidence exists.

Case07 failed during torch.jit import with a Python SystemError before GPU work.
Cases00-06 (210 updates) were retained. A prospective bounded recovery restarted
only unfinished case07, then ran cases08-11 in the frozen order and unchanged
environment. The failure log is retained. No completed scientific case was repeated.
Automatic approval review initially could not complete recovery setup; inspection
confirmed no files/process were created, and the subsequent setup succeeded.

CPU audit preparation then crashed in Python's indented JSON encoder before
writing either aggregate result or audit manifest. One CPU-only compact-JSON
recovery completed the same seal; original source and failure log are retained.
No GPU work was repeated for this operational recovery.

## Next decision

Run fresh longer training with at least three independent seeds on both corpora, paired ordinary/native-offload/buffer-offload controls, native validation and complete-update resource accounting. The short result earns that test; it does not prove fresh convergence or long-run quality. Preserve the existing structural FFN failures while pursuing the broader parameter-efficiency goal.
The full research goal remains open.

[Prospective plan](ordinary_complete_training_plan.md),
[summary](../results/ordinary_complete_training_v1/summary.json),
[receipt](../results/ordinary_complete_training_v1/receipt.json).
