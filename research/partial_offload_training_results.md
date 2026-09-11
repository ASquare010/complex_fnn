# H137: fewer checkpoint-input transfers

**PASS short qualification: buffer4.** Independent clipping/Adam/native-score/gradient-noise audit:
**True**. Eligible new variants: **buffer0, buffer4**.
Every corpus and mirrored repeat must pass for a new variant to qualify.
The eight-block arm is a control; it cannot override H136's long-run failure.

| Corpus | Repeat | Variant | GPU allocation saved vs ordinary | Complete CUDA ratio | Wall ratio | NLL ratio | Failed gates |
|---|---:|---|---:|---:|---:|---:|---|
| wikitext2 | 0 | buffer0 | 12.79% | 1.0085 | 1.0085 | 0.99999799 | none |
| wikitext2 | 0 | buffer4 | 18.69% | 1.0522 | 1.0522 | 0.99999888 | none |
| wikitext2 | 0 | buffer8 | 24.60% | 1.1059 | 1.1059 | 0.99999423 | none |
| wikitext2 | 1 | buffer0 | 12.79% | 1.0047 | 1.0045 | 1.00000054 | none |
| wikitext2 | 1 | buffer4 | 18.69% | 1.0530 | 1.0527 | 1.00000115 | none |
| wikitext2 | 1 | buffer8 | 24.60% | 1.1041 | 1.1038 | 1.00000114 | none |
| tinystories | 0 | buffer0 | 12.98% | 0.9929 | 0.9928 | 1.00000282 | none |
| tinystories | 0 | buffer4 | 18.98% | 1.0547 | 1.0546 | 0.99999895 | none |
| tinystories | 0 | buffer8 | 24.98% | 1.1056 | 1.1055 | 0.99999184 | none |
| tinystories | 1 | buffer0 | 12.98% | 0.9897 | 0.9898 | 0.99999362 | none |
| tinystories | 1 | buffer4 | 18.98% | 1.0534 | 1.0533 | 0.99999829 | none |
| tinystories | 1 | buffer8 | 24.98% | 1.1049 | 1.1049 | 0.99998366 | none |

`ordinary` uses native loss without input offload. All buffer variants use the
same H128 loss. The suffix gives the number of offloaded blocks: zero, last four,
or all eight. The original native CPU-save adapter is scoped through a view of
selected blocks; model registration, parameter identities and numerical layer
operations remain unchanged. There are still 9,099,648 trainable parameters.

Each FP32 checkpoint input has nominal payload 8 x 512 x 384 x 4 bytes = 6 MiB.
Selecting four blocks therefore transfers/retains half the nominal checkpoint
input payload of eight. This is not a formula for whole-job peak reduction:
the peak can move, and the table reports measured allocation including all phases.
No activation or offload-algorithm novelty is claimed.

## Measurements

| Run | Corpus | Variant | GPU peak MiB | Pinned host peak MiB | Complete CUDA ms | Wall ms | Final NLL | Timing halves ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | wikitext2 | ordinary | 406.399 | 0.000 | 64.025 | 64.053 | 4.912149 | 1.0012 |
| 1 | wikitext2 | buffer0 | 354.431 | 0.000 | 64.568 | 64.598 | 4.912139 | 1.0091 |
| 2 | wikitext2 | buffer4 | 330.431 | 32.000 | 67.365 | 67.396 | 4.912143 | 1.0050 |
| 3 | wikitext2 | buffer8 | 306.431 | 64.000 | 70.806 | 70.833 | 4.912120 | 1.0013 |
| 4 | wikitext2 | buffer8 | 306.431 | 64.000 | 70.715 | 70.744 | 4.912136 | 1.0029 |
| 5 | wikitext2 | buffer4 | 330.431 | 32.000 | 67.441 | 67.466 | 4.912136 | 1.0041 |
| 6 | wikitext2 | buffer0 | 354.431 | 0.000 | 64.348 | 64.379 | 4.912133 | 1.0081 |
| 7 | wikitext2 | ordinary | 406.399 | 0.000 | 64.047 | 64.090 | 4.912130 | 1.0077 |
| 8 | tinystories | ordinary | 400.220 | 0.000 | 64.279 | 64.309 | 2.620840 | 1.0034 |
| 9 | tinystories | buffer0 | 348.251 | 0.000 | 63.820 | 63.848 | 2.620847 | 1.0123 |
| 10 | tinystories | buffer4 | 324.251 | 32.000 | 67.793 | 67.822 | 2.620837 | 1.0142 |
| 11 | tinystories | buffer8 | 300.251 | 64.000 | 71.068 | 71.095 | 2.620818 | 1.0116 |
| 12 | tinystories | buffer8 | 300.251 | 64.000 | 71.126 | 71.159 | 2.620800 | 1.0056 |
| 13 | tinystories | buffer4 | 324.251 | 32.000 | 67.809 | 67.836 | 2.620838 | 1.0099 |
| 14 | tinystories | buffer0 | 348.251 | 0.000 | 63.713 | 63.743 | 2.620826 | 1.0032 |
| 15 | tinystories | ordinary | 400.220 | 0.000 | 64.373 | 64.402 | 2.620842 | 1.0478 |

Both corpora use H136 ordinary seed101 step800 checkpoints, including optimizer
and sampler states. Mirrored order is ordinary, buffer0, buffer4, buffer8,
buffer8, buffer4, buffer0, ordinary. Repeats share one starting state; they are
not independent seeds. The frozen selection rule prefers greater worst-case
memory savings among buffer0/4 passing every comparison, then runtime. Buffer8
remains in the table even when it passes this short test.

Limits are memory ratio <=0.90, complete CUDA/wall <=1.15, NLL <=1.01, candidate
pinned host peak <=128 MiB and timing halves <=1.15, plus numerical/source/telemetry
checks. No limit changed and no failed repeat is averaged away. Raw records retain
means, medians and sample variance. Candidate timings include forward, backward,
clipping, Adam, recomputation and transfers; they exclude sampling, gradient reset,
validation, serialization and inventory. Whole-job GPU peak includes all phases;
driver/context and other processes are excluded. Pinned peak is not total CPU RSS.

## Independent numerical evidence

Inherited NumPy checks verify original clipping and first Adam updates/moments.
Every final checkpoint is independently scored through native code, and all batches
are replayed. Parameter/optimizer/sampler hashes and finite states are checked.
Ordinary attention is nondeterministic, so H135's bounded native-repeat calibration
is retained rather than claiming bitwise model trajectories. Native-repeat noise,
bitwise flags and final weight differences are retained in the audit JSON.

| Corpus | Variant | Gradient | Repeat | Global error | Global limit | Max tensor error | Tensor limit | Pass |
|---|---|---|---:|---:|---:|---:|---:|---|
| wikitext2 | buffer0 | raw | 0 | 8.331e-08 | 1.000e-06 | 2.244e-07 | 1.000e-05 | True |
| wikitext2 | buffer0 | raw | 1 | 7.240e-08 | 1.000e-06 | 2.079e-07 | 1.000e-05 | True |
| wikitext2 | buffer0 | clipped | 0 | 8.331e-08 | 1.000e-06 | 2.244e-07 | 1.000e-05 | True |
| wikitext2 | buffer0 | clipped | 1 | 7.240e-08 | 1.000e-06 | 2.079e-07 | 1.000e-05 | True |
| wikitext2 | buffer4 | raw | 0 | 6.794e-08 | 1.000e-06 | 1.733e-07 | 1.000e-05 | True |
| wikitext2 | buffer4 | raw | 1 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 1.000e-05 | True |
| wikitext2 | buffer4 | clipped | 0 | 6.794e-08 | 1.000e-06 | 1.733e-07 | 1.000e-05 | True |
| wikitext2 | buffer4 | clipped | 1 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 1.000e-05 | True |
| wikitext2 | buffer8 | raw | 0 | 6.794e-08 | 1.000e-06 | 1.733e-07 | 1.000e-05 | True |
| wikitext2 | buffer8 | raw | 1 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 1.000e-05 | True |
| wikitext2 | buffer8 | clipped | 0 | 6.794e-08 | 1.000e-06 | 1.733e-07 | 1.000e-05 | True |
| wikitext2 | buffer8 | clipped | 1 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 1.000e-05 | True |
| tinystories | buffer0 | raw | 0 | 7.973e-08 | 1.000e-06 | 1.487e-07 | 1.000e-05 | True |
| tinystories | buffer0 | raw | 1 | 6.209e-08 | 1.000e-06 | 1.266e-07 | 1.000e-05 | True |
| tinystories | buffer0 | clipped | 0 | 7.973e-08 | 1.000e-06 | 1.487e-07 | 1.000e-05 | True |
| tinystories | buffer0 | clipped | 1 | 6.209e-08 | 1.000e-06 | 1.266e-07 | 1.000e-05 | True |
| tinystories | buffer4 | raw | 0 | 8.223e-08 | 1.000e-06 | 1.491e-07 | 1.000e-05 | True |
| tinystories | buffer4 | raw | 1 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 1.000e-05 | True |
| tinystories | buffer4 | clipped | 0 | 8.223e-08 | 1.000e-06 | 1.491e-07 | 1.000e-05 | True |
| tinystories | buffer4 | clipped | 1 | 0.000e+00 | 1.000e-06 | 0.000e+00 | 1.000e-05 | True |
| tinystories | buffer8 | raw | 0 | 7.973e-08 | 1.000e-06 | 1.487e-07 | 1.000e-05 | True |
| tinystories | buffer8 | raw | 1 | 6.872e-08 | 1.000e-06 | 1.450e-07 | 1.000e-05 | True |
| tinystories | buffer8 | clipped | 0 | 7.973e-08 | 1.000e-06 | 1.487e-07 | 1.000e-05 | True |
| tinystories | buffer8 | clipped | 1 | 6.872e-08 | 1.000e-06 | 1.450e-07 | 1.000e-05 | True |

Global limit=min(1e-5,max(1e-6,10*native noise)); tensor limit=min(1e-4,max(1e-5,
10*native noise)). Native noise must itself satisfy the caps. This two-repeat
noise diagnostic is not a statistical confidence interval. The existing exact
operator qualification remains complementary evidence.

## Reproducibility and limits

480 updates/backwards and 1,966,080 training targets. Thirty-two full study scores
plus sixteen independent native scores; 48 tensor artifacts, 1,072 memory
intervals and 49 zero allocator boundaries. All 61 maintained files and frozen
source/input/library hashes verify. The actual wrapped block indices are recorded
and checked for every run. Original FP32, default attention/workspace, TF32 off,
four threads, UV-managed Python and PYTHONMALLOC=pymalloc are shared by all arms.

Read-only GPU telemetry is sampled every 200 ms, coverage is required during
measured updates, and monitors stop in finally. Raw samples and sensor summaries
remain available; unavailable readings stay missing. No hardware power/clock
settings change. These data do not establish a causal explanation for H136's
runtime outlier. H136's failed gate remains unchanged. No maintained default is
modified, and this test establishes neither fresh training nor parameter reduction.

The initial CPU sealing process crashed in the compact JSON encoder before
writing an aggregate or manifest. One bounded CPU recovery streamed the existing
case JSON text verbatim into the aggregate, preserving numerical values. Original
source, failure log and exit are retained. No scientific work was repeated.

## Next decision

Validate buffer4 from fresh initialization against ordinary native training on both corpora and three seeds, using complete-update timing and native scoring. Preserve H136's failed full-offload result; this short continuation does not prove fresh convergence or robust long-run efficiency. The broader FFN/activation parameter-efficiency objective remains open.

[Prospective plan](partial_offload_training_plan.md),
[summary](../results/partial_offload_training_v1/summary.json),
[receipt](../results/partial_offload_training_v1/receipt.json).
