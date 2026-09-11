# H133: paired workspace timing with passive GPU telemetry

**PASS paired timing gate.** All 32 bursts match independently constructed native references:
loss, evaluation metadata and every gradient are bitwise identical at their
respective workspace capacity. Model, optimizer and sampler remain unchanged.

Candidate: buffer-reuse classifier, 8-MiB external cuBLAS workspace.
Both controls use the same CPU checkpoint-input offload and deterministic policy.
`native8` uses native loss at 8 MiB; `reuse32` uses buffer reuse at 32 MiB.
Each corpus/control comparison has ABBA then BAAB bursts; four adjacent,
order-balanced candidate/control ratios determine each median. Limit: 1.15.

| Corpus | Control | Median event ratio | Event ratio range | Median wall ratio | Failed gates |
|---|---|---:|---:|---:|---|
| wikitext2 | native8 | 0.9961 | 0.9895-1.0095 | 0.9960 | none |
| wikitext2 | reuse32 | 0.9979 | 0.9908-1.0012 | 0.9979 | none |
| tinystories | native8 | 1.0118 | 0.7862-1.0665 | 1.0119 | none |
| tinystories | reuse32 | 0.9793 | 0.9176-1.0006 | 0.9793 | none |

Median ratios span 0.9793-1.0118: no persistent >15% penalty appears in this
paired test. WikiText-2 burst medians stay near 74-76 ms. TinyStories includes
a 95-ms native burst and an 82-ms reuse32 burst, while most bursts remain
near 75-76 ms. The 95-ms native burst has a 2445-MHz median SM clock and
375-ms process CPU time; the adjacent native burst is 75 ms, 2422.5 MHz,
and 296.9-ms CPU time. Thus a simple low-SM-clock explanation is unsupported
by these samples. Host/dispatch variability is a hypothesis, not a diagnosed
cause. TinyStories/native paired ratios span 0.7862-1.0665; retain that
dispersion despite the passing median. Telemetry coverage is only 1-2 samples
per burst. The next training comparison must retain balanced ordering.

The optional figure is unavailable; complete paired ratios and burst telemetry appear below.

## Evidence and limits

260 backwards, zero optimizer updates, 36 one-batch BF16 evaluations; 1,064,960
diagnostic targets and 147,456 evaluation targets. Four independent native
reference constructions precede the two resident-model corpus experiments.
Seven allocator boundaries reach zero. All 61 maintained files are unchanged.
The CPU audit verifies every reference gradient artifact, raw JSONL equality,
frozen source/input hashes, exactness, budgets, order, timing ratios and telemetry.
Mean, median, variance and extrema are retained in the machine-readable summary.

Each burst has three warmups and five measured passes. Gradient resets occur
outside timing. CUDA-event and synchronized wall intervals include forward,
backward, recomputation and offload transfers. Allocation-inventory barriers
from H132 are absent. This is a different timing harness: cross-study absolute
times cannot isolate workspace or telemetry effects. Host process CPU time is
recorded separately; its sum across threads need not equal wall time.

Read-only nvidia-smi samples were collected every 200 ms in a hidden process
that was stopped after the experiment. Each burst must have at least one sample
in its measured interval. NVIDIA local timestamps are converted using the host
local timezone. Sensor readings have their own averaging windows and may lag.
Power-limit readings are unavailable, retained as missing. GPU clocks and power
were not changed. Correlation cannot prove throttling or identify kernel choice.
These data cannot retrospectively explain H132's unrecorded hardware state.

This is one fixed batch and one seed per corpus, with repeated measurements;
the four pairs are not independent training seeds. No complete-update timing,
quality, new memory reduction, parameter reduction or novelty claim follows.
H132's combined resource gate remains failed. Workspace selection uses the
existing [PyTorch API](https://docs.pytorch.org/docs/stable/backends).

## Every paired ratio

| Corpus | Control | Pair | Event ratio | Wall ratio |
|---|---|---:|---:|---:|
| wikitext2 | native8 | 1 | 0.9895 | 0.9895 |
| wikitext2 | native8 | 2 | 0.9925 | 0.9924 |
| wikitext2 | native8 | 3 | 0.9997 | 0.9997 |
| wikitext2 | native8 | 4 | 1.0095 | 1.0094 |
| wikitext2 | reuse32 | 1 | 1.0012 | 1.0012 |
| wikitext2 | reuse32 | 2 | 0.9992 | 0.9992 |
| wikitext2 | reuse32 | 3 | 0.9967 | 0.9966 |
| wikitext2 | reuse32 | 4 | 0.9908 | 0.9907 |
| tinystories | native8 | 1 | 1.0236 | 1.0238 |
| tinystories | native8 | 2 | 0.7862 | 0.7863 |
| tinystories | native8 | 3 | 1.0665 | 1.0663 |
| tinystories | native8 | 4 | 1.0001 | 1.0001 |
| tinystories | reuse32 | 1 | 0.9924 | 0.9924 |
| tinystories | reuse32 | 2 | 0.9176 | 0.9175 |
| tinystories | reuse32 | 3 | 0.9662 | 0.9662 |
| tinystories | reuse32 | 4 | 1.0006 | 1.0006 |

## Burst chronology and telemetry

CPU ms is process CPU time. Clock, power and temperature are medians of samples
whose timestamps fall between the first and last measured pass in that burst.

| Burst | Corpus | Arm | Event ms | Wall ms | CPU ms | SM MHz | W | Celsius | Samples |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | wikitext2 | reuse8 | 73.788 | 73.831 | 312.500 | 2227.5 | 74.7 | 64.5 | 2 |
| 1 | wikitext2 | native8 | 74.570 | 74.616 | 281.250 | 2437.5 | 102.6 | 74.0 | 2 |
| 2 | wikitext2 | native8 | 75.332 | 75.385 | 312.500 | 2415.0 | 100.6 | 75.0 | 2 |
| 3 | wikitext2 | reuse8 | 74.765 | 74.810 | 312.500 | 2370.0 | 119.4 | 69.0 | 2 |
| 4 | wikitext2 | native8 | 75.294 | 75.345 | 312.500 | 2385.0 | 100.8 | 76.0 | 2 |
| 5 | wikitext2 | reuse8 | 75.275 | 75.319 | 312.500 | 2400.0 | 101.0 | 77.0 | 2 |
| 6 | wikitext2 | reuse8 | 75.992 | 76.042 | 296.875 | 2437.5 | 93.8 | 77.5 | 2 |
| 7 | wikitext2 | native8 | 75.280 | 75.331 | 312.500 | 2415.0 | 98.5 | 76.5 | 2 |
| 8 | wikitext2 | reuse8 | 75.141 | 75.187 | 312.500 | 2475.0 | 103.4 | 76.0 | 2 |
| 9 | wikitext2 | reuse32 | 75.053 | 75.099 | 312.500 | 2422.5 | 96.0 | 78.0 | 2 |
| 10 | wikitext2 | reuse32 | 75.446 | 75.491 | 312.500 | 2355.0 | 96.5 | 78.0 | 2 |
| 11 | wikitext2 | reuse8 | 75.384 | 75.430 | 312.500 | 2415.0 | 86.0 | 77.0 | 2 |
| 12 | wikitext2 | reuse32 | 75.861 | 75.911 | 296.875 | 2430.0 | 94.9 | 78.0 | 1 |
| 13 | wikitext2 | reuse8 | 75.608 | 75.652 | 296.875 | 2310.0 | 100.3 | 79.0 | 2 |
| 14 | wikitext2 | reuse8 | 74.986 | 75.031 | 312.500 | 2355.0 | 91.5 | 79.0 | 2 |
| 15 | wikitext2 | reuse32 | 75.682 | 75.736 | 312.500 | 2317.5 | 98.9 | 80.0 | 2 |
| 16 | tinystories | reuse8 | 76.774 | 76.830 | 312.500 | 2250.0 | 64.2 | 73.0 | 2 |
| 17 | tinystories | native8 | 75.002 | 75.047 | 296.875 | 2422.5 | 97.3 | 79.5 | 2 |
| 18 | tinystories | native8 | 95.048 | 95.096 | 375.000 | 2445.0 | 110.7 | 79.0 | 2 |
| 19 | tinystories | reuse8 | 74.730 | 74.778 | 296.875 | 2407.5 | 80.9 | 79.0 | 2 |
| 20 | tinystories | native8 | 74.846 | 74.893 | 296.875 | 2445.0 | 86.7 | 79.0 | 2 |
| 21 | tinystories | reuse8 | 79.820 | 79.862 | 312.500 | 2400.0 | 99.9 | 82.0 | 2 |
| 22 | tinystories | reuse8 | 75.622 | 75.671 | 312.500 | 2490.0 | 98.8 | 82.0 | 2 |
| 23 | tinystories | native8 | 75.618 | 75.663 | 312.500 | 2490.0 | 95.4 | 83.0 | 2 |
| 24 | tinystories | reuse8 | 75.149 | 75.198 | 312.500 | 2362.5 | 101.0 | 83.0 | 2 |
| 25 | tinystories | reuse32 | 75.725 | 75.772 | 312.500 | 2490.0 | 75.4 | 81.0 | 2 |
| 26 | tinystories | reuse32 | 82.245 | 82.305 | 328.125 | 2370.0 | 101.0 | 81.0 | 2 |
| 27 | tinystories | reuse8 | 75.471 | 75.514 | 312.500 | 2460.0 | 105.7 | 84.0 | 1 |
| 28 | tinystories | reuse32 | 77.768 | 77.823 | 312.500 | 2400.0 | 128.8 | 82.0 | 2 |
| 29 | tinystories | reuse8 | 75.138 | 75.196 | 312.500 | 2332.5 | 108.5 | 83.0 | 2 |
| 30 | tinystories | reuse8 | 76.026 | 76.075 | 312.500 | 2490.0 | 103.1 | 82.0 | 2 |
| 31 | tinystories | reuse32 | 75.982 | 76.028 | 312.500 | 2400.0 | 111.9 | 84.0 | 2 |

The first CPU figure-rendering process failed with a Windows access violation
during Matplotlib import (exit 3221225477). Its log is preserved. One bounded
CPU-only recovery also failed during imports with the same exit code. Both logs
are retained and the optional chart is omitted. No scientific probe was repeated.

## Next decision

Run a separately frozen complete-update comparison at the explicit 8-MiB capacity, using alternating candidate/control windows with telemetry. Include both ordinary execution and matched deterministic native controls, original clipping/Adam, native validation and peak GPU/host memory. Do not combine H132 memory with H133 timing to claim a complete-training win.
The broad VRAM/quality and parameter-efficient FFN research goal remains open.

[Prospective plan](paired_workspace_timing_plan.md),
[raw and summarized evidence](../results/paired_workspace_timing_v1/summary.json),
[receipt](../results/paired_workspace_timing_v1/receipt.json).
