# H139: timing drift in the failed four-block validation

**H138 remains failed; no candidate is promoted.** This CPU-only audit verifies
all prior evidence hashes and accounts for every one of the 9,360 timed updates,
in 360 windows and 36 original timing blocks. It uses zero GPU updates/backwards.

| Case | Corpus | Seed | Arm | Three block mean wall times (ms) | Active-sample block median SM clocks (MHz) | Stability ratio | Window clock/time Pearson r |
|---:|---|---:|---|---|---|---:|---:|
| 0 | wikitext2 | 101 | ordinary | [64.11, 66.14, 68.99] | [2190.0, 2130.0, 2017.5] | 1.0761 | -0.910 |
| 1 | wikitext2 | 101 | buffer4 | [73.18, 76.61, 78.51] | [1995.0, 1860.0, 1830.0] | 1.0727 | -0.925 |
| 2 | wikitext2 | 113 | buffer4 | [76.65, 79.09, 80.06] | [1875.0, 1800.0, 1755.0] | 1.0445 | -0.872 |
| 3 | wikitext2 | 113 | ordinary | [75.17, 77.41, 78.73] | [1770.0, 1680.0, 1650.0] | 1.0474 | -0.782 |
| 4 | wikitext2 | 127 | ordinary | [76.02, 78.49, 79.27] | [1740.0, 1650.0, 1635.0] | 1.0427 | -0.865 |
| 5 | wikitext2 | 127 | buffer4 | [79.57, 82.07, 82.77] | [1770.0, 1695.0, 1665.0] | 1.0403 | -0.805 |
| 6 | tinystories | 101 | buffer4 | [80.12, 82.3, 82.93] | [1740.0, 1710.0, 1680.0] | 1.0351 | -0.645 |
| 7 | tinystories | 101 | ordinary | [76.97, 79.36, 79.74] | [1695.0, 1620.0, 1605.0] | 1.0359 | -0.812 |
| 8 | tinystories | 113 | ordinary | [77.61, 79.75, 80.19] | [1710.0, 1605.0, 1605.0] | 1.0332 | -0.836 |
| 9 | tinystories | 113 | buffer4 | [80.96, 82.63, 83.69] | [1710.0, 1695.0, 1650.0] | 1.0337 | -0.726 |
| 10 | tinystories | 127 | buffer4 | [80.05, 82.45, 83.74] | [1755.0, 1680.0, 1650.0] | 1.0461 | -0.836 |
| 11 | tinystories | 127 | ordinary | [78.04, 79.94, 142.0] | [1680.0, 1620.0, 915.0] | 1.8196 | -0.877 |

Each block has 260 updates; each correlation uses up to 30 consecutive 26-update
windows after the original 20 warmups. Sensor samples are included only when
they fall inside a measured update, excluding validation, serialization and
sampling gaps. Sampling is sparse (200 ms), not a per-kernel measurement; missing
sensors stay missing. Window observations are autocorrelated. Correlations are
descriptive, with no causal inference or significance claim. P-states, phase
summaries, sample coverage and every window are retained in `analysis.json`.

The first WikiText ordinary/candidate pair ran at different clock distributions.
The final TinyStories ordinary control changes speed substantially within its
run. These observations make sequential whole-run timing comparisons vulnerable
to time-varying device conditions. They do not establish whether temperature,
power management, scheduling or another workload caused the changes. No measured
time is clock-normalized, removed or replaced, and no failed gate is rescued.

## Localized failure

The last ordinary control (TinyStories seed127) has only P0 samples through
update540. Its541-566 window contains both P0 and P4; all subsequent sampled
windows contain only P4. The three original block means are78.04,79.94 and142.00
ms/update, while active-sample median SM clocks are1680,1620 and915MHz. Its
stability ratio1.8196 fails the unchanged1.15 limit. This is a measured change
in the control's execution conditions, not evidence of candidate instability.
The first WikiText pair also has different clock distributions; its16.7% runtime
overhead remains a failed observation, not a corrected or excused measurement.

A separate linear interval lookup independently reproduced every window's sample
count and P-state composition; original records reproduced all window timing
means. All12 runs remain included. The device-state cause remains unestablished.

## Next experiment

Separate the already audited convergence evidence from a new, explicitly timed
experiment: interleave short complete-training segments for both arms at the
same saved model/optimizer states, in balanced ABBA/BAAB rounds for every corpus
and seed. Keep only one model resident, account for its complete job allocation,
record state reload overhead separately, and include forward/backward/clipping/
Adam plus offload transfers inside the measured segment. Use every round, report
paired ratios and within-run drift, retain the original 15% runtime limit, and
freeze exact counts/order/numerical checks before GPU execution. This tests
runtime under temporal pairing; it does not replace H138 or establish deployment
throughput, convergence or generality. Do not repeat 9,600 training updates merely
to obtain a more favorable sequential timing result.

The broader objective still requires robust VRAM/quality/compute savings and
independent exploration of parameter-efficient FFN geometry. Offload and loss
buffer reuse are established memory mechanisms, not novelty claims.

[Plan](timing_drift_audit_plan.md) ·
[Machine-readable analysis](../results/timing_drift_audit_v1/analysis.json)
