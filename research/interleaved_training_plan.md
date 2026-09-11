# H140: temporally paired complete-training timing

Previous turn: progress. H139 located timing drift and a P0-to-P4 transition in
H138's ordinary control. H138 remains failed. This is a bounded timing study,
not another convergence test or an attempt to replace its failed observations.

Use all six H138 ordinary step800 checkpoints (two corpora, seeds101/113/127).
For each state run four independent30-update segments, always resetting to the
same source model/Adam/sampler state. Alternate ABBA and BAAB by fixture index,
where A=ordinary and B=buffer4. Execute all24 segments in one process, with only
one model resident and zero allocated/reserved CUDA memory between segments.
This removes per-segment process startup, not model/data reload overhead.

Use the H134 complete-update loop, with only seed metadata corrected to each
fixture and setup timing added. All training math, ten warmups,20 timed updates,
whole-job memory accounting and artifacts remain unchanged. Identical FP32,
TF32off, ordinary attention, default8.125MiB workspace, fourCPUthreads,
UV-managed Python, native AdamW, batch8/context512, whole-block checkpointing.
Buffer4 uses unchanged H128 loss and H137 blocks4-7 offload. No hardware changes.

Primary comparison per fixture: ratio of medians of all40 timed candidate versus
all40 timed control updates. CUDA and wall ratios must each be<=1.15; max whole
job allocated peak ratio<=0.90; candidate pinnedhostpeak<=128MiB. Both repeat NLL
ratios<=1.01. Every segment must pass its existing half-median stability<=1.15,
plus the ratio of the two repeats' median times per arm must be<=1.15. All six
fixtures must pass; no discarded rounds, normalized timings or selected repeats.
Report each temporal neighboring pair too, both repeated controls, mean/median/
sample variance, reload/setup wall time and entire segment wall time separately.
Setup measurement includes construction and initial hash verification; it is not
pure transfer time. Short-segment throughput excludes setup, scoring and saving,
so cannot establish long-run deployment throughput or rescue H138.

Require exact initial-state and batch hashes, all finite states, correct offload
indices, existing gradient-noise caps, independent NumPy clipping/Adam checks
and native final scoring. Passive200ms telemetry matched to actual timed updates;
missing sensors remain missing. Report P-states, clocks and telemetry coverage.
Any state transitions/drift stay in the data. A stability failure rejects scope.

Budget720 optimizer updates/backwards,2,949,120 targets,48 study validation scores
plus24 independent native scores,72 tensor artifacts,1,608 memory-phase records,
25 training+25 audit zero allocator boundaries. No extra audit backwards. Disk
available before preparation about56GB; artifacts expected below8GB. Freeze
sources/inputs/order before execution. Stop on process failure; preserve completed
segments and inspect live state before any prospective recovery. A full pass only
earns integration design with explicit scope and another-scale validation. It
cannot prove novel activations, parameter savings, SOTA or the broader goal.
