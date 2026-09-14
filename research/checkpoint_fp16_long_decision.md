# H160 decision and next research step

H160's evidence audit passes; its qualification fails. Whole-job allocated GPU
memory falls 23.72-23.76%, but WikiText2 seed113 final validation NLL rises 2.26%
(limit 1%). WikiText2 seed101 FP16 timing stability also fails.

The timing record contains a 6.6569-hour update at step503 and a 173.7-second
update at step524. These are elapsed-time observations, not proof of a hardware,
driver or suspend cause. Keep the failed gate; median timing does not establish
uninterrupted throughput. The separate seed113 pair spans the user pause.

The loss result does not isolate FP16 rounding causally: ordinary attention is
not deterministic and these are independently trained trajectories. It does
establish that the fixed recipe missed its prospective quality requirement.
No completed seed was rerun to replace an unfavorable outcome.

## Why progress has slowed

Parameter reductions have repeatedly reduced capacity or increased intermediate
storage and runtime. Short numerical and speed checks also failed to predict
fresh-training quality here. More minor variants and larger sweeps without a
specific mechanism would repeat this problem. H156's exact memory helper remains
qualified within its tested scope; the architectural breakthrough is unresolved.

## Next bounded direction

Prioritize extending the exact H156 helper to one larger workload, measuring
whole-job memory including validation and complete optimizer updates. Freeze a
small cost budget and gates before execution. Use a stable uninterrupted session;
record interruptions as failures of measurement, without silently trimming them.
This is an engineering generalization test, not new-architecture evidence.

Before spending on another architecture, require a specific capacity argument
and a cheap comparison with both the wide baseline and its matched narrow
baseline. Advance only if the mechanism survives quality and actual memory
checks. Do not launch another long FP16 sweep solely to rescue H160.

[Full result](checkpoint_fp16_long_results.md),
[reproducible CPU verification](checkpoint_fp16_long_postmortem.py),
[diagnostic evidence](checkpoint_fp16_long_postmortem.json).
