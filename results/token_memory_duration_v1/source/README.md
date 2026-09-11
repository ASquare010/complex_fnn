# H110 original duration experiment

This is an isolated research harness using the maintained Transformer and the
unchanged H101 execution adapter. It adds no registered model, recipe or default.
The [scientific plan](../../../research/token_memory_duration_plan.md) defines
all 800-step comparisons and acceptance thresholds before execution.

`common.py` holds fixed configurations, native unchunked validation and the
eight small loss/gradient/state qualification cases. `worker.py` owns one fresh
trial, with exclusive output creation, complete histories and 100/400/800-update
checkpoints. `study.py` freezes source/data provenance and launches the grid.
`launch.py` records UV execution, logs and terminal exit codes.

The original study is terminally interrupted after one completed control.
Its first loss-chunk candidate failed during a dependency import before
training. `partial_result.json` explicitly records one control, zero candidate
updates and zero pairs. `collect_partial.py` creates that posthoc view without
rewriting the original study outcome. `audit.py`, launched by
`launch_fresh_audit.py`, independently verifies the completed control with no
optimization. All five NLL checks are exact;800 input batches replay exactly.

`analyze.py` is the posthoc implementation of the frozen arithmetic and gates.
The final continuation imports it without modifying its contents. The initial
scientific source manifest excludes later audit/postprocessing helpers; the
continuation preservation manifest additionally freezes their then-current
bytes. This distinction is deliberate.

The [continuation source](../../token_memory_duration_fresh_v1/source/README.md)
and [duration report](../../../research/token_memory_duration_results.md) explain
the completed scientific evidence. The [runtime diagnostic](../../../research/runtime_import_diagnosis.md)
preserves both initial native failures and the CPU-only investigation. No
successful scientific trial is silently repeated or overwritten.

All paths above are relative to this file. Raw model/optimizer tensors and
per-update histories remain local under the corresponding `runs/` folder;
the complete lossless result archive retains hashes and numeric records. Do
not rerun historical launch commands in an occupied output folder: they use
exclusive creation intentionally. A new reproduction needs a new output root,
a separately frozen manifest and an explicitly recorded compute budget.
