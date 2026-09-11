# H110 recorded runtime continuation

This directory contains wrappers around the original frozen H110 scientific
worker. It introduces no model implementation, activation, optimizer, data
split, tolerance or precision change. The
[continuation plan](../../../research/token_memory_duration_recovery_plan.md)
precedes its qualification and training. `before.json` hashes the successful
original control, original protocol, both failed attempts, the CPU diagnostic,
the continuation plan and five bootstrap scripts before execution.

`prepare.py` verifies and preserves that evidence. `build_coordinator.py` records
how the preceding failed recovery coordinator was adapted before freezing.
`launch.py` invokes UV Python 3.12.9. `study.py` launches each child with bytecode
bypass (`-B` and an unused cache prefix), first repeating the eight frozen
qualification cases. `worker.py` preloads CPU dependencies, asserts CUDA is
uninitialized, and invokes the unchanged original worker. The first successful
control is reused from its original folder; only never-trained trials run here.
The first runtime or frozen fidelity failure stops further allocation.

After training, `audit.py` imports the independent native, unchunked checkpoint
audit. `analyze.py` imports the original descriptive statistics and frozen gate
calculation. `plot.py` presents all completed runs and separates training from
whole-job allocation. `launch_postprocess.py` records one exclusive invocation
per stage, retaining every failed exit if one occurs. These postprocessing
scripts are not represented as preregistered scientific code.

The first outer `uv` dispatch for the final audit panicked before starting
Python. Its separate failure record is retained. `launch_direct_postprocess.py`
then invokes the identical absolute UV-managed interpreter for audit/plotting,
with unchanged environment and cache-bypass flags. This is a launcher workaround,
not a new scientific trial or a proved runtime fix.

The [result report](../../../research/token_memory_duration_results.md) contains
decisions, per-seed metrics, limits and any remaining work. The
[final receipt](../../verification/token_memory_duration_final_v1.json) checks
preservation, exact checkpoint rescoring, sampler replay, independent summary/
gate arithmetic and lossless packing. A passing receipt certifies those checks,
not attainment of the broad research goal or resolution of runtime failures.

Compact numeric evidence is in `../result.json.gz`, `../summary.json`,
`../metrics.csv.gz`, `../diagnostics.json.gz` and `../audit.json`. Full tensors and
histories remain local in the original and continuation `runs/` directories;
no source or failed trial is deleted. The original8-case qualification and
continuation8-case qualification are separate from the maintained test suite's
previous 116-test pass, which is not rerun for unchanged maintained code.

Historical output folders are exclusive. To reproduce training, create a new
named output root and wrapper around the same worker, record that as a new
experiment, and freeze its source/data manifest before launching. Do not
overwrite historical evidence or treat an existing successful trial as a new
replication. The active repository remains two model folders, five variants
and eight recipes.
