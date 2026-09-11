# H111 source guide

`evaluation.py` implements three no-grad evaluators with unchanged attention
context and valid-target weighting. `qualification.py` checks tiny double
precision/masked cases. `prepare.py` freezes maintained files, five initial
scientific helpers, the plan and every existing checkpoint hash before scoring.
`study.py` evaluates all 24 saved states and measures full evaluation allocation
with explicit optimizer/zero-gradient fixtures. No neural weights are learned.

The later `analyze.py` applies the frozen gates and deterministic selection;
`audit.py` independently reloads all states, re-scores native forward and checks
all ratios/gates/selection. `inspect_progress.py` is a read-only preview helper,
not an allocator. `launch.py` uses exclusive logs and the absolute UV-managed
runtime, including the recorded bytecode-bypass settings. Historical folders
must not be overwritten. A new reproduction requires a new output root and
manifest, with its budget and fixture limitations recorded.

See [the plan](../../../research/streamed_evaluation_plan.md),
[the report](../../../research/streamed_evaluation_results.md),
[summary](../summary.json) and [audit](../audit.json). Raw complete records remain
in `../result.json`; the lossless public export is `../result.json.gz`. H110
checkpoints are referenced by hash in their original folders and are not copied
or deleted. Maintained source/defaults do not change. A composed old-training/
new-evaluation peak estimate is explicitly not an actual whole-job measurement.
