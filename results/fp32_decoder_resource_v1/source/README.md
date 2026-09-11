# H116 source and evidence

See the [prospective plan](../../../research/fp32_decoder_resource_plan.md) and
[results](../../../research/fp32_decoder_resource_results.md). All five policies
and all model fixtures use one shared study runner. Maintained architectures,
data, optimizer grouping, evaluator and diagnostics remain common and unchanged.

| File | Purpose |
|---|---|
| `common.py` | Execution modes and loss adapter; complete CPU-double qualification |
| `build_study.py` | Prospective, explicit derivation from the frozen H114 runner |
| `study.py` | One 50-update full-job loop for every architecture/policy |
| `prepare.py` | Verify H115 and freeze source, checkpoints, data and document snapshots |
| `launch.py` | Exclusive UV-managed worker, immutable attempt logs and exit records |
| `audit.py`, `prepare_audit.py` | Independent native scoring, FP32 gradient replays and stream/state verification |
| `analyze.py` | Statistics, original gates and raw Pareto comparisons |
| `analyze_recovery.py`, `prepare_analysis_recovery.py` | Preserve failed CPU analysis and invoke the unchanged analyzer in a fresh process |
| `plot.py` | Static all-fixture figure |
| `report_values.py`, `inspect_results.py`, `inspect_progress.py` | Read-only result/status inspection |
| `publish.py` | Check pre-study document hashes before updating current navigation |

The first five source files and plan are prospectively frozen by `protocol.json`
alongside inherited source hashes: 115 total. The audit has its own later freeze
covering code, result, qualification and all 60 tensor artifacts. Analysis
recovery additionally freezes its inputs and sources. Do not edit those files
or overwrite this study root. A scientific follow-up requires a new prospective
protocol and output directory.

The launcher uses UV's Python 3.12.9 interpreter and repository environment
packages, four CPU threads, TF32 disabled and one GPU process. Library preloads
precede other study imports intentionally. Lint exempts `study.py:E402` for that
ordering; no scientific source is changed to move imports after its freeze.
These runtime choices do not establish a fix for intermittent Windows native
failures. The original CPU-only analyzer access violation is preserved in
`analyze.log.gz`/`analyze_exit.txt`, and the successful unchanged-code retry in
`analyze_recovery.log.gz` and `analysis_recovery_protocol.json`.

Public compact evidence includes result/audit JSON gzip, summary, protocols,
qualification, environment, compressed logs, exit codes, 30 condition rows and
1,500 update rows. Local `runs/` retains 30 final model/Adam checkpoints, 30
initial gradient tensor files, histories and per-case metrics. Raw aggregate
JSON/logs and `before_documents/` stay local and ignored. A compact Git clone
cannot repeat tensor/data verification without these local artifacts.

The stdlib verifier
`results/verification/fp32_decoder_resource_final_v1.py` checks frozen files,
memory/count ledgers, complete table mappings, audit scope, gate arithmetic and
lossless exports without additional GPU work. The receipt preserves the earlier
116-test suite result as historical; that maintained suite was not rerun.

Budget: 1,530 main backwards / 1,500 updates; 24 CPU qualification backwards;
24 independent FP32 replay backwards and 36 native scores. Total 1,578 backwards.
BF16 reference replay is not part of this gate. The outcome earns broader
replication only for default FP32 chunks on the two narrow context-512 scopes,
not a new architecture, default change or completed broad research goal.
