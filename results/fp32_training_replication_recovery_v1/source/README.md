# H117 source and evidence

Read the [prospective plan](../../../research/fp32_training_replication_plan.md),
[startup recovery plan](../../../research/fp32_training_replication_recovery_plan.md)
and [complete results](../../../research/fp32_training_replication_results.md).
This isolated study changes execution precision and classifier memory use. All
arms share the maintained Transformer, optimizer grouping, data and evaluator.
The maintained model tree and defaults remain unchanged.

| File | Purpose |
|---|---|
| `build.py` | Prospective recovery of initialization order from the frozen original runner |
| `prepare.py` | Verify the zero-training failure and freeze recovery source, original evidence and data |
| `study.py` | One common 800-update fresh-training loop for all three policies |
| `launch.py` | Exclusive UV-managed child, exclusive log creation and explicit exit records |
| `audit.py`, `prepare_audit.py` | Regenerate initial states, replay FP32 gradients, score all saved checkpoints, verify streams and states |
| `build_analysis.py` | Explicit construction of the post-study analyzer from H116; no model execution |
| `analyze.py` | Every seed, descriptive statistics, fixed all-seed gates and raw Pareto comparisons |
| `plot.py`, `write_report.py` | Complete-seed figure and result-derived tables/report |
| `prepare_plot_recovery.py`, `plot_recovery.py` | Preserve the failed Matplotlib import and run the unchanged plot in one fresh CPU process |
| `inspect_progress.py`, `live_pairs.py` | Read-only monitoring; incomplete runs never imply qualification |
| `publish.py` | Verify original navigation hashes before publishing the final result |
| `normalize_navigation.py` | Correct publication line endings and compact-artifact allow rules after checking the preliminary receipt |

The original attempt is preserved at `results/fp32_training_replication_v1`.
Its frozen initializer supplies six CPU step-zero states with empty Adam state.
The original runner queried CUDA environment metadata before the initializer's
CPU guard and failed before any checkpoint/training batch/update. Recovery moves
initialization before metadata and qualification, with no scientific recipe or
decision change. The original attempt used 24 CPU qualification backwards.
Its later unexecuted audit/inspection drafts remain historical; they are not
evidence of an audit run. Original scientific source is unchanged.

Recovery `protocol.json` freezes 129 source/plan files, including seven recovery
sources and the recovery plan, plus all 61 maintained hashes. Do not edit frozen
files or overwrite either output root. Its on-disk `checkpoint_hashes` is empty
because initial states are created after freezing; `initializations.json` records
their hashes and the runner uses those hashes in memory. All runs start at step0.
The reused final-checkpoint field `continuation_steps=800` is not a claim that
these runs resumed a trained model.

The independent audit freezes 85 inputs: its code, protocol/results/qualification,
initialization manifest, a frozen H116 audit helper and 78 tensor artifacts.
It regenerates six initial states, verifies 54 trained checkpoints and all
14,400 batch samples, scores 60 complete native validation states, and runs
12 FP32 initial-gradient backward replays. BF16 replay and complete 800-update
trajectory reproducibility are outside this audit's claims.

The launcher uses UV's Python 3.12.9 runtime with the existing `.venv` packages.
PyTorch/SymPy preloads intentionally precede CUDA initialization. Scientific
runs use four CPU threads, TF32 disabled and one GPU worker. These choices are
not an established fix for earlier intermittent Windows native failures.
Always wait for a launcher process to terminate before its dependent stage.

The first plot process failed during Matplotlib import with raw exit 3221225477,
before producing any figure. Its source, failure and scientific inputs are
frozen in `plot_recovery_protocol.json`. One fresh CPU process imported the
unchanged plot successfully; the resulting figure was visually inspected.
No training, audit or analysis was repeated, and the native cause is unresolved.

Do not rerun preparation or scientific stages in these completed directories.
A reproduction needs a new frozen protocol/output root. The stages originally
executed in order were preparation, study, audit preparation, audit, analysis,
plotting, report generation, navigation publication and final stdlib verification.
The final verifier is
`results/verification/fp32_training_replication_final_v1.py`.

Public evidence includes compact protocols, environment/qualification, summary,
result/audit JSON gzip, compressed logs and tables with 18 cases, 14,400 updates
and 72 validation points. Local `initial/` and `runs/` retain six initial states,
54 trained model/Adam/sampler checkpoints and 18 initial-gradient files. Raw
histories/metrics, original navigation snapshots, datasets and raw aggregate
JSON/logs remain local and ignored, without deletion. Tensor replay requires
those local artifacts; a compact clone alone is insufficient.

Accounting across both attempts: 14,400 optimizer updates / 58,982,400 target
presentations; 14,418 main backwards, 48 CPU qualification backwards and
12 audit backwards = 14,478 backwards. All 28,998 job-memory intervals are
recorded. No maintained default or new architecture is introduced, and the
previous 116-test pass remains historical, not a new test run.
