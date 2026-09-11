# H120 source guide

Read the [plan](../../../research/optimizer_memory_plan.md) and
[report](../../../research/optimizer_memory_results.md). The experiment tests
existing native optimizer implementations with one common profiling loop.
It does not add a model variant or change the maintained trainer.

| Stage | Script | Work |
|---|---|---|
| Freeze | `prepare.py` | Previous receipts, code/library/data/checkpoint hashes and navigation snapshots |
| Profile | `profile.py` | Four corpus/loss fixtures × three AdamW modes × 30 updates; all phase peaks and live-storage families |
| Audit freeze | `prepare_audit.py` | Pin all 36 new tensor artifacts and recorded profiles |
| Audit | `audit.py` | NumPy general-step AdamW checks, exact batch/sampler replay and native final scoring |
| Analyze | `analyze.py` | Fixed per-fixture gates, compact tables and machine-readable decisions |
| Present | `plot.py`, `report.py`, `publish.py` | Audited figure/report and current-state navigation |

`launch.py` uses the recorded UV-managed Python 3.12.9 interpreter with the
existing `.venv/Lib/site-packages`; no package resolution or dependency edit
occurs. It creates exclusive stage logs and preserves exit status. Execute
modules from the repository root. Existing outputs intentionally prevent a
rerun from replacing evidence; a new run needs a separately frozen directory.

No scientific stage failed and no case was repeated. A later read-only shell
display crashed; [the observation note](../observation_failure.txt) records it.
The report received an editorial clarification that the first-step clipping
checks were inactive (norm<1); no measurement, tolerance or decision changed.

Scientific budget: 360 training updates/backwards and 1,474,560 targets;
24 study validation scores and 12 independent native scores. Audit has zero
optimizer steps/backwards. Both alternatives fail whole-job memory savings,
despite lower optimizer-phase allocation. All numerical/data/scoring checks pass.

Raw `.pt` files, detailed JSON and old navigation bytes stay local; compact
tables, reports, protocols, logs and receipts are retained for Git. A fresh
clone without raw tensors cannot independently rescore them. Hashes identify
the original artifacts but do not reconstruct them.

The [final verifier](../../verification/optimizer_memory_final_v1.py) checks
frozen inputs, accounting, all stored gate outcomes and prior-evidence integrity.
It does not turn a short single-seed profile into a broad quality qualification.
