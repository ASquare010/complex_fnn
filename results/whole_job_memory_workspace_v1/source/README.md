# H112 source and evidence guide

The maintained FFN and training implementation are reused unchanged. Read
[the prospective plan](../../../research/whole_job_memory_plan.md) and
[complete results](../../../research/whole_job_memory_results.md) first.

- `../protocol.json` freezes 92 source dependencies, both corpus manifests,
  the selected H111 evaluator and prior failed-attempt evidence.
- `study.py` wraps the preserved single-process coordinator from
  `results/whole_job_memory_continuation_v1/source/study.py`. It clears only
  cuBLAS workspaces between trials, then enforces zero live tensor allocation.
  It does not subtract workspaces from measured memory.
- The original H112 `adapters.py` chooses the corpus/output location, checks
  the six-evaluation schedule and saves the additional step-200 checkpoint.
  The H110 worker still performs all optimizer updates and measurements.
- `prepare.py` freezes the workspace recovery before any training. `launch.py`
  uses the absolute UV-managed Python and exclusive per-stage output logs.
- `audit.py` independently replays initialization/sampling and native scores;
  `audit_recovery.py` changes dependency import order after the preserved
  startup failure. Neither performs optimization.
- `analyze.py` applies the prospective one-sided gates using native primary
  scores and exports all trials/statistics. `plot.py` shows every paired
  outcome; `plot_recovery.py` preserves the failed first launch and preloads
  dependencies before calling that unchanged plotter, without CUDA.
- `inspect_progress.py` only reads metadata. Postprocessing sources are
  distinguished from the original training freeze and recorded by hashes.
- `update_navigation.py` preserves prior documentation and publishes the
  completed results in the entry pages. It explicitly writes UTF-8 after a
  caught Windows default-encoding failure; the original README was restored
  from its verified copy before the corrected write. No scientific file changed.

The twelve models are independently initialized but share one process; CUDA
context/library state persists. Every trial starts after zero allocated and
reserved tensor storage is verified. There are 25 verified boundaries and
zero repeated scientific updates. Models, histories, checkpoints and original
logs remain local under the corpus/run folders. They are not deleted.

Compact public evidence includes `result.json.gz`, `metrics.csv.gz`,
`diagnostics.json.gz`, `summary.json`, `audit.json`, recovery manifests and
compressed original logs. Raw results decompress exactly from the gzip export.
The final verifier is `results/verification/whole_job_memory_final_v1.py`.
Running it rechecks existing evidence and regenerates the CPU-only receipt;
it does not launch training or overwrite checkpoints. The previous 116-test
maintained suite is unchanged, and isolated qualification counts remain separate.

Historical output roots refuse reuse. A new reproduction must choose a new root
and freeze its paths/protocol explicitly; do not edit old sources or rerun a
launcher against a completed stage. The rejected two-corpus recipe does not
receive automatic extra steps or tuning.
