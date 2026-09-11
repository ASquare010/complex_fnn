# H108: derivative-aware selection and dense readout

This isolated screen tests whether teacher-neuron selection and a derivative-aware
ridge readout preserve a trained FFN's values and local sensitivities at lower
width. It creates 432 ordinary dense GELU/SwiGLU exports, with zero SGD updates.
The maintained model factory and tests are unchanged.

Read the [frozen plan](../../../research/sobolev_selection_plan.md) before running
anything; the [report](../../../research/sobolev_selection_results.md) gives the
positive component result, failed complete-compression gates and limitations.

## Frozen scientific sources

`protocol.json` records eight SHA-256 hashes: `features.py`, `selection.py`,
`qualification.py`, `launch.py`, `study.py`, the plan, and two maintained imports.
Those files must remain unchanged when auditing this result. `source.zip` is a
local snapshot. The protocol also records the runtime, GPU and repository state.

- `features.py`: smooth FP32 teacher features/JVPs, streamed sufficient statistics,
  fixed budgets and export with normalization folded into the output weights/bias.
- `selection.py`: random, weight, fluctuation, conditional variance and exact
  one-step regularized greedy-gain orders; selected ridge readouts.
- `qualification.py`: 13 groups covering autograd/finite differences, counts,
  export/input gradients, exhaustive small greedy fixtures and probe isotropy.
- `study.py`: 18 input sets, eight selector/readout methods and three widths;
  432 exports, held-out local errors and native inference resource measurements.
- `launch.py`: one-process runtime setup, exclusive stage logs and exit receipts.

The eight methods include value and Sobolev greedy selection/fitting plus both
cross-combinations. A fixed output bias is learned by unpenalized intercept
compensation; it counts toward parameters. The exported model has no JVP or
selection operation at inference. All weights remain trainable for future work.

## Replay and dependencies

Use the repository root, UV-managed Python 3.12.9 and the recorded PyTorch CUDA
environment. The original Windows launch was:

```text
uv run --no-project --python C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe python results/sobolev_selection_v1/source/launch.py study
```

The launcher uses the existing `.venv/Lib/site-packages`, four CPU threads,
`PYTHONMALLOC=malloc` and `PYTHONHASHSEED=107`. TF32 is disabled. One GPU job runs
at a time. These environment settings are recorded, not a proved remedy for
earlier native Windows failures. Existing stage logs are opened exclusively:
do not overwrite or blindly rerun this completed study. A scientific replication
needs a fresh output root and its own frozen protocol, preserving this run.

Required local inputs are both full seed-17 teacher checkpoints under
`results/ungated_duration_v1/runs/` and all 18 H107 `pairs/*/data.pt` files under
`results/affine_residual_fit_v1/`. The screen uses the first 8,192 and last 4,096
inputs only, recomputing smooth FP32 outputs; it does not use H107 BF16 labels.
Metadata and hashes are retained publicly; they cannot substitute for the local
tensor files. A clean clone without those inputs cannot replay tensor audits.

## Post-screen helpers and preserved audit failure

`analyze.py` computes all methods' means, medians, variances, seed/depth summaries,
paired decisions and `metrics.csv.gz`. `ablation.py` attributes the predeclared
crossed comparisons; its fractions are descriptive. `plot.py` and `report.py`
produce the figure and report. They do not change thresholds or train models.

`audit.py` independently rebuilds statistics with autograd JVPs, checks every
export/readout, rescores all 864 metrics and tests 432 sampled real greedy gains.
Its original failure is retained: changing teacher GEMM batch size from 512 to
257 caused one near-zero output to miss the elementwise tolerance by 8.6e-8.
`audit_recovery.py` restores only teacher-reference batching to 512. Students
remain rescored at 257 with automatic JVPs; every tolerance is unchanged.
The complete recovery passes. `audit_failure.json`, `audit_recovery_before.json`
and compressed original/recovery logs preserve that sequence. No original model,
selection, frozen source or recorded metric was replaced.

The [final verifier](../../verification/sobolev_selection_final_v1.py) independently
checks frozen hashes, summary statistics, all decisions, ablation arithmetic,
CSV consistency, preserved failure provenance and lossless evidence packaging.
It uses only the standard library. Qualification/audit receipts are separate
from the unchanged maintained suite's previous 116-test pass.

## Decision

Retain derivative-aware calibration as a promising component. All complete
recipes remain unqualified for language insertion. There is no language-model,
training-memory, convergence, universal-gradient or novelty claim. The next
learning experiment requires a distinct, preregistered protocol on fresh data.
