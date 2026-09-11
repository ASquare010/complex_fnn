# H107 reproduction and evidence

These are isolated experiment sources, not registered model variants. Read the
[frozen plan](../../../research/affine_residual_fit_plan.md) and the source hashes
in `../protocol.json` before attempting reproduction.

`model.py` contains eight models, the common teacher-row initializer, the common
FP64 ridge readout and normalization folding. `qualification.py` checks counts,
input/parameter finite differences, folding, teacher equivalence, shared prefixes
and the ridge solution. `study.py` runs the frozen fresh-window comparison using
the maintained CPU-data/CUDA-regression trainer. `launch.py` selects the qualified
UV-managed Python 3.12.9 runtime and existing CUDA package directory; it records
logs and exit codes and never retries a scientific run.

The completed experiment refuses to overwrite its protocol. Reproduce the
archived source in a separate workspace with its prerequisite teacher/data
artifacts, or specify a distinct follow-up protocol/output directory. Do not
delete a completed protocol to force a rerun. The archive identifies exactly the
four experiment Python files present at freezing; later audit/report helpers
are outside that source snapshot.

Post-run helpers:

- `audit.py`: independent full-decoder pair recapture, direct tensor forward
  algebra, all 864 endpoint scores, 144 ridge initializations, selections and raw
  exports. It does not use the experimental forward/folding/scoring helpers.
- `analyze.py`: all descriptive statistics, frozen gate decisions and the
  compressed table containing every endpoint, including initializations.
- `diagnostics.py`: compact loss/gradient trajectories, PReLU slopes and separately
  identified post-run deployment-memory profiles. It performs no training.
- `plot.py`: standalone Matplotlib figure, deliberately without a Torch import.
- `tests.py`: maintained test suite, after other GPU work has finished.
- `../../verification/affine_residual_final_v1.py`: frozen-hash/decision checks,
  lossless artifact packaging, link checks and final receipt.

Run helpers from the repository root using the recorded UV environment. For
example, the launcher takes `audit`, `analyze`, `diagnostics`, `plot` or `tests`
as an argument. Existing log names are exclusive: repeated checking should use
a separately named receipt, not overwrite the original audit history. There is
one GPU job at a time; CPU preprocessing and teacher capture are charged too.

Pair tensors, raw teacher checkpoints, normalized students, folded exports and
per-update histories stay local and ignored. Compact public metadata contains
their hashes but cannot recreate a missing tensor file. Restore `.json.gz`
metadata to a new plain file with exclusive creation when a historical source
expects `.json`; compressed evidence never deletes or changes the originals.
