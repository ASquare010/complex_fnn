# H118 source and evidence

[Plan](../../../research/training_variability_plan.md),
[results](../../../research/training_variability_results.md).
This is a conditional repeat diagnosis of H117's failed WikiText seed101, with
two additional executions per FP32 policy. It introduces no new model and
retains the original two trajectories rather than replacing their outcome.

| File | Role |
|---|---|
| `prepare.py` | Verify H117, its 78 tensors and maintained source; freeze input/data/source hashes |
| `study.py` | Call H117's unchanged common `run_case` four times; bind and restore only its output directory |
| `launch.py` | One exclusive UV-managed stage with exclusive log creation and an explicit exit record |
| `inspect_progress.py` | Read-only status and log tails |
| `prepare_audit.py`, `audit.py` | Freeze new artifacts, reuse independent H117 checks, compare every initial/checkpoint pair |
| `analyze.py` | All six trajectories, repeat statistics and unchanged conditional decision rules |
| `plot.py`, `report.py` | All-repeat static figure and result-derived documentation |
| `publish.py` | Verify pre-study navigation before updating current results and artifact rules |

The six initial scientific sources and plan, inherited sources, and H117's
stdlib verification helper produce **137 frozen hashes**. Do not edit frozen
files or rerun preparation/training inside this root. Study code imports one
existing runner; it does not copy training, change arithmetic or overwrite old
results. The output-root assignment is scoped and restored in `finally`.

All runs load the same H117 step-zero model/empty Adam state and use the same
training seed101 and sampler seed10101. The new runs are separate executions,
not new seeds. Source labels and output paths distinguish repetitions1/2.
The original policy pair is repetition0. Its already completed audit is reused
as prior evidence, not counted as a new audit.

The independent new audit verifies 12 trained states, four FP32 initial-gradient
backward replays, 13 native full-validation scores, all3,200 new batches and one
regenerated initial state. It also checks all six initial-gradient hashes,
15 initial-gradient pair distances and45 checkpoint pairs (model plus both Adam
moments). Distances use CPU double precision and the symmetric denominator
specified in the plan. CPU sanity checks cover identity/zero/symmetry cases.

New budget: 3,200 updates, 13,107,200 target presentations, 3,204 main backwards
including initial probes, and four audit backwards =3,208 new backwards. The
original pair contributes1,600 historical updates separately. The unchanged
loss adapter's H117 CPU qualification is reused, with no new qualification
backwards. No scientific or postprocessing retry occurred in this study.

The UV-managed Python3.12.9 runtime uses existing `.venv` dependencies, four CPU
threads, FP32 arithmetic and TF32 disabled. Preloads precede CUDA initialization.
Only one GPU stage runs at a time; the stage process must terminate before its
dependent stage. Successful execution does not establish a cure for historical
Windows native failures, which remain preserved in prior study roots.

Compact evidence includes protocols, environment, summary, compressed result,
audit and logs, six metric rows, 24 convergence rows and4,800 update rows with
explicit H117/H118 origin labels. Only3,200 update rows are newly executed.
Local `runs/` retains four initial-gradient files, 12 trained checkpoints,
histories and metrics. Original tensors remain in the H117 root. Raw aggregate
JSON/logs and navigation snapshots are ignored rather than deleted. A compact
clone needs the original/new tensors and dataset cache for full replay.

The final verifier `results/verification/training_variability_final_v1.py`
reuses H117's full case-accounting verifier and independently recomputes the
conditional classification. A PASS receipt means evidence integrity passed;
the scientific verdict remains INCONCLUSIVE, and H117's failed gate stays failed.
All61 maintained files and the two-model/five-variant/eight-recipe tree remain
unchanged. The historical116-test pass is not presented as a new test run.
