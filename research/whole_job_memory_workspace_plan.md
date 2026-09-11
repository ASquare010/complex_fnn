# H112 boundary correction after a measured workspace diagnosis

The single-process continuation passed its eight-case qualification and stopped
before training because17,039,360 allocated bytes survived ordinary garbage
collection and empty_cache. The failure and frozen sources remain unchanged.
It performed zero model updates.

A separately frozen two-repeat diagnostic identifies two active blocks of
8,519,680 bytes each. Calling only `torch._C._cuda_clearCublasWorkspaces`, then
ordinary cleanup, returns allocated bytes to zero in both repetitions. This
establishes the source of that boundary allocation; it does not resolve the
separate intermittent Python native-import failures. The sizes agree with
the documented default cuBLAS workspace allocation. See the
[PyTorch CUDA notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html) and the
[diagnostic](../results/cublas_boundary_diagnosis_v1/result.json).

One bounded corrected continuation uses `results/whole_job_memory_workspace_v1`.
It imports the unchanged preceding coordinator and original H110 worker, changes
only the output root, and clears cuBLAS workspaces at the already-declared
between-trial boundaries. The zero-active-allocation assertion remains exact.
Record allocated bytes before and after workspace clearing at every boundary.
Any surviving model allocation still causes a failure. Do not subtract workspace
bytes from measurements and do not change workspace size/configuration. Each
trial recreates its workspaces normally; they count fully toward training,
validation and whole-job peaks. The existing20-update timing warmup is unchanged.

The two-corpus, three-seed, twelve-trial/9,600-update plan, native quality audit,
all numeric gates, checkpoint schedule and single-process limitations remain
unchanged. All trials are still never-trained. Freeze this note, wrappers and
diagnostic evidence before execution. Stop on any further runtime/state/science
failure; do not automatically repeat a trial. Prior failed roots are terminal
history and must not be overwritten. No architecture or benchmark claim follows
from a runtime correction alone.
