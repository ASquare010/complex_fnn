# H098: explicit recovery of H096 after native termination

H096's worker PID4760 terminated with exit-1073741819 (0xC0000005). Its actual
session is terminal and the PID is absent; the unmodified worker status remains
RUNNING because a native crash bypassed Python's finalizer. The source archive
and frozen before/protocol exist, but no completed fit or parity records exist.
An external termination receipt records the observed exit instead of rewriting
the stale original status. No neural optimizer update was scheduled or executed.
The precise crash operation and cause are unknown.

H097 makes one separately frozen385-dimensional positive-definite matrix probe.
CPU and CUDA eigvalsh, Cholesky and solves all pass and agree within1e-10.
That failure to reproduce does not establish that the original run was correct
or diagnose the crash. Roughly17 GiB physical memory was free after termination;
that after-the-fact observation does not prove peak memory was harmless.

This is one explicit recovery with a new output root. Reuse H096's mathematics,
data, sample counts,24 fits,168 parity evaluations, tolerances, audit and decision
rules. Move only the small Gram/cross-product matrices to CPU before eigvalsh,
Cholesky and the solve. GPU FP64 sufficient-statistic products and FP32/FP64
evaluation stay unchanged. Add flushed phase labels and launch with faulthandler.
The backend change is a conservative execution choice, not a proven crash fix.
There is no tolerance relaxation, ridge tuning, additional sample or rate search.

Freeze this recovery source and plan along with H096's original sources and
H097's probe, before execution. Preserve all original H096 files by hash and
size. Stop on a new failure; do not automatically retry. Independent CPU
square-root-weighted statistics and direct solves must still reproduce the
coefficients/scores within the original stated tolerances. Original checkpoints
and historical results are not modified. Gold remains unmet.
