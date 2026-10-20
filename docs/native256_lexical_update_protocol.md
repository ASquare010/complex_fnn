# Remove redundant GPU synchronization from the optimizer body

18October2026, before any actual full-run update. Preparedv1 remains zero-update
and immutable; NEWv2 reuses the exact model/data/48762-update/two-original-epoch
recipe through a scoped runner. No architecture, loss, targets, clipping, optimizer,
initialization or sampler change. The sole change removes per-parameter CUDA
`isfinite(...).all()` synchronization; a CPU-only grad-present assertion remains.
The SAME `clip_grad_norm_(...,1,error_if_nonfinite=True)` computes the full norm
and prevents Adam on any NaN/Inf/overflow. No finite safety check is disabled.

CPU tests must show exact normal gradients/Adam state/weights and serialized
resume parity against the original updater, and explicit NaN/Inf rejection with
zero optimizer steps/unchanged weights. The wrapper redirects only process-local
output root/update/source callbacks to newv2 and restores them on exit; no old
source/export/run file changes. Freshv2 initial tensors and all original data/
metadata/order audits/preflights/first-update checks are repeated. The original
source archive remains available. All new sources stay frozen while live. No
speed or quality claim until measured, and no additional training epochs.
