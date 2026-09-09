# H069 logging correction after terminal BF16 hashing failure

The [original diagnosis](factor_balance_plan.md) passed four mathematical/software
checks and one CPU paired case (eight exact tensors), then returned code 1 on
its first CUDA case: NumPy cannot directly represent torch.bfloat16. The traceback
is in [the retained log](../results/factor_balance_v1/diagnosis.log). The worker
and coordinator are absent. No checkpoint state summary or final result exists;
there are no optimizer updates or corpus targets. This is an audit serialization
bug, not a failed numerical balancing or scientific gate.

The original plan prohibits automatic retries. This explicit one-time corrective
attempt preserves the failed source, log, process record and CPU tensors. Change
only the hash helper to hash canonical contiguous raw uint8 bytes, retaining its
original dtype/shape header, and override the output root to factor_balance_v2.
The old diagnosis module, transform, exponents, gates, checkpoints, shapes,
initialization, precision and budget remain byte-identical on disk. The runtime
helper replacement is explicit in the new source snapshot. No model/optimizer
function is patched. This correction does not relax any equality tolerance.

Freeze the correction and source snapshot, then run four logging checks: known
BF16 IEEE bytes (including signed zero), compatibility with the original helper
for FP32/FP64/int64, scalar/empty/noncontiguous tensors, and identical CPU/CUDA
BF16 hashes. Do not rerun the four unchanged original mathematical tests. If
logging checks pass, run the original fixed 216-record/24-pair diagnosis once
with the corrected logger. Compare the repeated first CPU case to its saved
original hashes/tensors. Preserve every outcome. No further retry is automatic.

Both attempts remain zero-update diagnoses. The corrected attempt completes the
original 24-pair allocation if successful; include the first attempt's one saved
CPU pair and uncompleted first CUDA case separately when accounting for work.
Use a hidden coordinator, one GPU worker, UV's existing environment and extras.
