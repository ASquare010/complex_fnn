# CUDA reduction diagnosis and corrected zero-update export

18 October 2026, before the corrected gate or any optimization. Preserve the
original v1 export and failed bitwise assertion. Its full original data/initial
and ALL production alignment-packet audits passed; it has zero optimizer updates.

Eight repeated full-size genuine-input GPU passes diagnose the failure. Even
ordinary versus ordinary clipped gradients differ by3.725290e-9; coefficient-zero
versus ordinary differs by at most5.587935e-9. Loss and global norm are bit-exact
in every pass, and all weights/empty Adam remain exact. Thus a GPU bitwise-gradient
assertion is stronger than reproducibility of the ordinary calculation itself.
The independent CPU coefficient-zero objective/Adam test remains bit-exact.
Receipt: `dump/native256-source-occurrence-parity-20261018/report.json`.

The new v2 EXPORT changes only that gate: require maximum coefficient-zero GPU
gradient difference below1e-7, while keeping exact loss/norm, the microbatch gate,
full independent21-layer CPU objective/all-parameter gradients, causal/padding/
NULL tests and unchanged-weight/empty-Adam assertions. Record the actual error;
do not call it bit-exact. All math, coefficient, initialization, architecture,
original data/orders/schedule and inference/KV stay unchanged. New unique run and
format prevent resuming the failed export. This is not an architecture variant.

To avoid repeating the unchanged five-million-block audit, transfer its proof
only after checking every original input/cache/source SHA, exact old/new orders,
actual fresh tensors/RNG/empty Adam and the immutable full-audit receipt. Recheck
EVERY alignment segment with the same actual production batch implementation.
The v2 data receipt must explicitly identify the inherited v1 audit and distinguish
it from the new packet checks. Preserve and hash every v1 file and the diagnosis.
Independent full-model CPU/GPU verification checks all protected earlier files
again before the first real update. No student quality inference follows from
these numerical gates.
