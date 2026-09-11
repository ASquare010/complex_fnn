# H112: post-training audit runtime recovery

All 12 training trials finished once, with study exit 0 and 9,600 updates.
The first separate native-audit child exited 3221225477 during Python module
loading. Its original `audit.log` and `audit_exit.txt` remain unchanged. No
native score or Python audit-failure record was produced by that attempt.
The launcher tool returned -1; the saved child exit is the more specific record.

One explicit recovery preloads SymPy, PyTorch and TorchDynamo in the order used
by the successful training coordinator before importing the unchanged audit.
It retains the same absolute UV-managed Python 3.12.9, installed packages,
malloc allocator, hash seed 107, bytecode bypass, four threads and TF32 settings.
The scoring implementation, saved models, splits, tolerances and native-primary
quality rule remain unchanged. No model is trained or resumed. The recovery
records the original source/result/failure hashes and writes exclusive new logs.

This is a bounded operational attempt, not proof that preload order caused or
fixes the intermittent native crash. A new failure must be preserved and
diagnosed before further attempts. Successful rescoring still requires all
72 native scores, 48 checkpoint checks and exact sampler/initialization audits.
