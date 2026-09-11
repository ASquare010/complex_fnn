# H158 bounded startup recovery

Original run exited 1 after case00 completed. Worker1 exited 3221225477 with a
Windows access violation while importing torch (__init__.py:2611), before worker
build/model construction or its output files. case01, gradient01 and boundary01
are absent. The run handle is terminal and nvidia-smi lists no compute process.
The immediate cause is not established; do not attribute it to the candidate.

Preserve original logs, exit files, source and protocol. Retry only worker1 once,
then continue cases2-17. Never rerun completed case00. Use the prior H157 launcher
setting `-X pycache_prefix=<study>/unused_cache` for all remaining children; the
model, codec, data, numerical recipe and H158 gates are unchanged. The flag avoids
reading default-path bytecode caches; it is not asserted to cure the native crash.
A second worker1 failure ends this recovery. Any other failure also stops execution.
The audit's sole change is reading worker1_retry_exit.txt for case1. Original
worker1_exit.txt remains failed. Freeze recovery source and this note before retry.
No extra successful backward or optimizer update is authorized by this recovery.
