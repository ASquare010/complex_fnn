# Scratch allocation correction to the span zero-update gate

18October2026, before the corrected GPU run. Original span source/model/math and
its six CPU tests/full21-layer CPU reference proof remain unchanged. The first
GPU worker passed cached/actual64 rollover, microbatch/future gradients and all
loaded timings, then intentionally failed the ZERO live allocation assertion
before isolated body profiling. Exactly64MiB remained; no training occurred.
Preserve its export, including failure receipt, and do not call it a passed gate.

A separate forward/backward diagnostic released every model/input/graph, then
observed64MiB. Clearing ONLY PyTorch cuBLAS workspaces reduced allocated bytes to
ZERO. cuBLAS configured32MiB/handle and cuBLASLt1MiB settings remained unchanged;
weights were exact and no optimizer update occurred. Installed torch exposes
the cleanup in _C/__init__.pyi and ATen/cuda/CUDAContextLight.h. No allocator or
numerical model settings are changed.

The corrected wrapper runs the unchanged gate into a NEW v2 export, protecting
the entire failed export and diagnosis. Before each isolated body: synchronize,
collect Python objects, clear library workspaces, empty the allocator cache, and
retain the original exact-zero allocation assertion. A retained model would
still fail. The first body call recreates normal library scratch and the measured
peak includes it. First body timing is a warmup. No extra GPU job may run during
timing. Independent verifier replays actual mean-embedding equations, all saved
gradients, public/cached/rollover outputs, bytes and medians; it also checks the
diagnosis, unchanged workspace configuration and prior-export hashes.
