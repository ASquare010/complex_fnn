# H119 recovery — continue the six unexecuted GPU cases

The original stage exited 1 at the assertion that CUDA remained uninitialized
after six CPU AdamW steps. All six CPU result records and 12 tensor artifacts
exist and are complete; no CUDA case directory was created. The failure log,
traceback, exit status, original sources and protocol remain unchanged.

The source-side cause of CUDA initialization has not been instrumentally
isolated. CPU parameter arithmetic does not imply the library leaves CUDA
uninitialized. Do not label this a numerical failure or a runtime cure.

Freeze these completed CPU artifacts and this continuation before further
work. Import the original `run_case` unchanged into a fresh process. Verify
all original inputs/sources and the six CPU records, then perform only the
six originally allocated CUDA cases. Require zero CUDA tensor allocation and
reservation before and after each case (seven boundaries). Combine those
results with the preserved CPU records. Total across attempts stays exactly
12 disposable optimizer steps, zero language-training updates, zero forwards,
zero backwards. No CPU case is repeated and no tolerance, seed, equation,
optimizer, saved gradient or numerical decision rule changes.

Original `prepare_audit.py` expects a successful original study exit. Preserve
it; use a separately frozen preparation script that instead verifies the
original failure and successful continuation, then freezes all 24 artifacts
and the unchanged primary analysis for the unchanged NumPy auditor.

Single-case timings survive in all records. The original failed stage did not
write an overall wall time; report that as unknown. Continuation wall time is
recorded separately. CPU-phase CUDA allocator peaks were not recorded and must
not be fabricated. Neither stage supplies whole-language-job resource evidence.
