# H110 runtime continuation: bytecode bypass

This is an execution recovery, registered before any candidate training in this
continuation. The original scientific plan and worker remain unchanged. No model
or optimizer restart selects a better result: the original completed control is
reused byte-for-byte, with its exact checkpoint/stream audit preserved.

## Evidence and bounded choice

The original loss-chunk worker crashed in SymPy import during AdamW construction,
before training. A separately recorded CPU-preload attempt passed two dependency
probes but then crashed importing SymPy before importing Torch or initializing
CUDA. Both failures remain in their own roots, with zero candidate updates.

A twelve-process CPU factorial tested UV Python3.12.9/3.12.12 and existing/bypassed
bytecode, three repetitions per condition. Counts of successful imports were
1/3, 3/3, 2/3 and 2/3 respectively. All 1,572 checked SymPy source-file hashes
match the package RECORD; no missing or mismatched hashed source was found.
Neither source corruption nor cached-bytecode corruption is established. Three
successes are too few to establish runtime reliability. The root cause remains
unknown; native access violations and one generated-Python SyntaxError are
preserved, including failures under bytecode bypass on Python3.12.12.

Use the original UV Python3.12.9 with `-B -X pycache_prefix=<absent path>` for one
explicit continuation. This bypasses existing bytecode and writes no new cache;
it does not alter source files, the installed environment or existing caches.
Preload SymPy and Torch compiler dependencies before CUDA allocation. First
repeat the frozen eight tiny qualification cases in an isolated process. If it
passes, run only the eleven never-trained original trials, in their original
order, each in a fresh process. Reuse the completed original control. Stop on
the first runtime failure; there is no retry loop. Retain the frozen early-stop
rule for any paired NLL failure. Do not relax any scientific gate.

Freeze this plan, the continuation wrappers, original protocol and completed
control, failed attempts and CPU diagnostic evidence before launching. Record
the bootstrap difference per result. CPU import/compilation overhead is outside
the already defined synchronized update timer but inside continuation wall time.
Any timing comparison involving the original control must disclose the temporal
separation and bootstrap difference. This is single-device development evidence.

Outputs: `results/token_memory_duration_fresh_v1`. The historical failure roots
and `results/token_memory_duration_v1/partial_result.json` remain untouched.
No parameter, inference, novelty or breakthrough claim follows from a recovery.
