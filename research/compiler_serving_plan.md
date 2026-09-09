# Compiler fusion probe after the trained scale result

Frozen before compiler installation or execution. At width384/layer8, seed17,
800 steps, BlockShuffle passes the local quality and training-memory gates.
Its paired factorized speed is .752 eager and .772 graph relative to full
SwiGLU. Packed eager reaches .802 but packed graph is .706. This motivates an
execution probe; it does not establish architectural novelty or robust speed.

## Environment and fixed treatment

Add triton-windows==3.8.0.post28 as an optional `compile` UV extra restricted to
Windows. Keep torch2.14.0+cu132 and the rest of the locked environment unchanged.
Primary package metadata and the local Triton minor requirement are recorded in
[the compatibility audit](compiler_backend_audit.md). Installation does not make
compatibility a proven fact; compile failures must be saved.

Use the completed width384/layer8 seed17 800-step factorized BlockShuffle and
full SwiGLU checkpoints. First test torch.compile with backend=inductor,
mode=default, fullgraph=True, dynamic=False, under no_grad and CUDA BF16.
No architecture, learned weights, fixed validation targets, matrix arithmetic,
or precision setting changes. Do not use packed or dense-cache candidates in
this first compiler comparison. The full reference receives the same compiler.

Compilation and warmup are excluded from steady-state timing and separately
recorded. Use a workspace-local ignored compiler cache. Keep a single GPU job
at a time and do not mix compiler profiling with the frozen serving audits.
A failed fullgraph compile is a recorded failure, not an eager fallback.

## Correctness and acceptance

Before timing, compare each compiled model with its own eager checkpoint on
three distinct fixed validation inputs. Record max absolute and RMS logit
errors; require all outputs finite, RMS logit error <=.01 and max error <=.15.
Those explicit BF16 tolerances are not bitwise equivalence. Evaluate all 32768
fixed validation targets and require |compiled NLL - eager NLL| <=.001 for
each model. Recheck with fresh inputs after warmup to catch static-input errors.
The compiled model's actual NLL is authoritative for its speed/quality claim.

Use six rotating timing rounds, four warmups and 15 repeated full-sequence
forwards per model per round, with CUDA synchronization. Report paired ratio
samples, median and range, plus per-model separately measured peak allocation.
Both compiled models may be co-resident for timing only; do not label that
process allocation as isolated model memory. Report compile time and any
persistent cache/storage without treating disk compiler artifacts as weights.

Retain a compiler path only if both models pass numerical checks, compiled
BlockShuffle remains within 1% of compiled full SwiGLU, and the median relative
throughput reaches .8. Also compare same-process compiled versus eager timing
for each model to distinguish general compiler speedup from a structured-layer
benefit. No new training trajectory or training-speed claim follows from
post-training compilation. If it fails, record the concrete failure/overhead
before any further kernel or architecture work. This is one compiler mode,
not an unbounded autotuning sweep.

Implementation clarification before execution: each model gets a fresh child
process for memory and initial correctness, with a 300-second worker deadline.
Worker stdout/stderr and any compiler exception are retained. The parent then
loads both models for timing and repeats compiled validation; co-resident memory
is never reported as single-model memory. Validation uses the actual supplied
forward callable, because delegating .loss can bypass a compiled module. Two
focused evaluator tests passed before this probe. Local disk compiler caches
are ignored through .gitignore; no global cache is deleted.

## Native worker failure and identical-recipe retry

The first full-reference worker exited with native status 3221226505 before
producing a worker JSON result. The parent retained its failure and native log
under results/compile_scale384_default_v1. Its process was confirmed terminal;
no compiler speed or numerical result was inferred from the failed attempt.
A standalone CUDA FP32 compiler smoke test then completed with maximum absolute
error 9.536743e-7 for sin(x)+x^2 on 128 values. This justifies one identical-mode
full audit retry with stage logging and Python fault tracing, retained under
results/compile_scale384_default_v2. No weights, precision tolerances, compiler
mode or promotion threshold changes. The reason for the native abort is not
established by the successful tiny test.

## Diagnostic follow-up after the failed relative-speed gate

The identical-mode retry passed numerical checks for both models. Median
compiled speedups versus each model's eager path were 3.593 for full SwiGLU
and 3.494 for BlockShuffle, but the compiled candidate/full throughput ratio
was only .632. The .8 retention gate failed. Do not promote this backend as
a solution or compare compiled BlockShuffle against an eager full reference.

Before proposing a kernel change, collect four diagnostic profiles: eager and
default-compiled full SwiGLU and BlockShuffle, ten warmups and ten profiled
forwards each. This introduces no new compiler mode, model or training run.
Keep operator CPU/device events and identify layout/pointwise/GEMM costs;
profile instrumentation is excluded from serving claims. If device timings
are unavailable, report that limitation instead of inventing a GPU breakdown.
