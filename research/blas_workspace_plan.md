# H131: isolate deterministic cuBLAS workspace cost

H130 is progress: exact short optimization and memory gains, but ordinary-relative
runtime fails. Test the workspace hypothesis cheaply before another training matrix.

Two fresh sequential processes keep all H129 deterministic-default settings:
strict deterministic algorithms, cuDNN deterministic on/benchmark off, default
SDPA, FP32 training, TF32 off, four threads. Change only environment workspace:
`high` = :4096:8; `low` = :16:8. Query the installed cuBLAS workspace size in bytes.
No setter, backend override or algorithm fallback. Lower workspace can change
GEMM selection, so cross-workspace bitwise identity is not assumed.

Use both H121 native FP32 step-800 fixtures, same states and sampled batch.
Native classifier versus H128 layout-aware classifier, CPU checkpoint-input
offload in both. Ordinary RMSNorm, resident gradients, unchanged Adam state.
Reuse H123's ten-probe loop/three warmup; reverse arm order across corpora.
Before each case, evaluate one BF16 validation batch using the established
chunked evaluator to create the evaluation workspaces also present in H130.
Its allocations remain inside the construction interval: do not reset peaks.
Save the evaluation metadata. No optimizer updates.

After each case returns and Python garbage collection releases its objects,
measure allocated bytes before and after clearing cuBLAS workspaces. Require
zero live CUDA allocations afterward, then the standard zero allocator boundary.
This release measurement distinguishes live BLAS workspace storage from tensors.
Compare the measured high-low release difference with twice the queried per-
workspace size difference; this is a testable accounting hypothesis, not a
pre-assumed number of internal handles. No claim that it explains runtime.

Budget: 2 workspace policies x2 corpora x2 arms x10 =80 probe backwards.
Separate per-policy native replay for each of the eight saved gradients adds
8 backwards; total 88, zero updates. 327,680 probe and 32,768 replay targets
are diagnostics; each case/replay separately evaluates one 4,096-target batch.
Eight saved gradients; all cases fresh, no completed-case retries.

Gates: per-policy replay loss and gradients bitwise equal to saved artifacts;
unchanged model/optimizer/sampler and matched data provenance, finite gradients,
zero boundaries. Per corpus low-workspace candidate peak <=0.9x low native and
<=0.9x high candidate; median event and wall <=1.15x those controls; host <=128 MiB.
Retain every timing plus mean/median/variance. H123 also checks gradients against
its older reference at unchanged 1e-5 global/1e-4 per-tensor tolerances.

Passing earns complete-update/quality and ordinary-runtime controls at low
workspace. These diagnostic cases include one validation batch, not a complete
training job. No parameters change, no novelty claim, no H130 gate rescue.

[PyTorch CUDA environment variables](https://docs.pytorch.org/docs/main/cuda_environment_variables.html)
explain the size/count syntax. [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)
documents both settings and warns the smaller one can limit performance.
The broader VRAM/quality and parameter-efficient FFN goal remains open.
