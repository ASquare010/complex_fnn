# Feature flow execution revision 4: short resource screen passed

[Registered execution](feature_flow_v4_hypothesis.md) retains the original
FFN equations, counts, initialization and three steps. Independent CPU/CUDA
checks passed before profiling. All 48 profiles completed with finite updates,
reserved memory below 6 GiB, and all six primary resource comparisons passing.
This permits integration and the registered language screen; sustained resource
benefit and language quality remain unproven.

| Corpus | Dense reference | Allocated memory reduction | Throughput ratio |
| --- | --- | ---: | ---: |
| TinyStories | Full GELU | 25.54% | 1.029 |
| TinyStories | Full native SwiGLU | 28.36% | 1.015 |
| TinyStories | Full kernel SwiGLU | 25.68% | 1.027 |
| WikiText | Full GELU | 25.54% | 0.984 |
| WikiText | Full native SwiGLU | 28.36% | 1.024 |
| WikiText | Full kernel SwiGLU | 25.68% | 0.992 |

Allocated peak 401.7 MiB; median time per 100 primary updates about 4.28 seconds
on TinyStories and 4.30 seconds on WikiText. TinyStories primary rounds were
4.96/4.28/4.23 seconds, showing variation; all raw rounds are preserved. Against
dense the worst median update-time increase is about 1.6%. Three alternating-order
rounds, 20 warmup/100 measured updates and maximum allocated memory were used.
The narrow margin and between-round variation require sustained confirmation;
these separate screens do not isolate the causal timing contribution of caching.

Version 3's forward kernel is unchanged. Backward reconstructs each trajectory
while keeping each step's bounded reaction and derivative in lane-local
registers, then reuses them in the reverse pass. Group matrix-gradient tiles
reduce deterministically through shared memory and Torch. Transpose, partial
buffers, casts and remaining dispatch count toward measured time/memory.
Register pressure may offset the removed repeated nonlinear evaluations.

CPU FP64 independent equations, all adjoints/finite differences, counts,
initialization, calibration and five complete-model causality/locality/save-load/
exact optimizer recovery checks passed. CUDA FP32/BF16 checks passed for group
sizes 1/8/32, explicit size-37 fallback, incomplete tiles, nonzero M, removals,
full-model ordinary-autograd comparisons and actual 2048-token gradients, at
the unchanged tolerances. Native BF16 cast edgecases passed; raw kernels use
FP32 and TF32/FMA fusion remain disabled.

Evidence: `records/feature-flow-v4-checks.json`, frozen source/protocol under
`dump/feature-flow-v4/`, its completed `result.md` and
`records/feature-flow-v4-profiles.json`. Earlier failed executions remain preserved.

Production integration passed: initialized weights/counts/outputs/gradients
match prototypes exactly on CPU FP64 and CUDA BF16 AMP, including width 512
with 32-coordinate groups and nonzero M. All five modes and the calibrated
dense control have exact shared-Trainer CPU recovery. Variance-calibrated compact
SwiGLU preserves all other weights/counts and scales only down initialization
by sqrt(1376/352); its Gaussian residual-variance scaling identity is checked.
See `records/feature-flow-v1-integration-checks.json`.

The [conditional confirmation](feature_flow_confirmation.md) is registered
before language training. Eleven variants per corpus will receive the original
fixed one-seed budget, with all registered removals and full eligible validation.
No language or originality claim follows from the passing hardware screen.
