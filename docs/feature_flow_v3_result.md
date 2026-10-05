# Feature flow execution revision 3: retired on resources

[Registered execution](feature_flow_v3_hypothesis.md) retains the original
FFN equations, counts, initialization and three internal steps. All CPU/CUDA
numerical checks passed before profiling. All 48 profiles completed; updates
were finite and reserved memory stayed below 6 GiB. Five of six comparisons
pass, but TinyStories versus full GELU fails the slowdown limit. Retire this
execution recipe before language. Language quality remains unknown.

| Corpus | Dense reference | Allocated memory reduction | Throughput ratio | Pass |
| --- | --- | ---: | ---: | --- |
| TinyStories | Full GELU | 25.54% | 0.932 | No |
| TinyStories | Full native SwiGLU | 28.36% | 0.966 | Yes |
| TinyStories | Full kernel SwiGLU | 25.68% | 0.970 | Yes |
| WikiText | Full GELU | 25.54% | 0.984 | Yes |
| WikiText | Full native SwiGLU | 28.36% | 1.016 | Yes |
| WikiText | Full kernel SwiGLU | 25.68% | 1.020 | Yes |

Primary allocated peak is 401.7 MiB. Median time per 100 updates is about
4.34 seconds on TinyStories and 4.31 seconds on WikiText. TinyStories GELU
takes about 4.04 seconds, giving 7.3% longer update time. Relative to separately
measured version 2, throughput improved 11.42%/11.02%; these are separate short
screens, not paired simultaneous measurements or sustained confirmation.
Three alternating-order rounds, 20 warmup and 100 measured updates per profile
used the same full Transformer, data, BF16 execution and optimizer.

Forward evolves each token/group's whole trajectory in warp registers, then
writes the final reaction. Backward reconstructs that trajectory in registers,
accumulates M coefficients across reversed steps and writes the input adjoint
once. Eight token warps reduce deterministically through shared memory, followed
by Torch reduction of 32-token tile partials. Transposed M copies, FP32 partials,
casts and remaining Python dispatch are included in measurements. Cached
execution remains ordinary autograd, and sizes above 32 use the explicit fallback.

Independent CPU FP64 equations, all adjoints, finite differences, initialization,
counts, calibration and all five complete-model causality/locality/save-load/exact
optimizer-recovery checks passed. CUDA FP32/BF16 output/input/M gradients passed
the unchanged version-2 tolerances, including group sizes 1/8/32, explicit size-37
fallback, incomplete tiles, nonzero M, all removals and actual 2048-token inputs.
Full-model ordinary-autograd comparisons and native BF16 cast edgecases passed.
Raw kernels read/write FP32; TF32 and FMA fusion remain disabled.

Evidence: `records/feature-flow-v3-checks.json` and the frozen protocol/source
under `dump/feature-flow-v3/`, with completed `result.md` and
`records/feature-flow-v3-profiles.json`. Frozen source fingerprints were verified
after the process exited successfully. The next
[execution hypothesis](feature_flow_v4_hypothesis.md) caches reactions and
derivatives inside backward registers, preserving the equations and thresholds.
It is registered but not implemented or measured.
