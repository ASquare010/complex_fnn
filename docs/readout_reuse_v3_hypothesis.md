# Readout reuse execution v3: native grouped forward, fused adjoints

Registered after v2's whole-model check failed twice with identical rejection
and its [saved-fixture diagnosis](readout_reuse_v2_result.md) completed. V2 is
retired before resources/language. V1 already completed and failed resources.
No architecture, count, initialization, calibration, control or gate changes.

Keep v2's projected-z storage and native projection adjoints, but preserve
ordinary Torch grouped BMM in forward and when reconstructing the readout for
its P/Q gradient. The scalar-only CUDA response kernel matched native responses
exactly on the saved fixture. Use it before native BMM, with a fixed ones vector
to avoid introducing signs before the grouped operation. Fixed/untied readouts
retain their native registered alternating signs. That scalar finding must be
checked again at all registered shapes, not generalized from one fixture.

Retain fused input/correction adjoints and their deterministic 32-token tiles;
do not use v2's fused grouped forward. The forward BMM keeps native accumulation
and native BF16 cast sites. Reconstruction uses exactly that same forward
response/readout operation, never a lower-precision surrogate. Both shared-P
gradient contributions still use separate native projection products and sum
in FP32. The output gain keeps native BF16 rounding. A nonpersistent FP32 ones
buffer costs hidden*4 bytes per layer and is included in resource measurements;
it is not a parameter or a change to the mathematical map.

This tests whether projected activation storage, scalar fusion and fused
adjoints can repay dispatch/temporary costs while preserving the complete
forward tolerance. Native BMM, transposes, FP32 response banks, tile partials
and saved z may still fail memory/speed. The previous trace identifies forward
rounding on its fixture, not a demonstrated causal resource bottleneck.

Use separate v3 prototype/kernel/check/profile files, preserving all v1/v2
failures and the saved fixture. Kernel code may reuse v2's scalar/adjoint
implementation, with its original Torch context/current-stream driver; raw
kernels are FP32 only, no atomics/TF32/fast math/FMA fusion. Explicit ordinary
fallback remains for group size>32 or unsupported dtype. Counts and projection
work stay v2's: primary 29,491,200 training projection FLOPs/token plus scalar,
reduction, transpose and dispatch work.

Repeat all v2 CPU checks and the complete CUDA suite: 192 independent FFN
fixtures (FP32/BF16, sizes 2/8/32, incomplete rows and size-34 fallback), 24
whole Transformers including actual [8,256,512], nonzero corrections and all
removals; raw forward/input/correction fixtures; native BF16 cast edgecases.
Also require the saved v2 failing model/tokens now pass the original whole-model
output and parameter-gradient comparisons. Original tolerances remain FP32
1e-5/3e-4, BF16 output .004/.02 and parameter gradients .0001/.05. Preserve
errors/repairs; do not skip a failed fixture or relax its tolerance.

Only after terminal complete checks, freeze all sources and repeat the original
54 profiles: nine variants, two corpora, three alternating-order rounds,
20 warmup/100 measured updates, unchanged backbone/optimizer/data. No overlapping
GPU work or source changes during that screen. Retire any primary resource
failure before language. Only a survivor enters original exact production
equivalence and 24-run language comparisons, then conditional repeated-seed
confirmation. No novelty, language or sustained-resource benefit is established.
