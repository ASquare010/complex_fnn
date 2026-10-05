# Readout reuse v4: retire before resource or language screening

The separate v4 implementation passed the CPU equation, gradient, initialization,
parameter-count and exact recovery checks. All 192 independent GPU FFN fixtures
and the first 18 complete-model comparisons passed. The full BF16 grouped model
then failed its embedding-gradient comparison: 2 of 2,097,152 elements exceeded
the original absolute/relative tolerances (0.0001/0.05). Maximum disallowed
absolute error was 0.000144506. Forward outputs matched exactly.

Evidence: [failure](../records/readout-reuse-v4-gradient-failure.json),
[CPU checks](../records/readout-reuse-v4-cpu-checks.json), and
[saved-v3-fixture trace](../records/readout-reuse-v4-gradient-diagnostic.json).
The diagnostic compared identical layer inputs and incoming gradients. Projected
BF16 adjoints differed at 0/2/1/2 coordinates across four layers, versus v3's
63/79/64/69 with its raw grouped backward. The group parameter gradients matched
exactly in this trace. This local improvement does not override the failed
complete-model gate. The later raw/cast/regression suite did not execute.

FFN parameters remain 2,506,752 (70.35% below full SwiGLU). V4 training speed,
peak training VRAM and language quality are **unmeasured**. Retire v4 without
loosening tolerances or starting profiles/language training.

The ordered expression was informed by the [PyTorch CUDA SiLU implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSiluKernel.cu).
Matching its mathematical expression does not guarantee matching the installed
binary's rounding. The next execution hypothesis uses the installed native
SiLU-backward operator directly while retaining projected-feature storage.
