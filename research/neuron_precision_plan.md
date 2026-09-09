# H089 - Diagnose rounding boundaries before any neuron-geometry training

H088 stops after21 passing CPU checks when the first BF16 whole-FFN gradient
comparison exceeds atol0.02,rtol0.02. The log reports55 mismatches among175,104
up-projection gradient entries and maximum absolute error0.0625. No fitting
process, data cache or optimizer update was created. Preserve H088 unchanged.

The production twist adds a BF16-rounded correction to the BF16 input. Its
independent trigonometric reference rounds the whole FP32 activation once. Their
backward paths also combine gradients at different precisions. This is a
testable explanation, not yet the diagnosed cause. An explicit FP32 activation
region might preserve the intended derivative more closely while leaving the
planned FP32 training function exactly unchanged.

Run one bounded diagnostic on the four original learned candidates and both
CUDA FP32 and BF16 autocast. Reuse H088's exact seed17 model, nonzero controls,
input generator9830 and cotangent. Evaluate four modes on the same weights:

1. original native activation;
2. original independent reference;
3. native activation with its input converted to FP32 and one output cast;
4. independent reference with the same FP32 region and one output cast.

For FP64 inputs, any later wrapper must retain FP64; this diagnostic's network
cases themselves are FP32/BF16. There are32 full-FFN forward/backward evaluations
and zero optimizer updates, corpus targets or fitting examples. Save every
output, input gradient and parameter gradient, along with comparison counts and
maximum errors. Do not alter the frozen H088 model or original reference.

Use H088's BF16 tolerance unchanged. For FP32/reference comparisons use
atol1e-5,rtol1e-5, fixed here before execution. Native and FP32-region execution
must be BIT-EXACT in FP32 for outputs and all gradients. Observing the old BF16
failure is an outcome to record, not a reason to rerun or change tolerance.

A subsequent explicit qualification recovery is justified only if the FP32
region is exact for FP32 native behavior and its independent-reference checks
pass for every candidate in both precisions. That later stage needs its own
frozen plan, must retain the original BF16 tolerance, must preserve H088's21
completed CPU tensor results, and may not claim BF16 training fidelity from
these local tests. This diagnostic itself does not allocate training.

Use one GPU process, UV, four CPU threads, TF32 off, durable exclusive writes,
source hashes and original H088 file hash/size/mtime records. Preserve failure
information. No automatic repetition or numerical-threshold adjustment.
