# H157: FP16 checkpoint storage as an alternative to CPU offload

**FAIL pinned-host gate; numerical and GPU-memory checks pass.** This is a numerical/storage result: 36 GPU backwards and six
separate diagnostic forwards, with zero optimizer updates. It does not establish
faster complete updates or preserved training quality. H156's qualified exact-path
memory helper remains unchanged; the broader research objective stays open.

## Mechanism and mathematical meaning

Each of the eight whole-block checkpoints saves only its FP32 input as FP16 on
GPU, then restores FP32 for recomputation. The original forward uses FP32 inputs;
weights, optimizer moments and classifier/log-softmax buffers stay FP32. Identity
packing and native repeats control for hook semantics and nondeterministic noise.

For a block and fixed upstream vector, the observed backward matches the native
block VJP at Q(x), where Q is the FP16 roundtrip, while the output remains F(x).
The CPU input-and-parameter check passes with maximum relative error
0.000e+00. Across a stack, independently rounded
saved inputs generally do not describe one coherent original forward trajectory.
This is approximate gradient computation, not exact differentiation of the FP32
forward, and no unbiasedness claim follows from rounding.

The codec matches pointer, shape, stride and storage offset within each block's
checkpoint scope; unexpected nonempty saved tensors raise an error. Empty
checkpoint sentinels pass through. It closes over primitive metadata, not the
original input tensor, so retaining a closure cannot silently retain FP32 storage.
All temporary forward overrides are restored. Exactly eight inputs are saved and
unpacked in each identity/FP16 backward. Diagnostic CPU input copies are taken in
separate forwards and excluded from the clean memory probes.

[PyTorch saved-tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)
provide the mechanism. [GACT](https://proceedings.mlr.press/v162/liu22v.html) establishes
generic activation-compressed training as prior art. This prototype tests a limited
checkpoint-storage tradeoff; it is not claimed as a novel activation or architecture.

## Every source fixture

All six H156 ordinary step800 checkpoints are used, with their original Adam
moments and sampler positions: two corpora, seeds 101/113/127, batch16/context512,
9,099,648 parameters. Each arm samples the same next batch and preserves its model
and optimizer state hashes. All native/control gradients use FP32 autograd.

| Corpus | Seed | Global gradient error | Max tensor error | Gradient cosine | Allocation saved vs native | Allocation / offload4 | All gates |
|---|---:|---:|---:|---:|---:|---:|---|
| wikitext2 | 101 | 0.01672% | 0.02976% | 0.999999986 | 22.83% | 1.0000 | False |
| wikitext2 | 113 | 0.01575% | 0.02810% | 0.999999988 | 22.83% | 1.0000 | False |
| wikitext2 | 127 | 0.01522% | 0.02806% | 0.999999988 | 22.83% | 1.0000 | False |
| tinystories | 101 | 0.01906% | 0.02961% | 0.999999982 | 23.04% | 1.0000 | False |
| tinystories | 113 | 0.01948% | 0.02864% | 0.999999981 | 23.04% | 1.0000 | False |
| tinystories | 127 | 0.01961% | 0.03118% | 0.999999981 | 23.04% | 1.0000 | False |

The FP16 thresholds were fixed before execution: global relative L2 <=0.002,
maximum tensor relative L2 <=0.02, global cosine >=0.99999 and scalar loss error
<=1e-6. These are approximate-gradient guardrails, not H156's exact-path limits
and not a guarantee of long-run quality. Per-tensor values remain in audit.json.
For native repeats, buffered loss, identity hooks and offload4, the original
1e-5 global / 1e-4 tensor / 1e-6 loss limits apply. Their maximum observed global
relative gradient error is 1.096e-07 and
maximum tensor error is 1.727e-07.

## Measured allocation and host memory

| Arm | GPU peak range MiB | Maximum pinned host MiB | Mean GPU peak MiB |
|---|---:|---:|---:|
| native0 | 650.14-656.32 | 64.00 | 653.23 |
| buffer | 548.33-554.51 | 64.00 | 551.42 |
| identity | 548.33-554.51 | 64.00 | 551.42 |
| offload4 | 500.33-506.51 | 64.00 | 503.42 |
| fp16 | 500.33-506.51 | 64.00 | 503.42 |
| native1 | 650.14-656.32 | 64.00 | 653.23 |

The gate requires FP16 peak <=1.02 times offload4, <=0.90 times native0, and zero
pinned-host peak, on every fixture. Loaded model, Adam moments and datasets remain
resident. Peaks include construction, forward, backward and gradient-copy/check
phases; the larger construction or later peak is retained. There is **no optimizer
step**, so these are not complete-update training peaks, and no cold timing is
reported as throughput. They are not isolated inference memory. Driver/context
and other processes are excluded. Pinned host memory is not total CPU RSS.

**The host gate fails on all six fixtures.** Each FP16 probe follows offload4 in
one process. Offload4 leaves 67,108,869 pinned bytes owned by the allocator, and
both FP16 and the subsequent native repeat retain that same allocation. Their
active pinned allocation returns to zero, with a five-byte active peak. Even the
first native probe reports five allocated bytes. The boundary helper resets host
peaks but does not release the pinned cache. Consequently this design cannot
establish the promised zero-pinned-host result for FP16 in isolation.

The [PyTorch host allocator documentation](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.host_memory_stats.html)
distinguishes allocated bytes (active plus cached) from active bytes. The cache
carryover diagnosis follows from that definition and the measured arm sequence;
the source of the five-byte activity is not established. We neither subtract the
cache retrospectively nor replace the frozen gate with an active-byte threshold.
The original audit remains failed. This is a host-measurement qualification
failure, not evidence that FP16 checkpoint storage needs 64 MiB of pinned buffers.
A follow-up must preregister fresh-process controls and distinguish incidental
host transfers from checkpoint offload before claiming a host-memory advantage.
No additional GPU runs or optimizer updates were spent on this diagnosis.

Eight FP16 inputs and four GPU-resident FP32 inputs both nominally occupy 48 MiB
at this shape; casts, unpacking temporaries and actual tensor lifetimes still need
measurement. The buffer-only and identity controls expose the cost of keeping all
eight FP32 inputs. Native loss additionally uses the ordinary classifier path.

## Activation rounding and finite-range scope

The separate diagnostic forward saves every full original checkpoint input to CPU.
Independent NumPy FP16 conversion verifies all 48 tensors, including the following
maxima over source fixtures. Underflows count nonzero FP32 inputs rounded to zero;
counts are summed over six input tensors per block.

| Block | Maximum input magnitude | Maximum relative rounding L2 | Underflow count |
|---:|---:|---:|---:|
| 0 | 0.20430 | 0.02090% | 27 |
| 1 | 4.13092 | 0.02077% | 2 |
| 2 | 4.57513 | 0.02079% | 1 |
| 3 | 4.68076 | 0.02079% | 2 |
| 4 | 4.76476 | 0.02077% | 1 |
| 5 | 5.19323 | 0.02078% | 2 |
| 6 | 5.61984 | 0.02078% | 0 |
| 7 | 5.21190 | 0.02075% | 0 |

All actual rounded inputs and all gradients are finite. A separate CPU cast of
100,000 to FP16 produces infinity, deliberately demonstrating the range limitation.
The prototype is not an arbitrary-range production API; it does not hide overflows
with clipping. Tensor gradient errors and approximate long-run effects must be
requalified at other model sizes, scales and precisions.

## Audit, startup repair and decision

An independent NumPy FP64 audit recomputes every gradient error and cosine from
saved full gradient arrays; it also recomputes every diagnostic rounding error.
There are 43 clean zero-allocation/reservation GPU boundaries. All sources,
checkpoints, batch identities, unchanged model/moment hashes and hook counts verify.
The full-model reference is measured native FP32 autograd twice, not CPU FP64.
The local CPU semantic check contributes two additional backwards, no updates.

The first launch failed before any stage log, protocol or numerical work because
operator.py shadowed Python's standard-library operator. Renaming it to codec.py
and changing two imports repaired startup; original contents and the error record
are preserved in operator.before.txt and startup_failure.md. No recipe or numerical
threshold changed. The single retry passed local preflight before GPU execution.

The numerical and GPU-allocation subchecks pass, but the frozen host gate fails. The candidate remains unqualified: resolve host-memory measurement with fresh-process controls before complete-update timing or long-training allocation. Preserve this failure; do not interpret it as a numerical rejection.

[Prospective plan](checkpoint_fp16_plan.md),
[raw measurements](../results/checkpoint_fp16_v1/result.json),
[NumPy audit](../results/checkpoint_fp16_v1/audit.json).
