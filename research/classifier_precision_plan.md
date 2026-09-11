# H113: classifier gradient precision at identical saved states

Status before execution: HYPOTHESIS. H112 supplies a verified memory saving but
fails WikiText quality in two fresh seeds. Its 3.40e-8 evaluator drift cannot
explain percent-level NLL differences. This diagnostic performs no optimization
and does not reopen that fixed recipe or establish a learning result.

## Question and derivation

For hidden rows H, classifier W, N valid targets and P = softmax(H W^T),
let G = (P - one_hot(y)) / N, with ignored rows zeroed. In real arithmetic,

    dH = G W
    dW = sum_chunks G_chunk^T H_chunk.

Partitioning preserves this derivative. In finite precision, summing chunk
gradients in BF16 before casting differs from casting each partial gradient to
FP32 before summation. Autocast can cache one BF16 conversion of a leaf FP32
weight. Several graph branches then share the conversion node. This is a
hypothesis about the current execution, not yet evidence of the observed error
or its effect on language learning. No new mathematical identity is claimed.

## Fixed data and methods

Use all twelve H112 completed trials at checkpoints 100 and 800: 24 states,
two corpora, three training seeds and two training policies. For each state,
capture final normalized decoder hidden rows using the unchanged native BF16
decoder on one B8/T512 training batch drawn with seed = training seed + 30000.
The same batch is used across checkpoint/policy within a corpus/seed. This is
a fresh probe of reused training data, not independent held-out quality data.
Preserve input tokens, targets, FP32 hidden rows and FP32 classifier weights.
Capture and audits count toward diagnostic work but are not optimizer updates.

On these exact detached inputs compare five methods:

1. native BF16 classifier and mean CE;
2. maintained checkpointed 512-token CE chunks, autocast cache enabled;
3. the same maintained helper with only its autocast weight cache disabled;
4. native FP32 classifier and CE;
5. the maintained chunks with FP32 classifier computation.

Compute a native FP64 loss and both gradients once per fixture as a numerical
reference, using the exact captured FP32 values promoted to double. It is a
local classifier reference, not a FP64 decoder or a claim of exact arithmetic.
All methods use the same objective, weights, inputs and targets. No additional
learned parameters, kernel implementation, clipping, optimizer or updates.

Record loss; relative L2 and maximum absolute errors for dH/dW versus FP64;
cached/uncached equality and error ratios; exact weight-cast graph-node counts
and gradient dtype arriving at those nodes in the warmup graph. Test a separate
five-term linear-sum witness, with gradient terms 1,1,1,1,256 and reverse order,
on CPU and CUDA BF16. Report actual cached/uncached sums and graph topology;
do not assume summation order or that this witness proves language causation.

## Correctness and resource procedure

Before real-state captures, verify CPU-double values/gradients for all five
methods on two tiny shapes with partially ignored and uneven chunk boundaries,
and run finite-difference gradcheck on the three chunk methods. No all-ignored
training-gradient claim is made. Source/runtime/numerical qualification failures
stop the stage with preserved evidence; ordinary comparative failures do not
truncate the 24-state grid.

For each fixture/method use one warmup and three synchronized timed forward+
backward repetitions, each with fresh gradients. Require identical gradients
across repeats. Inspect cast hooks only during warmup, outside timing. Rotate
the five-method order by fixture index. Report forward, backward, total time,
allocated/reserved peaks, input storage and all per-fixture metrics. FP64 gets
one untimed reference backward. Thus 24 reference + 480 profiled/warmup backward
passes = 504, plus tiny qualifications, with zero optimizer updates.

Clear live tensors and cuBLAS workspaces only between method fixtures; verify
zero allocated/reserved storage at each boundary. Normal workspaces are fully
charged within each measured method. CPU captures/reference tensors are outside
CUDA method peaks. Separately record capture and overall diagnostic peaks; none
is a full training-job or inference-serving measurement. One GPU worker only.

## Prospective decisions

Report mean, median, sample variance and range by corpus, checkpoint and method.
The cache mechanism is supported only if graph inspection shows one shared
weight conversion for cached chunks and eight separate conversions for uncached
chunks, and all fixtures have identical cached/uncached loss and hidden gradient
while some weight gradient differs. Otherwise refine or reject that explanation.
The scalar witness is a separate scoped example, not a substitute for these gates.

An altered method earns only a whole-model resource/correctness screen if:

- median dW relative-error ratio versus cached chunks is <= 0.75 in each corpus;
- its dH error is <= 1.01 times cached error on every fixture;
- allocation is <= 0.85 times native BF16 on every fixture;
- median paired total-time ratio is <= 1.5 versus native BF16 in each corpus;
- loss and gradients remain finite.

No language training is allocated here, even on a pass. Smaller numerical error
does not prove better optimization or NLL. A failure closes the fixed candidate
under these local gates without eliminating all precision-aware execution.
Do not tune chunk size, precision, thresholds or profiling repeats after results.

## Provenance and runtime

Read current H112 evidence and verify its receipt before source changes. Freeze
this plan, diagnostic sources, all dependencies, both corpus manifests and the
24 source checkpoints. Preserve current entry documents separately before later
updates. Use the existing absolute UV-managed Python 3.12.9 and packages, four
threads, BF16 autocast, TF32 off, malloc, hash seed 107 and bytecode bypass.
Preload SymPy, PyTorch and TorchDynamo in the order that completed H112 before
CUDA. Prior native import failures remain unresolved; no automatic retry loop.

Primary precedents: [AMP weight-cache documentation](https://docs.pytorch.org/docs/2.14/amp.html),
[PyTorch cache implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/autocast_mode.cpp),
[floating-point accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
and [Cut Your Losses](https://arxiv.org/abs/2411.09009). This is an execution
diagnosis of established techniques, without an activation/SOTA/novelty claim.
Maintain the broader VRAM/quality and separate architectural goals unchanged.
