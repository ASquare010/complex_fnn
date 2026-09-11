# H111: classifier streaming removes the evaluation bottleneck

Classifier chunking qualifies at both tested shapes. Across 24 saved model
states, it reduces evaluation tensor allocation by **16.0-16.4% at T128** and
**31.11% at T512**, with median evaluation-time overheads of **0.79% and 1.49%**.
Its largest relative NLL difference from native evaluation is **1.24e-8**,
well below the frozen 1e-4 tolerance. No optimizer update is performed.

The conventional control—processing fewer complete sequences per batch—saves
slightly more memory but costs **3.71x / 7.07x** native evaluation time. Both
policies preserve scores within tolerance. The fixed sequence-chunk recipe
fails the time gate, so the deterministic selection chooses classifier chunks
for both shapes. This is a useful execution component, not a new FFN or proof
of lower memory during a complete training job.

## What was measured

The [frozen plan](streamed_evaluation_plan.md) reuses every checkpoint at updates
100/400/800 from H110's eight completed trials: 18 states at B16/T128 and six at
B8/T512. These are correlated saved states from three training seeds at T128
and one at T512, not 24 independent training replications. All measurements
use the same already-used WikiText-2 development split and tokenizer. There is
no test split access or new model training.

Three evaluators share the original outer batch order and full attention
context. Native evaluates the complete decoder/classifier batch. Classifier
chunks retain the decoder batch and reduce classifier/CE in 512-token pieces.
Sequence chunks evaluate whole sequences in batches of four at T128 or one at
T512, weighting each sub-batch mean by its number of valid targets. The sequence
control never cuts attention context or redefines the data being scored.

The model is the same 9,099,648-parameter narrow GELU decoder. Evaluation
fixtures include the real saved final Adam moments on CUDA, CPU scalar steps,
and explicit FP32 zero-gradient buffers of the original shape. These buffers
match storage sizes, not gradient values. They and the full CUDA train/valid
cache remain allocated throughout measurement. Earlier model checkpoints use
the same final persistent-state fixture. This is not an optimizer replay.

UV-managed Python 3.12.9, PyTorch 2.14.0+cu132, RTX4070 Laptop, BF16 autocast,
FP32 weights, TF32 off, four CPU threads, malloc and hash seed107 match the
recorded runtime. Bytecode is bypassed with the previously recorded mechanism;
it is not claimed as a fix for earlier intermittent native failures. The study
takes 662.14 seconds, with no SGD or profiling optimizer steps.

## Fixed gate results

Every saved state must preserve NLL within 0.01% relative and save at least 10%
evaluation allocation. The median paired evaluation-time ratio must be <= 1.5.
Timing covers the same first 16 outer validation batches after two warmups, with
five repeats per policy and cyclic policy order. Full-validation scoring and
memory use all complete contexts. Mean/median/sample variance/min/max for each
metric are retained in the machine-readable summary.

| Shape | Evaluation | Peak allocation across states, MiB | Median paired time | Maximum relative NLL difference | Decision |
|---|---|---:|---:|---:|---|
| B16/T128 | Native |255.58-256.62|1.000x|0|Reference|
| B16/T128 | Classifier chunks |213.58-215.49|1.008x|1.24e-8|Qualified|
| B16/T128 | Sequence chunks |208.45-208.61|3.712x|2.50e-6|Time gate fails|
| B8/T512 | Native |321.42|1.000x|0|Reference|
| B8/T512 | Classifier chunks |221.42|1.015x|1.24e-8|Qualified|
| B8/T512 | Sequence chunks |209.38|7.065x|2.38e-6|Time gate fails|

These allocations include persistent fixtures. They exclude driver/context
overhead and other processes. The maximum of the complete H111 measurement
process, which includes the native controls, is 321.42 MiB. Reserved-memory
measurements are stored separately. First-case allocation differs slightly
from subsequent cases; all cases are retained without timing or memory cherry
picking. No uncertainty interval over independent training seeds is inferred
from the state-level distribution.

Both streaming evaluators have fixture peaks below the corresponding H110
training peaks. Therefore max(old training peak, new fixture evaluation peak)
predicts retention of the old training saving. This is a **composed allocation
estimate**, used only to justify fresh measurement. It is not reported as an
observed training-job result. A fair integrated comparison must give both
training policies the same optimized evaluator and measure the complete job.

## Mathematical and state checks

Valid-target-weighted sums of CE over a disjoint partition preserve the mean
objective in real arithmetic. Ignored targets contribute neither loss nor
denominator, and an entirely ignored sub-batch must be skipped rather than
adding zero times NaN. Native and streamed all-ignored batches return NaN.
The implementation preserves double precision for the tiny mathematical checks.
[PyTorch's CE reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html)
defines these reduction and ignore semantics.

All 36 CPU-double shape/mask/policy qualifications pass at 1e-10 absolute/relative
tolerance, including uneven final chunks and fully masked sub-batches. They
perform no backward pass and do not claim gradient verification. Across all 72
full scores, model weights, zero-gradient buffers, CPU/CUDA RNG, sampling RNG
and train/eval mode are unchanged. Native final-checkpoint scores also match
the previously audited H110 scores exactly.

The independent audit reloads all 24 checkpoints and uses the maintained native
forward evaluator for rescoring. It checks the 72 policy-comparison arithmetic,
state/fixture hashes, timing statistics, every gate and both policy selections.
The [audit record](../results/streamed_evaluation_v1/audit.json) is authoritative
for completion and discrepancies; no training update is part of the audit.

## Research decision

Keep classifier streaming as a **PROMISING COMPONENT** for evaluation. Eliminate
the fixed sequence-microbatch recipe on its runtime cost at these shapes.
This does not eliminate microbatching generally or demonstrate superiority over
a fused classifier implementation. [Cut Your Losses](https://arxiv.org/abs/2411.09009)
is prior work on the same memory bottleneck; this code uses ordinary PyTorch
chunks and does not implement or claim a new CCE method.

The conditional [H112 plan](whole_job_memory_plan.md) is the next actual memory/
quality test: fresh training seeds, WikiText-2 and TinyStories, equal budgets,
the same qualified evaluator for both arms, and independently native final
scores. It uses a prospective one-sided quality criterion. It does not reopen
H110's stopped fidelity grid or retroactively excuse its deviation.

All maintained models, variants, recipes and defaults remain unchanged. The
previous 116-test pass is separate from these 36 new evaluation qualifications.
The full VRAM/quality goal and separate parameter-efficient-architecture target
remain unfulfilled. Sources and numeric records are kept in
[the source directory](../results/streamed_evaluation_v1/source/README.md) and
[summary](../results/streamed_evaluation_v1/summary.json).
