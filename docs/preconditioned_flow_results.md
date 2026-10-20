# Reversible latent preconditioning: bounded result, 2026-10-13

**No useful generation improvement or baseline win.** Three new models completed
2,000 updates each; independent reload reproduced every saved generated ID and
first-draw vector exactly. Original weights and data are unchanged. The full-v4
position-memory run remains paused at step 1,001. The broader goal is still open.

## Hypothesis and zero-update audit

The previous [continuous-flow pilot](probabilistic_vector_results.md) used
per-coordinate normalization. Its 2,048 training target vectors have a covariance
condition number of approximately 9.26 million even after that normalization.
The leading 16 axes contain 69.09% of variance, and the leading 64 contain
99.9985%. This is not evidence that the remaining axes contain no information.

We tested a full-rank ZCA transform fitted only on training vectors. Eigenvalues
are floored at 1e-5 for numerical stability; all 256 coordinates remain and the
transform has an explicit inverse. No codec weights or trainable parameters are
added. The theoretical transformed covariance condition number is about 5.31.
This is inspired by [Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337),
not a reproduction of its experiments or a claim that linear conditioning is
sufficient for text generation.

Transforming and restoring genuine codes changed no decoded IDs across 2,560
saved train/dev/previous-test examples. Maximum FP32 code roundtrip error was
0.000733. The codec itself was not universally exact: original reconstruction
accuracy was 99.9207% train, 100% dev, and 99.9023% previous test.

Equal-RMS noise tests challenged the simple "outside the text space" explanation:
at normalized noise RMS 0.3, isotropic noise retained 95.12% development token
accuracy, whereas training-covariance-shaped noise retained only 26.42%. Changing
directions that normally carry text variation can be more damaging than changing
other directions. These perturbation tests are not generated-language results.

## Matched protocol and scope

Exactly 23,209,216 trainable parameters in each arm, original encoder and decoder
frozen, same fresh initialization as the previous controls, same batch and noise
streams, batch 64/microbatch 32, AdamW 3e-4, weight decay 0.01, gradient clipping 1.
The only changed interface is the output/active-state coordinate system and its
associated regression geometry. History still consists of original codec vectors.

- One-pass: transformed clean-vector MSE plus 0.02 decoded CE after update 200.
- Flow: transformed velocity MSE, independent Gaussian-to-target straight paths.
- Flow+CE: velocity MSE plus 0.02 CE on the estimated clean endpoint after 200.

The data remain 2,048 training dialogues (458 everyday, 1,590 chat) and 256 dev
dialogues (5 everyday, 251 chat), predicting only the first eight tokens of the
first eligible full assistant span. The new 256 held-out cases are all chat,
selected after the 512 test cases previously examined. IDs are disjoint and no
literal new-test input prefix equals a training prefix. Near duplicates are not
excluded. These are not representative full-v4 or complete-response tests.

All six models are evaluated on the same new test. Existing controls receive no
additional optimizer updates. Samplers are chosen on development data using the
same Euler 8/16/32 and Heun 16 options. Preconditioned flow selects Heun 16
(32 network evaluations); preconditioned flow+CE selects Euler 16. Both original
flow controls select Euler 8. The genuine-code oracle on the new test recovers
2,046/2,048 tokens and 255/256 complete eight-token blocks.

## Generated results

First independent draw, 256 new prompts and 2,048 reference positions:

| Objective | Original coordinates | Preconditioned coordinates | Exact blocks, original/new |
|---|---:|---:|---:|
| One-pass MSE+CE | 6.54% | 5.37% | 5 / 1 |
| Flow | 1.32% | 2.05% | 0 / 0 |
| Flow+CE | 2.05% | 2.25% | 0 / 0 |

Four-draw mean agreement is 2.32% for preconditioned flow and 2.43% for
preconditioned flow+CE, with zero exact blocks across 1,024 draws in each arm.
The paired-dialogue 95% intervals for the preconditioned-minus-original change
are [-2.54, 0.05], [-0.05, 1.61], and [-0.73, 1.17] percentage points respectively.
Each includes zero. Both transformed flow arms are clearly behind the original
one-pass control on this metric. These intervals reflect case uncertainty, not
training-seed uncertainty; there is one training seed per recipe.

On 256 checked training examples, preconditioned one-pass/flow/flow+CE agreement
is 41.26% / 19.43% / 13.87%. The corresponding previous control scores were
99.95% / 10.11% / 15.72%. Preconditioning changed optimization, but did not yield
a useful generator at this budget. This does not establish an impossibility or
rule out different schedules and representations.

Example reference: ` First, find the cost of the j`

- Preconditioned one-pass: ` Aug 2006egE ( new:`
- Preconditioned flow: ` Aary un was anoneed co`
- Preconditioned flow+CE: ` Aary un a wasededine`

All 24 ordered examples per arm are saved. References may end midword because
the target length is eight BPE tokens. The negative practical judgment relies on
the actual fragmented samples as well as scores; reference disagreement alone
cannot reject every legitimate alternative continuation.

The unchanged full-v4 token-memory model reaches **40.58% token agreement and
64/256 exact blocks** on the same new prompts. It trained for 48,762 updates on
much more data and receives recent raw tokens. It is a quality reference, not
an equal-budget architecture or training-efficiency comparison.

## Conditioning and sample diversity

| Preconditioned arm | Correct history | Wrong history | Blocked history |
|---|---:|---:|---:|
| One-pass | 5.37% | 0.44% | 0.20% |
| Flow | 2.05% | 0.73% | 0.24% |
| Flow+CE | 2.25% | 0.24% | 0.24% |

The wrong/blocked controls also remove or change the question because it is
inside history. They show conditioning dependence, not isolated old-fact retrieval.

Across independent samples for each prompt, same-position token collision rises
from 3.82% to 61.52% for flow, and from 5.77% to 45.30% for flow+CE. This is
between-sample similarity, not a measurement of repetition within an answer.
Using the unbiased token Brier estimator `2*agreement - pairwise_collision`,
the scores fall from -0.0070 to -0.5688 and -0.0106 to -0.4044 respectively
(higher is better). Corresponding 2/4/8-gram estimates are also retained. These
are four-sample diagnostics, not the published aggregate BrierLM benchmark or
perplexity. A deterministic model has collision probability one; its Brier score
does not substitute for evaluating its underlying token probability distribution.

## Cost and verification

Training plus periodic development evaluation and checkpoint saves took
239.84 / 221.17 / 246.78 seconds, excluding preparation and final verification.
Sampling plus frozen decoding for batch 32 took:

| Arm | Original | Preconditioned |
|---|---:|---:|
| One-pass | 19.5 ms | 19.6 ms |
| Flow | 138.0 ms | 541.7 ms |
| Flow+CE | 138.0 ms | 273.4 ms |

History encoding is excluded. Most of the increased flow cost comes from the
development-selected higher number of network evaluations. These are not
end-to-end chatbot timings.

Four CPU contract tests passed, including path/solver behavior, transformation
invertibility, masks and frozen-decoder gradient flow. Additional Brier checks
verified deterministic limits and the exact Bernoulli expectation. Independent
GPU verification checked train-only geometry reproduction, four newly re-encoded
test records, every saved generated ID/vector, frozen original decoder identity,
parameter counts, all Adam counters at 2,000, checkpoint agreement, source archive
bytes and protected original data/model hashes. Samplers receive no target codes.
The pilot has periodic optimizer/RNG checkpoints and a resume CLI; a full
interrupt/replay lifecycle was not separately tested.

Evidence: `dump/preconditioned-flow-20261013/`. Tools:
`audit_flow_geometry.py`, `preconditioned_flow_common.py`,
`train_preconditioned_flow.py`, `verify_preconditioned_flow.py`.
No production code changed, no full-corpus training resumed, and no old weights
were deleted. The exact new training cost is 6,000 optimizer updates across
three bounded arms.

The next proposed diagnostic isolates **reading**: original compact codes versus
separate frozen-decoder position features under the same output model. It must
disclose the expanded attention cost and later test consistent generated-vector
feedback. See the [research direction and decision gates](frozen_codec_research_direction.md).
