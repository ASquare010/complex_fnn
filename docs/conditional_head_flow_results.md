# Separate context transformer and conditional diffusion head

2026-10-13. Both bounded training arms completed 1,088 updates, 2,176 new optimizer
updates in total. Seven CPU checks and full-size zero-update GPU preflight passed.
Independent reload and memory-control verification passed. No useful generation improvement
or baseline win is established. The original full-v4 production model remains
paused at step 1,001; its weights, original codec, original v4 files and previous
experimental controls are protected by hashes.

## Change being tested

The [previous shared-transformer flow](data_prediction_flow_results.md) processes
the noisy query through every transformer layer on each refinement step, even
when the history KV cache is reused. Here a constant causal query first reads
the history once, producing a fixed 512-dimensional conditioning state. A small
MLP then refines the noisy 256-dimensional output code. Only this MLP is repeated
within an eight-token output block. The sampled code is appended unchanged to
the causal history; the original frozen decoder renders its eight tokens.

[Autoregressive Image Generation without Vector Quantization](https://arxiv.org/html/2406.11838v2)
motivates separating context prediction from a conditional denoising head. This
implementation is a text-codec experiment with our existing linear flow path,
not a reproduction of that paper's image model, diffusion schedule or results.
The paper's success does not establish language generation with our codec.

The eight-layer, width-512, eight-head backbone remains. Its four FFNs of width
1,774 and four of width 1,773 become four of 1,454 and four of 1,453. Removing
320 channels in every FFN frees exactly 2,621,440 parameters. Two width-512
residual MLP blocks with affine-free LayerNorm, conditional shift/scale/gates,
and SiLU use exactly that many parameters. The existing latent projection and
final RMSNorm are shared between context and denoising; the existing output
projection and eight time-basis vectors serve the head. Every retained parameter
participates. Total trainable count remains **23,209,216**; frozen codec costs
remain separate. Architecture and capacity allocation change together, so this
is not a single-scalar ablation or proof that any gain comes from AdaLN alone.

The context state cannot see the current target/noise/time or any future memory.
Only the denoising head receives the noisy current target during training and
noise during free generation. Neither component receives token IDs. Labels and
true target codes are used for the training loss, never free sampling.

## Matched protocol

Reuse the unchanged prepared literal-end-marker data: 4,352 train dialogues,
489 development, 504 exploratory test, all 17 sources. Each arm sees the same
two shuffled epochs, batch 8, 1,088 updates and 5,200,858 assistant-label exposures
as the completed controls. This is a bounded pilot, not full-v4 exposure or an
untouched final benchmark. No new encoder pass or full-corpus cache is required.

| Arm | Predicted quantity | Objective | Existing matched loss control |
| --- | --- | --- | --- |
| velocity | `target - noise` | normalized velocity MSE + 0.02 frozen-decoder CE after step 200 | shared-transformer `flow_ce` |
| clean | normalized clean target code | clean-code MSE + 0.02 frozen-decoder CE after step 200 | shared-transformer `data_cleanloss` |

Both use seed 47 for inherited initialization, a documented independent seed
for the new head, and the same batch/noise/uniform-time streams, FP32, AdamW,
weight decay 0.01, clip 1 and 50-step warmup/cosine 3e-4 to 3e-5. CE acts through
the frozen decoder on the estimated clean code. Development data selects Euler
4/8/16; the clean head uses `(prediction - noisy_state)/(1-t)` and never evaluates
at the singular endpoint t=1. Current output is eight tokens, not a growing
8-to-64 code. Generated feedback is predicted code, not re-encoded output text.

Training computes the context graph once. Loss chunks accumulate gradients on
a temporary condition leaf and in the shared/head parameters, then chain the
condition gradient through the backbone once. All scored targets are retained;
MSE is normalized by real code targets and CE by real assistant tokens.

## Checks before training

Seven CPU tests passed. They cover exact parameter count/original decoder
identity, both chunked objectives versus joint gradients, gradients reaching
the backbone while the decoder stays frozen, future and padding invariance,
cached/full/truncated context equality, direct code appending, one context-query
pass regardless of refinement count, and in-memory Adam/RNG resume equivalence.
These are implementation checks, not quality evidence.

Full-size CUDA checks on three real query locations compare all parameter
gradients: max absolute difference 9.835e-7 for velocity and 2.444e-6 for clean.
Cached/full conditioning differs by at most 2.027e-6. Zero optimizer updates;
parameter fingerprints unchanged. First-batch and repeated-longest-query-dialogue
probes include Adam-sized moment buffers and all targets. The heaviest checked
batch has 20,096 labels/2,512 code targets, peaks at 2.166 GiB allocated and
2.461 GiB reserved. It is not a general worst-case guarantee.

| Fresh-model cached-prompt probe | Shared transformer | Separate head |
| --- | ---: | ---: |
| 4 refinement steps, one output code | 70.3 ms | 23.4 ms |
| 16 refinement steps, one output code | 282.8 ms | 40.6 ms |

One fixed prompt, batch 1, ten repetitions after warmup; excludes encoder,
history prefill and frozen output decoding. This measures the factoring benefit
on that probe, not whole-run throughput or a quality-adjusted speed win.

Trainer `tools/train_conditional_head_flow.py` has source/data checks, periodic
atomic checkpoints and `--resume`. A real update-1 save/pause finished, and the
bounded run resumed from it. No update was discarded. Complete CUDA replay
equivalence is not claimed from the CPU in-memory test.

Code: `tools/conditional_head_flow.py`, `tools/conditional_head_evaluation.py`,
`tools/train_conditional_head_flow.py`, `tools/verify_conditional_head_flow.py`,
`tests/test_conditional_head_flow.py`. Evidence: `dump/conditional-head-flow-20261013`.

## Evaluation gate

Score the full prepared dev/test targets, then 34 self-fed complete answers
capped at 128 tokens. Twenty-two references exceed that cap; exact-reference
matching alone cannot judge valid paraphrases or longer answers. Inspect actual
text, stopping and factual content. Sampled-code decoder CE is not raw-token LM
NLL. Independently reload weights, reproduce all saved answer token IDs/code
floats, check frozen decoder tensors and Adam counters, and reuse the 32 paired
old-note correct/changed/wrong/blocked conditions while preserving the entire
later question. Four stochastic draws check sensitivity to the sampling seed.
No long run or useful generation claim follows from preflight speed alone.

## Completed bounded comparison

Both arms finished the planned two epochs. Content agreement below excludes the
literal end marker and compares sampled next-block outputs with one reference,
given genuine previous codes. This is not a language-model likelihood, general
factual accuracy, or a sufficient quality metric for stochastic text.

| Measure | Velocity head | Clean-code head |
| --- | ---: | ---: |
| Optimizer updates | 1,088 | 1,088 |
| Development-selected refinement steps | 4 | 16 |
| Test content agreement, first draw | 0.1652% | 0.7984% |
| Exact scored blocks / 36,134 | 0 | 198 |
| Exact self-fed answers / 34 | 0 | 0 |
| End-marker stops / 34 | 0 | 0 |
| Training + periodic dev/checkpoint seconds | 267.079 | 267.500 |
| Generation seconds for 34 answers | 23.016 | 32.922 |

Training timings exclude preparation and final dev/test/generation. Cached-input
answer timings exclude prompt encoding; every answer reaches the 128-token cap.
The preceding shared-transformer counterparts took 49.250 seconds for velocity
at four steps and 163.781 seconds for clean prediction at sixteen steps on these
34 prompts. The separate head reduces this sampling work, but both produce
incoherent text; this is not a speed improvement at a useful quality level.
Training itself is not faster than those controls (258.125/255.500 seconds).
Single local runs are not repeated hardware benchmarks.

Actual clean-head output for the Rina notebook example starts:

```text
 from_randice. No interest) ( 12 4j 1 But.ap||., So
```

The reference is "Rina keeps the notebook in the green box in the study."
The full output is incoherent and does not supply that location. The same problem
occurs across sampled broad-chat and other sources. The memory source's 10.51%
teacher-history content agreement includes fixed wording and acknowledgments;
it does not establish correct factual answers. The 198 exact blocks likewise
do not establish correct complete answers. Twenty-two references exceed the
generation cap, so zero exact full answers is not alone the failure diagnosis;
the actual output and failed stopping matter.

The result separates a computation improvement from the unresolved learning
problem. A smaller head is not enough to make the frozen reconstruction-code
space usable for broad generation under this budget. This limited pilot does
not prove an impossibility or a 23M capacity ceiling. The next proposed work is
evidence-supervised reading and short real grounded-answer construction, with
oracle-passage and learned-selection results reported separately. Another long
v4 run is not recommended from this result.

## Independent verification completed

`tools/verify_conditional_head_flow.py` finished successfully with zero optimizer
updates. Both final checkpoints reproduce all 34 saved answer token sequences
and predicted-code floats exactly. Final and last-checkpoint tensors match;
all Adam counters are 1,088; each model has exactly 23,209,216 trainable parameters;
the frozen decoder tensors match the original. All source/protected hashes pass.
Original v4/reference weights, production checkpoint and encoder/decoder are
unchanged. Trainer session 92890 and verifier session 33323 both exited 0; no
worker from this comparison remains running.

Four test draws give mean content agreement **0.1789% velocity / 0.8133% clean**.
The matched shared-transformer controls were 0.1684% / 0.8065%; these small
differences do not establish a meaningful quality improvement. All are poor
single-reference next-block results and cannot establish a language-model win.

The same 32 old-note cases were generated with correct, changed, wrong and
blocked memory. Both arms score zero in all four conditions. The correct set
contains 27 answerable and five missing-answer cases; all 32 changed-fact cases
are answerable. The wrong-memory set contains 13 missing-answer cases and the
blocked set contains 32. Scoring checks the full location or a narrowly defined
missing-answer phrase; it is not general QA judging. The entire later question
and its encoded representation are identical across each case's interventions,
although replacing/removing old records can change prefix lengths/positions.

Artifacts: `results.json`, `preflight.json`, `verification.json`, source archives,
and both complete model/optimizer checkpoints in
`dump/conditional-head-flow-20261013`. The broader generation goal remains open.
