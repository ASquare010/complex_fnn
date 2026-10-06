# Step 3: a tiny language model with external memory and bounded recurrent reasoning

Status: **research goal and design draft**, 2026-10-05. Step 2 is officially complete
as a bounded reconstruction milestone. This document begins the integration stage;
it does not authorize an unspecified training run or freeze every architecture choice.

## Goal and parameter contract

Build an autoregressive language model that learns reusable language/reasoning
operations, reads relevant facts from external compressed memory, and allocates
**one to four shared-weight latent loops per output token**. Permit memory creation
and reading; no memory update or deletion in this stage. Use ordinary readable
PyTorch and Curve-Wide. The hardware ceiling remains the available 8 GB GPU.

The user's current ambition is **at most 10M learned system parameters, eventually
beating a fixed roughly 10B reference on a declared benchmark suite**. This is an
ambitious target, not a conclusion supported by the component results. The preserved
[original charter](research_goal.md) says 1M; the user revised the operational target
to 10M for Step 3. Do not silently substitute a 10M core plus an uncounted encoder.

Count all unique learned parameters: frozen encoder, token embeddings, generator,
recurrent core, adapters, key/query encoders and action/halting heads. Shared weights
count once. Also report trainable parameters and GPU-resident parameters separately.
The selected span-32 encoder has 4,798,720 parameters, leaving **5,201,280** under a
10M system cap for the remaining learned modules if there is no parameter sharing
with that encoder. An optional reconstruction decoder is not needed in the normal
answering path; if used at runtime, include its additional parameters and cost.
External storage, indexes, CPU RAM, GPU working state and total compute are separate
reported resources, not free or hidden parameters.

Memory can reduce pressure to memorize changing facts in weights. It cannot remove
the need to learn language, operations, relationships and retrieval. An FFN is a
learned transformation, not exclusively a fact database. Fewer layers and a
10M-versus-10B advantage are hypotheses to measure, not automatic consequences.

## What Step 2 supplies

Freeze the selected 32-token encoder initially. It maps a full 256-token window to
eight ordered 256-feature vectors, with explicit lengths. The full reconstruction
model has 7,292,928 parameters; only the 4,798,720-parameter encoder is needed to
create memories. See [closure evidence](step_2_results.md) and the
[compression sweep](latent_memory_compression.md).

The separate `latent-text-compressor` repository packages this component. Preserve
its trained model identity and tokenizer with every stored bundle. The decoder's
successful recovery establishes usable reconstruction information; it does not
establish that an unrelated language model understands the vectors or that their
distances are semantic relevance scores. Step 3 must teach and test that interface.

## Mental map: address, search key, payload

A memory record has three different jobs, which should not be confused:

| Item | Meaning | Who creates it? |
| --- | --- | --- |
| Record ID | Stable database address; an opaque identifier | Storage code, e.g. source/content/version hash |
| Search key | Small representation used to rank relevance | A key encoder trained for retrieval, or an initial lexical index |
| Payload | Ordered compressed vectors preserving the source window | Frozen Step 2 encoder |

Also retain document identity, token offsets, window length, encoder/tokenizer/key
versions and source provenance. IDs and offsets are metadata, not semantic features
that the model can be expected to guess for unseen records.

Example: the source says "Amina ordered 17 batteries. Nine arrived; eight are missing."
The database stores a bundle containing that information and a searchable key.
For "How many are missing?", the reasoning state produces a query. A search service
compares that query with keys and returns the top matching bundles plus IDs. The
language model attends to the returned payload, combines the information, and emits
"Eight." It does not need to predict the database's arbitrary ID in its vocabulary.

```text
source documents -> sliding encoder -> append-only store of keys + ordered payloads
                                                  ^
question + short answer prefix -> query head -> search service
                                       |          |
                                       |      selected bundles only
                                       v          v
                              shared reasoning core
                                       |
                           answer now or loop (maximum 4)
```

### How retrieval becomes learnable

Start with oracle supporting-record IDs to isolate whether a tiny consumer can
answer from the codes at all. Then train a query/key pair with positive supporting
records and hard negatives (same names but different quantities, dates or negation).
A contrastive loss should rank the correct key above these alternatives. A small
learned projection of frozen encoder features is one candidate key function, not
an already validated semantic retriever. Keep a lexical search control to expose
whether the learned retriever improves anything. Any additional key network counts
against the parameter budget; no hidden pretrained embedding model.

Top-k database lookup is discrete: ordinary backpropagation through cross-attention
does not automatically train the search decision. Train retrieval with explicit
support labels first; train answer generation from a mixture of correct and
actually retrieved memories, including irrelevant and missing-memory examples.
Report retrieval recall separately from answer quality. When changing the key
encoder, rebuild/version the index; incompatible query and stored key spaces must
not be silently mixed. Cache retrieval for an identical query/store version.

## Memory operations: create and read only

Creation initially means ingesting a supplied document: split it, encode it and
append immutable records. The storage layer assigns IDs and may return an existing
record for the exact same source span, content and encoder version. It never relies
on exact floating-point equality or aggressively merges near-duplicate facts.

After fixed ingestion works, train an explicit CREATE action selecting an available
source span, and a READ action carrying a query. The host validates span boundaries,
provenance and budget before acting. Do not initially persist the model's unverified
answers as facts. No overwrite, deletion, learned forgetting or in-place corrections.
Conflicting records coexist with provenance; selecting between them is a later
policy question, not a silent mutation. Distinguish persistent store writes from
ordinary temporary recurrent-state updates.

## Sliding encoding and bounded active memory

Begin with the already supported independent windows of at most 256 tokens and
stride 256. Test stride 224 (32-token overlap) as a separate boundary-coverage
experiment. Overlap repeats storage/encoding work and must be charged. Keep global
source offsets, document boundaries and per-window ordered payloads. Do not join
unrelated documents or claim that independent windows understand cross-window
relations automatically. Test a relation whose subject and value straddle a boundary.

Scan one bounded window at a time and write results to CPU/disk storage. Keep the
large store and index off GPU. Proposed starting retrieval budget: four windows per
read, eight unique windows active at once (at most 64 span-32 vectors), and no more
than one new read per loop. These are draft budget choices, to fix before training.
At capacity, use a declared deterministic working-set selection rule, retaining the
persistent records on disk. Dropping a GPU copy is not deleting the database record.

Measure at 1k, 4k and 16k source tokens first, then increase the corpus while keeping
GPU limits fixed. Encoding still reads every token; storage and search cost grow.
A sliding encoder enables bounded-window processing, not infinite exact recall at
constant total resource cost. Keep one-shot encoding and repeated-query costs separate.

## Where to retrieve, and what loops reuse

Preferred starting schedule: form an initial query from the question/prefix before
loop 1; optionally issue a new query **between loops**, then use the results in the
next loop. Each iteration reuses the same small block stack and memory reader.
Try a width-256 two-block shared core first, subject to an actual parameter count.
A learned loop-position signal and a residual connection can distinguish successive
refinements without creating four copies of the core's weights.

| Placement | Benefit | Cost or limitation | Initial role |
| --- | --- | --- | --- |
| Once before answering | Simple; reuse a fixed memory set | Cannot follow a missing second-hop fact | Required baseline |
| Between loops | Later query can depend on earlier retrieved facts | Up to four searches per token; needs a read gate/cache | Preferred hypothesis |
| Every layer/block | Frequent opportunities for interaction | Many searches and difficult credit assignment | Defer |
| Only after the final loop | Simple final lookup | Retrieved evidence cannot affect that token's completed reasoning without another pass | Do not use as primary design |

A database search and attention over already-fetched vectors are different costs.
Several blocks may reuse the same fetched vectors without another database call.
The READ decision and STOP decision should be distinct: a model may refine its
state using existing evidence without searching again. Record both counts.
Generated answer tokens remain autoregressive. Internal loops refine hidden states;
they do not append four visible reasoning tokens for every answer token.

## Preventing "always loop four times"

First measure fixed-depth 1/2/3/4 performance with shared parameters. We need to know
whether extra computation helps and on which inputs before asking a gate to allocate it.
During initial training, supervise every exit (e.g. equal-weight mean loss over all
four exits, or a balanced sampled-depth control). Do not let an initially random gate
starve the other depths of training. Running all four exits during training does not
produce early-exit training savings; the intended benefit is adaptive inference.

Then freeze the core temporarily and fit a stopping gate on separate training/calibration
examples. A proposed target is the earliest depth within a small declared margin of
the best available loss/correct answer, accounting for compute and retrieval cost.
A locally small improvement is insufficient if the third/fourth loop later provides
an important gain. Use the full four-step rollout to create training targets, while
the deployed gate sees only the state/evidence available at its current step.

For a gate-based objective, define the conditional stopping probability h_t from
the current state. Convert it to actual exit probabilities:

    p_1 = h_1
    p_2 = (1 - h_1) h_2
    p_3 = (1 - h_1)(1 - h_2) h_3
    p_4 = (1 - h_1)(1 - h_2)(1 - h_3)

The final step receives all remaining mass; the model must stop by four. A draft
minimized objective, after balanced depth training, is:

    L = sum_t p_t * L_task(t)
        + beta * KL(p || prior)
        + lambda_loop * E[executed loops]
        + lambda_read * E[database reads]

Cost coefficients, margins and the deployment threshold are selected on calibration
validation data, not the final test. For stochastic halting, sum(t*p_t) is the expected
loop count. A deterministic cumulative-probability threshold has a different actual
execution profile; measure its real loop/read counts rather than equating the two.
Uniform-prior KL can support exploration early; lower its strength later so easy
inputs are allowed to exit early. It does not by itself establish optimal stopping.

**Image correction:** with beta > 0 and a loss that is minimized, the screenshot's
`- beta * KL(p || uniform)` encourages divergence from uniform, not diversity.
Use `+ beta * KL`, equivalently `- beta * entropy` up to a constant. KL applies to
the whole normalized exit distribution. The plotted probabilities alone do not
show that difficult inputs receive more useful compute.

Monitor exit histograms, per-depth loss/correctness, easy-versus-hard behavior,
mean/p95 loops and reads, and the quality/latency curve. Compare with always-one,
always-four and matched-average-budget random/fixed policies. Always-four behavior
may be justified for some difficult inputs; on easy inputs it must earn its extra
cost. Gate confidence or low output entropy alone is not proof of correctness.

## VRAM and the KV-cache issue must be resolved before implementation

Repeated weights keep parameter memory small, but training unrolls still retain
activations; four loops are not automatically free in VRAM. Use bounded microbatches
and ordinary activation checkpointing where measured necessary. Target under 6 GiB
peak allocated memory for headroom, and report peak reserved and total device use.

A conventional looped Transformer can require a different key/value (KV) cache for
each loop depth. Reusing one cache with states from an incompatible depth can change
the model. Variable token exits also leave missing deeper cache entries. Do not
claim cache savings while secretly computing skipped loops just to populate them.

Start with a clear fixed-depth reference and its matching training/inference cache
semantics. For adaptive per-token exits, a promising separate design is one causal
prefix-memory bank computed independently of the current token's refinements; every
loop queries that fixed bank and the selected external memories. Then later tokens
do not require missing per-loop histories. This is a changed architecture and must
be trained that way, not a post-hoc cache swap. Another conservative control chooses
a fixed loop budget for an entire answer, but that is not the desired per-token gate.

**Unresolved design choice:** choose and test the cache/state contract before a long
run. Require cached-versus-full-prefix numerical checks, causal-mask tests and true
measured savings from early exit. A cache-free short-context prototype can test the
quality hypothesis, but its latency must not stand in for a practical decoder.

## Training sequence and leakage controls

1. **Language foundation:** establish a tiny causal LM baseline using Curve-Wide,
   tied token/output embeddings and bounded context; record ordinary held-out NLL.
2. **Memory reading:** hold the Step 2 encoder fixed; provide oracle memories and
   train QA/continuation from them. Compare compressed payloads with raw-token memories.
3. **Retrieval:** learn/query actual keys; introduce distractors, missing facts and
   unseen entities. Prove the model uses newly ingested facts with weights frozen.
4. **Looping:** test shared 1/2/3/4 computation at matched data and disclosed compute.
5. **Adaptive compute and reads:** fit gates after useful exits are trained; calibrate
   quality versus cost and enforce the four-loop/read budgets at inference.
6. **Creation:** add supervised source-span admission actions after read-only memory
   use is reliable. Do not add update/delete actions in this stage.

Use controlled synthetic facts/relations first, with disjoint entities/templates
and random fact assignments per episode so fixed weights cannot answer by memorizing
the evaluation facts. Follow with natural document QA and ordinary language evaluation.
FineWeb reconstruction examples alone do not teach query selection or instruction
following. Choose named datasets, revisions, finite budgets and test splits in the
next experimental protocol; this draft does not invent a completed training budget.

For causal LM loss, the bidirectional encoder may only see windows fully available
before the prediction boundary. Encoding a window containing future target tokens
would leak the answer and invalidate NLL. In document QA, access to supplied evidence
is intended; the reference answer, future answer tokens and gold record labels must
not be fed into the query/gate at inference. Perturb future tokens in a causality test.
Freeze test memory snapshots and prevent evaluation records being used for training
retrieval positives or task supervision. Newly ingested evaluation evidence is allowed
only under the declared memory-access protocol shared by the relevant baselines.

## Evidence gates and fair comparisons

| Gate | Required evidence before moving on |
| --- | --- |
| Direct latent use | Oracle-memory answers beat zeroed/shuffled memories; compare raw-token memories and source reconstruction |
| External fact access | With weights frozen, answer new facts introduced only through CREATE/ingestion; identify missing/conflicting evidence |
| Real retrieval | Recall@k and end-task quality with actual search, hard negatives and cross-window/multi-hop questions |
| Useful recurrence | Fixed-depth curves show where extra loops improve outcomes, at disclosed extra compute |
| Adaptive allocation | Lower measured compute/reads than always-four within a predeclared quality margin; no hidden full-depth cache work |
| Resource control | Total unique parameters <=10M; GPU memory within budget; CPU/index/storage, encode/search/answer latency all disclosed |
| Large-model comparison | Frozen named ~10B checkpoint/revision and protocol; equal-memory and native baselines; category-wise quality and uncertainty |

The initial bar is a functioning small memory-using language model, not victory
over 10B. Compare no-memory versus oracle/retrieved memory, compressed versus raw
payload, one versus four versus adaptive loops, and retrieval once versus between
loops. Keep parameter-matched and compute/latency-matched comparisons separate.
Give the large reference the same source information and suitable retrieval access
as an additional control. If it cannot run locally, a separately identified remote
or quantized run is needed; do not claim an unperformed comparison. A win on a narrow
retrieval benchmark is that specific win, not general 10B capability.

## Research grounding and open decisions

The supplied transcript discusses [Ouro / LoopLM](https://arxiv.org/abs/2510.25741).
Its paper uses shared recurrent layers, an entropy objective followed by gate
refinement, and discusses depth-specific caches. Those results motivate experiments;
they do not establish success at our size. The paper's loss equations (3) and its
uniform-prior derivation confirm the KL-sign correction above.
[PonderNet](https://arxiv.org/abs/2107.05407) studies learned halting with a prior.
[Recurrent Depth](https://arxiv.org/abs/2502.05171) studies additional latent computation.
[RETRO](https://arxiv.org/abs/2112.04426) supplies precedent for retrieved external
information; it is not evidence that our reconstruction latents are suitable keys.

Before solidifying the implementation, settle: the causal cache/state contract,
the tiny core size within the total cap, key/query training data, maximum active
bundle/read budgets, source-admission policy, and the first named dataset/update
budget. The starting preferences above are proposals, not measured conclusions.
