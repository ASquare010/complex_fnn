# Step 2: context into a variable number of latent vectors

Status: **officially complete**, 2026-10-05, for the bounded reconstruction milestone.
The selected handoff is the **32-token span** position compressor: full model
7,292,928 parameters; encoder only 4,798,720. It recovered 1,000/1,000 development
paragraphs and 1,000/1,000 fresh document-disjoint confirmation paragraphs exactly,
plus all measured uniform, repeated-pattern and Unicode/formatting suites.
See [results](step_2_results.md) and [compression evidence](latent_memory_compression.md).
The original 8-token checkpoint and notebook default remain preserved.

Completion means tested reconstruction and a working encoder interface, not
universal losslessness, smaller UTF-8 storage, semantic search or direct LLM reasoning.
The next active work is the [Step 3 memory-using LLM design](step_3_memory_llm.md).
The standalone `latent-text-compressor` repository contains the release-ready
component, training recipe, notebook and Gradio app.

**Historical plan below:** retained to explain the original pooled baseline and
ideas that motivated the experiments. Its future-tense instructions and original
pilot status are superseded by this closure and the Step 3 draft. Streaming memory
integration now belongs to Step 3.

The following describes the preserved original baseline and future streaming plan.
The initial experiment uses ordinary PyTorch, width 256, two encoder and two
decoder blocks, Curve-Wide FFNs, and one learned pooled memory per eight tokens.
The eight-example exact-reconstruction gate passed at update 300. The fresh
2,000-update FineWeb-Edu pilot finished with reconstruction NLL 2.327738 and
0/1,000 exact validation recoveries. Correct memories improve over zeroed/shuffled
controls, but reliable unseen-text recovery has not been achieved.
The wider streaming design below remains
future research, not an implemented capability. Step 1 limitations still apply.

Start with the [notebook](../notebooks/paragraph_compressor_inference.ipynb),
[model](../dump/step2-cleanup-v1/src/models/paragraph_compressor/transformer.py), and
[current results](step_2_results.md). The initial dataset contains 50,000 training,
1,000 validation and 1,000 reserved test paragraphs, each 32-256 reversible BPE
tokens. No pretrained embeddings, custom kernels, similarity penalties or VAE
sampling are used. Word-for-word recovery is the objective, not a guarantee.

## Goal

Build a readable PyTorch streaming encoder that scans context in bounded sliding
windows and converts it into N compact latent memory vectors. The downstream
Transformer should consume those vectors instead of the full original context.
N should ultimately adapt to retained information: repeated content can reuse
memory, while distinct facts and relationships require more capacity. A count
based only on text length is an initial control, not the final objective.

Use the autoencoder idea of compression followed by reconstruction to teach and
test preservation. A traditional probabilistic VAE is not required. During normal
answering, do not reconstruct the entire source first: use the latent memory
directly. Original text can leave the model's active context after encoding;
this does not authorize deleting datasets or source documents from disk.

Scanning ten million tokens is a long-term scale target, not a demonstrated
capability. Register finite test lengths and explicit memory limits at each stage.

The eventual consumer is the Atlas reasoning core. First establish that this
representation preserves usable information at a worthwhile total resource cost.
This is a component experiment, not proof of the 1M-total versus 10B ambition.
Keep the [main charter](research_goal.md) intact. This new operational Step 2
investigates the charter's semantic-representation component before whole-system
integration; it does not claim that the charter's intervening stages are solved.

## Mental map and terms

```text
Available text: L tokens
        |
        v
Encoder: combine information across the input
        |
        v
Latent bottleneck: N vectors, each with d numbers, plus a validity mask
        |
        v
Reconstruction decoder: try to generate the original text

Deployment: text windows -> encoder -> latent memory + question -> answer
            no full-source reconstruction in the answering path
```

An **encoder** converts input into a representation. A **latent vector** is a
learned list of numbers, not a guaranteed human-readable concept. A **bottleneck**
limits how much representation can pass through. A **decoder** maps it back to
text. An **autoencoder** learns this encode/reconstruct pair. A **mask** marks real
slots so padding cannot be mistaken for information. **Reconstruction loss**
measures how poorly the decoder predicts the original tokens.

A **VAE** additionally learns a probability distribution for the latent state,
samples from it during training and penalizes deviation from a chosen prior
distribution. Its KL term is that distribution penalty. This can be useful, but
is not required for compression or concept learning. Start with a deterministic
autoencoder; consider a VAE only as a later controlled comparison. A powerful
decoder can ignore the latent state, so test dependence on the encoded input.

The user's first target is original-text recovery, with facts and meaning tested
as well. Perfect recovery at an arbitrary compression ratio is not promised:
fixed-precision representations have finite capacity, and some text has little
redundancy. A plausible paraphrase is not exact reconstruction.

## Streaming memory: the intended architecture

Read a window of W tokens, form candidate memories, and update the memory carried
from earlier windows. Release the window's token activations before reading the
next. An overlap may help preserve sentences and references across boundaries;
charge repeated tokens to encoding cost and test duplicate-fact handling.
The encoder must still read every input token at least once. Avoid full attention
over the entire source: local window attention has approximately O(LW) attention
pair work at fixed overlap, excluding projections and memory-update work.

The research hypothesis is that a learned update can reuse a slot for redundant
information, refine an existing slot, or allocate a new slot when current memory
cannot preserve a new detail. Specify a trainable allocation rule and a cost for
using slots before implementation; an informal novelty threshold is not yet an
algorithm. Test repeated facts, corrections, contradictions, names and numbers,
and relations whose two ends occur in different windows. A slot is not guaranteed
to correspond to one interpretable concept. Short dense text may need more slots
than a much longer repetitive passage.

The answer model attends to retained latent memory, the question and its generated
answer prefix. It still needs a mechanism for ordering output tokens; a normal
autoregressive decoder retains causal self-attention over that prefix. The aim is
to remove attention over all original source tokens, not remove attention entirely.
If it jointly self-attends over N memories, that portion still costs O(N^2).
Cross-attention from T output positions to memory costs O(TN), excluding other work.

An unbounded input stream is not infinite exact recall at fixed memory and cost.
For illustration, 10 million tokens at 100:1 token-to-slot compression still
produce 100,000 latent vectors. If memory grows indefinitely, attending to all of
it eventually becomes expensive. A bounded active memory would require controlled
merging/forgetting, or selecting a subset from an external latent store. The latter
also has storage and retrieval costs and must be evaluated separately. Neither
policy is selected or implemented in this draft.

The central tradeoff is retained detail versus memory and compute. Exact arbitrary
text recovery at fixed finite capacity cannot be guaranteed. Retain original-text
recovery as the previously chosen diagnostic, alongside factual and semantic tests;
do not silently replace it with plausible summarization. The full reconstruction
decoder is used in training/evaluation, not required on the deployment answer path.

## First design to discuss

Keep Curve-Wide's learned transformation and existing language model untouched.
Reuse its FFN design inside a separately defined encoder, without assuming its
language-model weights are already a good compressor. No C++, new kernels or
execution optimization in this research phase.

Use token embeddings and positions followed by a small encoder. For a bounded
initial context or individual streaming window, bidirectional attention may read all supplied input tokens:
they are already available. This would be a separate encoder attention policy,
not a change to the measured causal Transformer. A longer-context design must
explicitly address the encoder's own attention cost; compressed output alone
does not make full-input attention cheap.

Begin with **length-dependent pooling** as a transparent baseline:

1. Partition the encoded sequence into consecutive spans of r real tokens.
2. Average or learn a normalized weighted pool within each span.
3. Emit one vector per span: N = ceil(L / r). The final partial span includes all
   its tokens and excludes padding. Preserve span order and length metadata.
4. Let a small autoregressive decoder attend only to these latent vectors,
   positions and declared metadata when reconstructing the text.

Illustrative r=16: 128 tokens produce 8 vectors; 512 produce 32; 1024 produce 64.
This is adaptive to **length**, not yet to semantic complexity. It is a baseline
we can understand, not a novelty claim or the final architecture.

Proposed interface: encoder returns latents [batch, max_N, d], a slot mask,
original lengths and declared span boundaries. Decoder returns next-token scores
and generates to an end marker or the supplied length. Count all metadata as
part of the representation. Out-of-range inputs must raise a clear error or use
an explicitly tested chunking policy; never silently truncate the context.

Compare this against **learned latent queries**: N query vectors gather relevant
input information through cross-attention. Cross-attention means the output
slots ask for information from the input tokens. It is more flexible but adds
cost and needs position/coverage checks. Do not run both as an uncontrolled search.

Only after a baseline works, test **content-adaptive boundaries**. A small score
could accumulate evidence that a segment needs a new slot, with minimum/maximum
span lengths. The hypothesis is that repeated content needs fewer slots and
distinct facts need more. This is unproven: grammatical boundaries or a learned
score are not automatically measures of information. A discrete boundary needs
a specified training method; do not present thresholding alone as differentiable.
Compare adaptive and fixed layouts at matched average latent bytes.

## Prevent reconstruction shortcuts

The reconstruction decoder receives no original-text skip connection or hidden
copy of the input. During teacher forcing it may receive only the preceding
ground-truth output tokens, with a causal mask. Teacher forcing is a training
aid; report free generation starting from a start token separately.

Compare correct latents with zeroed and shuffled latents using the same decoder,
and with an unconditional decoder baseline. If performance barely changes, the
decoder is not demonstrating useful compression. Shuffling should use examples
of matched length so it cannot be detected solely from metadata.

For question answering or continuation, encode only the supplied context, never
the hidden answer or future continuation. Autoencoding loss is not ordinary
language-model perplexity: the encoder has seen the text it reconstructs.
Keep Step 2 tables separate from the existing language leaderboard.

## Measurements that decide whether it works

| Question | Required measurement |
| --- | --- |
| Can it reproduce text? | Full held-out reconstruction token NLL, free-generation token accuracy/edit distance, and exact sequence match, split by length. |
| Does important information survive? | Names, numbers, negation, ordering, who-did-what relations, and evidence-based QA; report each category separately. |
| Does it use the latents? | Correct versus zeroed/shuffled latent comparisons and an unconditional decoder. |
| Is the bottleneck smaller? | L/N, latent dimensions, precision, masks, positions and metadata bytes; report actual stored bytes. |
| Does the system save resources? | Encoder, decoder and consumer parameters; encode time, first-answer latency, allocated/reserved VRAM and repeated-query cost. |
| Does it generalize? | Disjoint documents/templates and unseen combinations of facts, with beginning/middle/end probes and longer-input tests. |

Fewer vectors do not necessarily mean fewer bytes than token IDs. Compare the
latent state with contextual activations or KV cache separately from raw text
storage. Account for the encoder's one-time cost and all downstream work; test
both one-use and repeated-use contexts. A reconstruction decoder is charged
during training/evaluation, and any decoder needed in deployment remains part of
the total system cost. Neither compression nor a learned slot count proves reasoning.

## Approved paragraph experiment and later streaming work

The approved first implementation uses width 256, two encoder/two decoder blocks,
span length eight, and a maximum of 256 input tokens. Training uses AdamW at
3e-4, weight decay 0.01, 100 warmup updates, clipping at one, seed 17, microbatch
four with four-step accumulation and CUDA BF16. The pilot runs 2,000 updates from
fresh initialization after exact recovery of eight short examples. Source/data
identities and actual hardware settings are recorded. Full validation uses all
1,000 held-out paragraphs; the test split remains reserved. This pilot does not
yet include an uncompressed trained control or a compression-ratio sweep.

After this bounded check, test multiple windows and memory carry before scaling
input length: compare independent window compression against memory updates at
matched latent bytes. Probe facts from early windows after later windows arrive,
cross-window relations, repetition and interference. Measure encoding cost versus
total source length, active memory growth, and answer cost versus retained slots.
Only then investigate content-adaptive slot allocation and larger input lengths.

The pilot establishes functionality and measures reconstruction quality. A later
controlled comparison must measure the quality/compression curve before learned
boundaries or VAE sampling. A resource-advantage claim needs a declared matched
reference and full-system cost accounting. Reject designs that ignore latents or
hide encoder/decoder costs. No further sweep is authorized by this document.

The next discussion should settle: initial context limits, which exact-recovery
rate is useful, acceptable QA degradation, whether to prioritize one-use or
repeated-use contexts, and the first fixed resource budget. The first mechanism
to investigate is representation compression; persistent storage and retrieval
policies remain later work.

## Related work to compare against

These establish relevant existing ideas, not that our eventual implementation
will be novel:

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): the VAE framework and its probabilistic latent objective.
- [Perceiver IO](https://arxiv.org/abs/2107.14795): attention through a latent bottleneck and decoding by output queries.
- [In-context Autoencoder](https://arxiv.org/abs/2307.06945): compressing language context into memory slots for an LLM.
- [Compressive Transformers](https://arxiv.org/abs/1911.05507): compressing past memories for long-range sequence modelling.

Use these as prior-art baselines when the design is fixed. Our contribution, if
any, must be supported by a specific difference and a fair measured advantage.
