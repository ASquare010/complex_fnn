> **Current status — 16 October 2026:** Growing/vector-generation experiments are retired. Their code, checkpoints and caches have been removed at the user's request. The text below is historical; use the [retirement record](retired_vector_experiments.md) for preserved results and the current token-memory64 direction.

# Using the frozen codec for language generation

The user's subsequently authorized [token-output growing-prefix pilot](growing_token_memory_pilot.md)
has now been implemented and independently verified. It predicts ordinary tokens
and re-encodes committed prefixes with frozen E; it does not predict new codec
vectors. At the same23,209,216parameter count and1,024matched fine-tuning updates
from the trained baseline, old-fact exact scores are16/204 baseline versus6/204
growing, with both0/204correct paired fact changes. It is a tested negative
alternative at this budget, not an untried proposed fix or proof of impossibility.
Descriptions below of a strict direct-code constraint refer to the earlier
experiments; the user explicitly relaxed that constraint for this pilot.

## Current recommendation

Keep the original codec as a verified representation of known text. Train the
consumer explicitly to answer from that representation. The next scientific
question is whether a reader can use a supplied correct passage on varied real
questions, separately from whether a generator can create a new valid code.
Neither replacing all token inputs with vectors nor lowering vector regression
loss has established both abilities in the completed experiments.

For the retained direct-vector-output constraint, the next bounded curriculum
should use short, grounded answers with evidence labels, rather than another
full-v4 restart. Keep both codec networks frozen and include all new heads inside
the 23,209,216 trainable-parameter budget. Start by supplying the correct source
block to the constructor; then supervise selection among several blocks using
answer-span locations. A small successful fixed-template toy is already in the
record; repeating that success would not establish transfer to ordinary text.

The [SQuAD 2.0 preparation](grounded_codec_protocol.md) is now independently
verified: 8,192 train, 512 dev and 512 exploratory test questions, disjoint
articles and literal-v4-overlap exclusions. All annotated dev/test evidence
tokens and end markers were recovered by the codec; two dev answer-token errors
were retained. The [bounded comparison](grounded_codec_results.md) is complete
and independently verified: raw 1/410 answerable exact, deterministic and flow
0/410 on held-out questions. The raw control also fails, so this does not isolate
compression as the cause. There is no new winner. Keep the trained token + memory
checkpoint as the practical language foundation; test evidence supervision there
before another full-vector restart. This first stage supplied the relevant full
passage; span labels were retained but not used as auxiliary losses or inference
inputs. Original weights/data are unchanged; no full-v4 run was restarted.

Initially use one relevant passage and short answers; next add distractors,
question paraphrases, unseen entities and varied passage formats. Test new
answer construction, not just selection of an existing answer vector. Later
include complete-sentence paraphrases, multi-fact questions and several generated
blocks. Safe synthetic counterfactual pairs can require the answer to change
when only the fact changes; unrestricted replacements in natural passages can
create inconsistent labels and should not be applied blindly.

Normalize the grounded-answer objective separately from ordinary language so
rare factual values are not overwhelmed by long generic answers. Evidence
selection CE is available when the source labels are known. For vector output,
retain genuine-code targets and frozen-decoder answer CE; compare deterministic
and stochastic construction with matched exposure. There is no established
universal coefficient for combining these losses. Select a mixture on held-out
fact correctness and language preservation, not total training loss alone.

Use the same question with correct, changed, wrong and absent evidence. First
evaluate with the correct passage supplied, then autonomous selection. Finally
evaluate the full answer with predicted-code feedback; true-history next-block
accuracy is insufficient. Full-position decoder features were already tested
without a broad-language improvement, so merely expanding those inputs again
is not the proposed fix: evidence supervision and real grounded tasks are the
new ingredients.

The more practical alternatives are (a) a task-trained compact-memory reader
with ordinary token output, or (b) selecting stored records and decoding their
text before answering. Both can use the existing codec; both relax the strict
direct-code design. They are options to discuss, not silent replacements. A
robustness-trained replacement codec is a third, distinct option that relaxes
the frozen-codec constraint. None is guaranteed to make a 23M general chatbot.

Research links below distinguish the evidence for these choices from our own
unresolved hypotheses. PISCO/xRAG use much larger language backbones; CALM changes
codec training and uses token-derived feedback; MAR's conditional denoising-head
evidence is in image generation. Their results are not a performance forecast
for this frozen 23M system.

In particular, PISCO's "no pretraining" means no additional compressor
pretraining stage. It starts with pretrained instruction language models, not
randomly initialized small chatbots. Our fresh bounded controls are useful for
comparing representations under matched exposure; their failure would not show
that a capable pretrained reader could never use these codes.

## Practical choices with the existing codec

1. **Decode selected memory before ordinary generation.** Store genuine encoder
   codes; select a few relevant records, reconstruct them with the frozen decoder,
   then let a token language model answer from the recovered text. This avoids
   asking the model to learn the codec's inverse just to read known facts. It
   still needs a trained retriever and a capable answering model. Re-expansion
   costs compute and attention slots; it is primarily a reliability/control
   route, not an established speed win. It relaxes vector-only predictor inputs.
2. **Teach a compact-memory reader with grounded targets.** Keep the encoder
   fixed and train the bridge/reader on passage-question-answer pairs, with
   evidence-selection labels where available and answers verified against the
   source. Use the best existing token model as initialization when testing
   this practical route, preserving a copy. Distill a raw-context teacher only
   on answers that it gets right. Include factual changes, absent evidence and
   distractors; evaluate answer content separately from common formatting.
   This can keep memory compact while generating ordinary tokens. It relaxes
   direct codec-vector output, but does not require retraining the codec.
3. **Keep the strict vector-input/vector-output experiment.** Learn a conditional
   distribution over new codes, decode each prediction with the original frozen
   decoder, and feed predicted codes back unchanged. This remains an open
   research problem in this repository. The new grounded comparison tests a
   narrower prerequisite, not a chatbot solution. If even supplied-evidence
   construction remains poor, do not spend another full-v4 run on a sampler
   tweak alone. A decoder-tolerant codec would be a separate experiment and
   requires explicit relaxation of the original frozen-codec constraint.

The existing full-position-decoder-feature experiments already showed that
expanding the representation by itself is insufficient. The important next
distinction is supervised fact reading versus new-code construction. A database
can preserve facts; it does not replace the language and reasoning skills needed
to interpret the retrieved facts.

## Constraints and prior evidence

The current constraint is 23,209,216 trainable parameters, original encoder and
decoder frozen, and direct codec-vector output. Retrieval, reconstruction and
generation are different capabilities; success at one does not establish another.

Latest architecture evidence: the
[separate conditional diffusion head](conditional_head_flow_results.md) has
completed both bounded arms. It substantially reduces repeated sampling work,
but produces incoherent complete answers. Faster output refinement did not fix
generation. Independent reload reproduced all saved answers/codes; both arms
failed the correct/changed/wrong/blocked old-fact controls. Four-draw content
agreement is 0.1789% / 0.8133%. No full-v4 restart is justified.

Earlier evidence: the [complete-answer comparison](full_answer_flow_results.md)
has finished its three bounded training arms, 1,088 updates each over 4,352
dialogues from all 17 sources. Independent reload/control verification passed.
It did not produce useful direct-code text: deterministic outputs
collapse to fragments and flow outputs are incoherent. A fresh equal-parameter
raw control learns more recognizable language but is also not a working chatbot.
This is broader than the earlier first-eight-token pilots, still a small training
budget, and not a proof that 23M is insufficient or that every latent generator
must fail. Do not recommend another full-v4 restart on these results.

The saved v4 grounding diagnostic recovered all 2,202 fact-body tokens through the
codec, yet the language model answered 0/8 simple questions when those facts were
only in older vectors. The same small test gave 8/8 with recent raw facts. Those
are diagnostic cases, not a universal compression result. See the
[data, loss and grounding audit](v4_grounding_diagnosis.md).

## What relevant research actually establishes

[CALM](https://arxiv.org/html/2510.27688v1) studies next-vector language generation.
Its authors report that reconstruction-only codes were difficult to model, so
they trained a robust variational codec. They also found raw latent feedback
weaker than a token-derived input representation. Its generative head uses an
energy objective and samples in one pass. These are meaningful differences from
our frozen codec and vector-only input constraint, not a drop-in validation of
our architecture. Its published experiments are substantially larger than ours.

[Large Concept Models](https://arxiv.org/abs/2412.08821) explores regression,
diffusion and quantized prediction in a sentence representation space. That
supports investigating distributions over vectors, but its semantic embedding
space differs from our exact-text reconstruction codes.

[Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337) motivates making
latent geometry easier to optimize through reversible transforms. Our bounded
geometry experiment tests this hypothesis while keeping all 256 dimensions and
both codec networks frozen; it does not reproduce the paper's image experiments.

[Diffusion Loss](https://arxiv.org/abs/2406.11838) demonstrates a small generative
head conditioned by an autoregressive model in image generation. It motivates
avoiding repeated context-backbone computation. It does not establish language
quality with our codec.

[SONAR-LLM](https://arxiv.org/html/2508.05305v1) is another relevant frozen-codec
study. Its TinyStories experiments include models with 11M and 34M **trainable**
parameters, excluding the pretrained codec. The paper's size labels refer to
larger token-model configurations, so those labels are not total system sizes.
Its reported story-quality evaluation still favors the ordinary token model.
This is evidence that small trainable predictors can be studied, not evidence
that our codec or data should achieve the same results.

The [released training implementation](https://github.com/FusionBrainLab/SONAR-LLM/blob/main/train_sonarllm_fast.py)
passes shifted ground-truth token sequences to the frozen SONAR decoder. That
decoder has token-level autoregressive conditioning, unlike our parallel codec
decoder. The [released generation implementation](https://github.com/FusionBrainLab/SONAR-LLM/blob/main/generate.py)
re-encodes each generated sentence before appending it to history. It therefore
does **not** validate the stronger requirement that both networks receive no
token inputs and raw predicted codec vectors can be fed back unchanged.

[ICAE](https://arxiv.org/abs/2307.06945) trains context compression using both
reconstruction and language-model objectives, followed by instruction tuning.
[500xCompressor](https://arxiv.org/abs/2408.03094) uses compressed KV states and
reports a quality cost at high compression. These support task-aligned memory
interfaces; neither establishes that an independently trained reconstruction
code automatically becomes readable to a fresh small language model.

[xRAG](https://arxiv.org/html/2405.13792v2) trains a bridge from document embeddings
to a frozen language model. Its stages include document reconstruction and
context-dependent answering, with distillation from the same model reading raw
documents. The published backbones are much larger than ours. The transferable
idea is explicit training to use the representation, not a claim that one
compressed vector is automatically understandable to any language model.

## Candidate within the present constraints

1. Store frozen encoder codes with their actual lengths and order.
2. Use frozen decoder hidden states as the reader interface. Preserve useful
   position information instead of assuming a single average exposes every fact.
3. Run a causal context backbone to produce conditioning for the next block.
4. Use a small conditional stochastic head to generate the next codec vector.
   Allocate its parameters inside the fixed budget, not on top of it.
5. Decode with the frozen original decoder. Train and test the same predicted
   vector feedback format before claiming multi-block generation works.

This is a proposed experiment, not an implemented production replacement. The
existing position-memory model already tried a two-pass reader/constructor; a
new variant must identify the concrete change and beat that reference. Merely
renaming the same soft mixture of records would not be progress.

The [bounded reader diagnostic](decoder_reader_flow_results.md) compares original
compact-code inputs against all valid frozen-decoder position states, keeping
the output predictor, training targets and budget fixed. Records remain separate.
The two completed 2,000-update arms do not improve held-out generation; the
one-pass model memorizes checked training answers while transferring poorly.
Expanding input positions alone is therefore not a demonstrated fix. The tiny
training protocol also prevents a broad conclusion about reader capacity. Any
next reader training must use more representative tasks and complete answers,
with consistent history/feedback block formats, before attempting to recover
efficiency through selective expansion or a learned resampler.

First separate two failure modes on varied real text: can the reader recover
the requested information, and can the generator construct an answer when that
information is supplied correctly? Then test them jointly with held-out layouts,
wording and values. Ordinary text continuations need a broader, source-stratified
sample and full answers, not only the first eight tokens from chat openings.

## Decision gates

Compare the same source information and target positions, fixed trainable budget,
data order and supervised-token exposure. Report generated examples, answer/fact
accuracy, correct/changed/wrong/blocked old memory, and end-to-end runtime. Keep
the current question fixed for old-memory interventions. Decoder CE on one
generated vector is not autoregressive language-model perplexity. Single-reference
agreement also cannot alone rank the quality of stochastic continuations.

A more reliable storage-only alternative is to retrieve and decode records before
ordinary token generation. That would relax the direct-vector-output requirement
and is not silently substituted here. A new robust codec would likewise require
an explicit change to the frozen-codec constraint. No claim that either change is
necessary or sufficient follows from the current bounded failures.

## Concrete interpretation and training sequence

The codec solves `known text -> code -> the same text`. A language model must
also select relevant facts, combine them with a question, and choose a plausible
new continuation. A deterministic squared-error predictor optimizes toward the
conditional mean when several continuations are possible; that mean need not
decode as any of them. A probabilistic head can represent alternatives, but does
not by itself repair an unreadable input interface or poor training coverage.

One frozen-codec research path is to reuse the decoder twice: expose
its position states to a trained reader for known memory, then use its original
output decoding for a newly generated code. Current work tests all separate
valid positions before attempting another bottleneck. If this works, selective
record expansion or a task-trained resampler can later reduce compute. A fixed
mean across record positions should not be assumed sufficient for exact facts.

Train a reader first on questions whose answers vary when only the old memory
changes, including unseen values/layouts and unanswerable questions. Keep the
question intact in wrong/blocked-memory controls. Then compare a deterministic
output head with a probabilistic output head from matched initialization and
supervision. Finally train complete continuations with consistent history block
lengths and test generated-vector feedback and EOS over several blocks.

This sequence diagnoses where learning fails. It does not assert a particular
loss weight, memory-attention multiplier, diffusion sampler, or parameter count
will solve general language generation. Do not launch another full-v4 run until
held-out answer quality and feedback work in a broader bounded pilot.

There is also a major data limit in the current matched flow pilots: 2,048
examples times8 scored positions is only16,384 distinct target positions, with
1,647 unique eight-token answers. Replaying them for2,000 updates is not new
language data. Selection requires an eligible64-token assistant span and then
scores only its first8 tokens, excluding short-only answers and later answer
content. Only everyday/chat sources enter training. These pilots can isolate a
representation or optimization change; their poor transfer cannot establish
that a23M model or latent generation has reached a capacity limit.

The next broader protocol should sample documents across sources, supervise
successive8-token assistant blocks throughout complete answers, retain short
tails/EOS, and use the same8-token representation for prior assistant blocks as
for generated feedback. Include held-out facts/layouts while also measuring
ordinary continuation. Freeze that protocol before comparing architectures.
Use an equally trained raw-token control for causal claims; the48,762-update v4
model remains a useful quality reference but is not that control.

The [full-answer preparation protocol](full_answer_flow_protocol.md) now records
a source-stratified selection with 2,578,931 training assistant labels. Full-size
GPU prediction/gradient checks and bounded code preparation have also finished.
These are zero-update preparation results, not a trained replacement.

## Practical use versus direct-vector research

The lowest-risk use of the existing codec is compressed storage: recover selected
records with the frozen decoder, then let a language model answer from their
text. This retains storage compression but spends attention on recovered tokens.
It relaxes the user's current vector-only input/direct-vector-output constraint,
so it is an alternative to discuss, not a change implemented behind that constraint.

A more ambitious memory interface exposes frozen decoder features to a learned
reader. Train the reader on reconstruction, factual questions, paraphrases and
continuation, with answers that change when only the stored fact changes. A
raw-context teacher can provide training targets when its answers are verified;
blindly distilling a weak or incorrect teacher would preserve its errors. Keep
ordinary language examples to detect a tradeoff in fluency. A separate memory
attention path or adapter may help optimization, but attention weight alone is
not evidence of useful retrieval.

Direct vector generation adds a second unsolved requirement: sampling plausible
new codes that decode coherently. Flow matching addresses a conditional output
distribution; it does not automatically teach memory reading or make a brittle
codec smooth. Test reading and generation separately, then together on complete
answers, with the same frozen codec and budget. Re-encoding generated text would
change the feedback contract and must be compared explicitly. No evidence so far
justifies another long run merely by adding MSE and CE or increasing iterations.

## Research decision: teach access before changing the output space

Review updated 2026-10-13. No new research optimizer updates or production
restart were launched for this review. The previous bounded preparation process
has finished successfully; its results are described in the linked protocol.

[PISCO](https://arxiv.org/html/2501.16075v1), especially section 4.3 and appendix I,
is directly relevant to the reconstruction-versus-use question. Its authors
obtained strong compressed-text reconstruction/keyword-continuation scores
without corresponding QA gains, and instead train on document-conditioned
answers distilled from a teacher. Its compressor and language decoder receive
adapters, and its backbone is billions of parameters. We can transfer the
task-alignment principle, not its size, speed or accuracy claims. Its frozen
decoder is an LLM, not our parallel reconstruction decoder.

Together with xRAG, this motivates a concrete change in supervision: supply
context whose facts actually determine the answer, then train the consumer to
answer from the compressed version. Reconstructing the whole input again does
not by itself teach entity binding, relevance selection or answer construction.
Likewise, making every input a codec vector removes the easy token route but
does not teach the missing operations; our vector-only experiments demonstrate
that this change is not sufficient under their tested protocols.

There are three distinct uses of the existing codec:

| Use | Encoder/decoder role | Main tradeoff |
| --- | --- | --- |
| Recover selected records | Store encoder codes; frozen decoder reconstructs selected text for a token LM | Most direct way to expose recoverable facts, but expanded records use token attention and direct-vector output is relaxed |
| Train a memory reader | Frozen encoder codes feed an aligned reader; frozen decoder can provide position-feature supervision or a diagnostic view | Keeps a compact interface possible, but needs downstream QA/continuation training; expanding all decoder states already failed to improve the small generation pilot |
| Generate new codes | Train a conditional stochastic code generator; original frozen decoder renders its new output | Preserves direct-vector output, but adds output-distribution, latent-precision and self-feedback problems to memory reading |

Recommendation for the fastest route to useful text: improve the existing
token-memory consumer with explicit grounded-answer training before changing
its output representation. This is an alternative to the retained direct-vector
constraint, not an authorized architecture replacement. For the stricter
vector-only research path, retain both codec networks frozen and isolate reader
learning from new-code construction before a joint complete-answer pilot.

The next memory curriculum should contain paired examples with the same question
and surrounding text, but changed names/values in the old records. Include one
record, distractors, paraphrased questions, unseen value combinations, different
record layouts, corrections and missing answers. Use programmatically verifiable
facts first, then document-based QA with checked answers. Mix ordinary full
answers for language preservation. Average grounded-answer loss separately from
ordinary-language loss; a long reasoning trace must not dilute a rare factual
answer to an incidental fraction of the objective. Tune that mixture on held-out
language and memory metrics rather than assuming a universal coefficient.

For an explicit reader diagnostic, first supply the relevant genuine record
(oracle selection), then test learned selection. Predicting the correct record
index is useful auxiliary supervision when the data supplies that label, but is
not successful answer generation. A reader-only temporary classifier can measure
held-out source/value understanding without asking the frozen decoder to render
a new code. Count its parameters if retained, and disclose this diagnostic as a
relaxation of direct-code output rather than a final architecture.

For direct-code generation, a conditional flow head can receive a context state
computed once and refine only a small eight-token output code. This efficiency
change is now tested in the separate-head pilot, within the same 23,209,216
trainable-parameter budget; the earlier shared FlowModel remains a control. Training
targets are genuine encoder outputs; those targets are loss-only for the current
query. Flow loss models a distribution, while auxiliary frozen-decoder CE checks
decoded text. Deterministic MSE alone estimates an average under ambiguity; a
small coordinate error is also not a guarantee of correct decoding.

At inference, sample a code, decode eight ordered tokens, append the predicted
code directly and repeat until EOS or a fixed cap. Verify complete self-fed
answers: testing only genuine-history next-block predictions cannot establish
that this loop works. If retained generated vectors and their decoded text drift
apart, explicitly compare a decode/re-encode feedback control instead of changing
the contract silently. Neither that control nor a new robust codec is currently
selected. CALM's robust codec and token-derived feedback are material differences
from our constraint; its success does not validate ours.

Before another long run, require a matched raw-token control on identical source
information and assistant targets, a task-trained compressed reader, and a
separate vector-output comparison. Keep question text/representation fixed while
changing, replacing or blocking only old facts. Measure factual answers, normal
language quality, EOS/repetition, multi-block feedback and complete wall time.
Do not choose a winner by MSE or sampled-vector decoder CE versus token LM CE.

## Practical next experiment, and what each result would mean

The current codec can still be useful as a store of recoverable text. A storage
key used for searching does not have to be the reconstruction code itself:
cosine distance between our codes has not been shown to represent relevance.
Keep reconstruction codes as values; evaluate any question-to-record selector
separately. A database can increase available facts, but cannot supply the
missing ability to select and use them by itself.

Be precise about the saving: our 256-dimensional FP32 code occupies 1,024 bytes,
whereas 64 uint16 token IDs occupy 128 bytes, before metadata. In this format the
code compresses attention positions relative to per-token model states, not disk
bytes relative to token IDs. Reconstructing selected records expands their
attention cost again. Storage, key lookup, encoder/decoder work, and resulting
KV memory must all be included in any efficiency claim.

Before another output-space redesign, use a single grounded-answer corpus to
separate reading from generation. Start with an oracle-selected passage so a
retriever cannot hide the reading failure. Compare (a) original passage tokens,
(b) the same genuine stored codes, and (c) text recovered by the original frozen
decoder. Keep the question and answer targets identical. This diagnostic would
use token output to remove new-code construction as a confound; it explicitly
relaxes the direct-vector-output requirement and has not been implemented as a
replacement. If raw and recovered-text controls both fail, first fix grounding
data/training. If they succeed but compact-code input fails, train the memory
interface. Only after that should learned record selection be added.

For broader reading supervision, [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
provides passages, extractive answer spans and deliberately unanswerable
questions. Its answer spans can supply evidence-location labels without trusting
a weak teacher's guesses. Split by article and check overlap with the existing
training corpus before treating it as held-out evidence. Use short answers first;
mix a separate, controlled language-preservation batch stream. Counterfactual
name/value changes should be restricted to examples where all affected evidence
and answer references can be updated consistently, not applied blindly to prose.

Use answer CE plus optional evidence-selection supervision. Normalize memory
answers independently from ordinary-language tokens, validate the mixture, and
require correct answers to change when only the old fact changes. Distillation
from a raw-input teacher is useful only on questions that teacher answers
correctly from the supplied passage; our weak fresh raw control is not a reliable
general teacher. Existing v4 contains broad text, but that does not replace this
explicit training objective or adequate coverage of informative answer tokens.

For the retained direct-vector research path, preserve all originals and do not
silently swap in token output, a token-autoregressive decoder, or re-encoded
feedback. The next gate is accurate answer-code construction when supplied the
correct passage on varied real reading tasks, then autonomous complete answers.
The earlier all-position input expansion was already tested and did not suffice;
proposing it again without a new objective is not a new solution. A smaller
conditional flow head now reduces sampling cost in the bounded test, but does
not establish useful generation quality. A robustness-trained
replacement codec is a separate, explicitly incompatible option if the original
codec must remain frozen; the existing codec's preservation is unaffected.

## Conditional diffusion head: a distinct architecture hypothesis

[Autoregressive Image Generation without Vector Quantization](https://arxiv.org/html/2406.11838v2)
uses a sequence model to compute a conditioning state, then a small jointly
trained denoising MLP to model the next continuous vector's distribution. Its
evidence is image generation, not reconstruction-code text generation. The
transferable design is to separate expensive history processing from repeated
output refinement. This does not establish that our frozen text codec will work.

The earlier shared flow implementation caches history keys/values, but every sampling
step still moves the noisy query through all eight transformer layers. A
separate small head is a real architecture change: the history backbone
computes one condition per output block, and only the head runs each denoising
step. Both must fit inside 23,209,216 trainable parameters. Keep causal language
order, the original frozen encoder/decoder, eight-token output codes and direct
predicted-code feedback; image-style random token orders are not assumed valid
for this text interface.

Before this architecture experiment, the
[clean-code/velocity diagnosis](data_prediction_flow_results.md) distinguishes
near-target denoising from generation out of noise. More solver iterations did
not rescue the shared model. The now-completed separate-head training
likewise fails to produce useful free text, despite its sampling speed benefit.
Quality must be established separately through actual generated answers and
old-fact interventions. Do not present the computation saving as a cure for a
memory reader that has not learned its task. See the linked new report for
implementation checks, controlled protocol, actual output and verification.
