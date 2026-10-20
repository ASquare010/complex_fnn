> **Current status — 16 October 2026:** Growing/vector-generation experiments are retired. Their code, checkpoints and caches have been removed at the user's request. The text below is historical; use the [retirement record](retired_vector_experiments.md) for preserved results and the current token-memory64 direction.

# Raw64 gated cross-attention ready for user training (2026-10-16)

[Implementation and commands](../src/models/token_memory_64_crossattn/README.md).
The user selected raw64 first, deferring any larger-window experiment. Raw causal
self-attention and memory cross-attention are separate. A nonlinear pointwise
reader and zero-initialized channel gates at layers 4 and 8 receive the budget
freed by reducing the four-branch FFN widths to 1214/1213. Total trainable count
remains exactly 23,209,216. Fresh seed47, same original data/order/labels/effective
batch32/two epochs/48,762 updates. Reuses the verified plain64 FP32 cache; no
growing-prefix cache changes or full training launch. Current runs untouched.

Fourteen CPU tests pass, including full-size initialization/count, independent
packed/reference logits and all gradients, causality, cached generation and tiny
serialized two-epoch resume. Full-size real-data CPU preflight passes with zero
optimizer updates (logits max 1.34e-6, gradients max 7.08e-8). GPU capacity and
end-to-end speed are unmeasured; no factual improvement is claimed. This is a
parameter-allocation and attention-path experiment, not a one-variable ablation.

# 64-token attention residuals evaluated; no factual-recovery gain (2026-10-14)

[Full three-model report](token_memory_64_attnres_results.md). User intentionally
ends at37,045updates; no further training requested or launched. All batches/
targets/rates match older runs through this step. At equal37,000-step validation,
original512/plain64/attention residual64 accuracies60.3123/52.7197/52.9645%.
Retained-checkpoint fulltest61.00/53.51/53.04%; older two have48,762updates, so
this is not an equal-update generation comparison. All three0/103 notebook
exact and0/32 changed facts. Missing response13/25,17/25,22/25; false missing
on answerable facts49/103,65/103,87/103. New other16 answers manually wrong.
Recent color8/8,0/8,0/8; old color0/8,1/8,0/8. No reliable factual reading.
All312 old outputs/testmetrics reproduce; public NEWmodel paths, depth-reference
parity, blockedcontent invariance, serial replays and artifact hash checks pass.
Evaluator38788 finished0; no worker remains and no optimizer updates performed.
Keep original512 as the strongest retained language foundation.

## Historical attention residual preparation (2026-10-14)

The user requested the codec's attention residual mechanism in the64-token LM
before retraining. [Model and command](../src/models/token_memory_64_attnres/README.md).
The exact encoder `mix_depth` function selects among initial embeddings and
earlier attention/FFN outputs, including the final mixture. The codec decoder
itself has ordinary residuals. Sixteen zero-initialized512-wide queries add8192
parameters, totaling23,217,408. Original widths and fresh seed47 core tensors
are unchanged. This separate version preserves the completed plain64 run.

Original data/order/labels/effective32/twoepochs/48762updates/cache remain.
Only depth mixtures are activation-checkpointed to control memory. Six new plus
seven existing CPU tests pass; full-size GPU packed/reference logits and all
gradients pass, with every query receiving gradients and zero optimizer updates.
Micro16 fits at5.381GiB allocated with Adam-sized buffers. Depth mixing adds
compute; warmed first-batch forward/backward about.75s excludes Adam/eval/saving/
preparation. No quality gain or full-run speed guarantee. Training is left for
the user; original checkpoints, codec, data and caches remain unchanged.

# 64-token raw rollover completed: original performs better (2026-10-14)

[Full comparison](token_memory_64_results.md). The user completed48,762updates.
Every logged batch, label count and learning rate matches the original512 run.
Full test1,757,421labels: CE1.80695 original vs2.21695 new; accuracy61.00% vs53.51%.
All17 development/test sources worsen. Correct-memory notebook exact0/103 both;
changed-old-fact0/32 both; missing-fact exact13/25 vs17/25. Short color prompts
8/8 vs0/8; old compressed color0/8 vs1/8, but blocked1/8 both. No reliable factual
reading improvement. New location-substring hit has the wrong person's name.
Read-only checks pass; checkpoints/source/encoder unchanged, zero optimizer
updates in this comparison. Retain original as the language foundation.

## Implementation and preparation (2026-10-13)

[Model and commands](../src/models/token_memory_64/README.md). User chose fresh
initialization matching the baseline. Exactly23,209,216trainable and identical
seed47 tensors; samev4 corpus/tokenizer/masks,32original segments perupdate,
twoepochs/48,762updates/361,357,104assistantlabels. Every64tokens becomes one
frozen-encoder vector and clears the raw buffer; the next token has only memory
and BOS. This differs from rebuilding a growing vector after every token.
Eight independent64-token raw windows pack into one original512-token segment;
memory is causal and future codes/other raw windows cannot leak. Up to39memory
vectors reflect moving recent text into memory sooner, not extra source tokens.
SevenCPUtests and zero-update fullsizeGPUcheck passed before the user's full run.
PersistentFP32cache4.756GiB is complete and independently verified. All original
train/dev labelcounts/cachehashes match; fresh128blocks/split re-encode within
1.955e-5FP32. Cacheverification receipt:
dump/token-memory-64-cache-verification-1791912914940285600.json.
The GPUprobe's heaviestmicro16 used
3.838GiBallocated/3.988reserved withAdam-sizedbuffers; excludes optimizerstep.
Cached batch timings~.294s new vs.294-.320s baseline are not an end-to-end speedwin.
Evidence: dump/token-memory-64-preflight-1791911684777184500.json.

# Growing token-output prefix pilot verified: no improvement (2026-10-13)

The user's requested design is implemented: predict an ordinary token, re-encode
the committed active prefix with frozen E, replace its vector until64tokens,
then append it to completed memory. This explicitly authorized token-output
experiment differs from the older direct predicted-vector feedback experiments.
Both arms start from the trained v4 baseline with exactly23,209,216trainable
parameters; same4,096examples/eightpasses/1,024updates/265,296labels each. Training
prefixes are cached once; live generated prefixes are re-encoded.

Test answerable old facts: baseline16/204, growing6/204. Real reading:1/209 versus
0/209. Both0/204 for answering original and changed facts correctly; exploratory
fresh histories with familiar names/wording also both0/102paired. The growing arm
does not beat the baseline and often fails output stopping. Its short-prefix
codec oracle has18errors/15,968tokens, although all38,063checked full fact-record
tokens recover exactly. SevenCPUtests, fullsizezero-updatepreflight, checkpoint
and source identity, allAdam1024counters,64savedanswers+128controlsperarm replay,
target/blocked invariance and cached/full generation checks passed. No new GPU
worker remains from this pilot. [Full report](growing_token_memory_pilot.md).

User-authorized cleanup removed only seven redundant completed optimizer states,
preserving their final weights and evidence;2,051,418,249bytes freed. Original
baseline/codec/data/pausedproduction hashes are preserved by the pilot. No full-v4
restart or broader language-quality claim follows this negative bounded result.

# Real grounded-answer comparison verified: no replacement (2026-10-13)

[Preparation and protocol](grounded_codec_protocol.md): official SQuAD2 subset,
8,192 train / 512 dev / 512 exploratory test, disjoint articles and conservative
literal-v4-overlap exclusions. Independent source, mask, causal history and
sampled re-encoding checks passed. Every annotated dev/test evidence token and
end marker is recoverable; two dev answer-token errors are retained.

Three fresh 23,209,216-parameter arms compare raw tokens, deterministic code
prediction and conditional clean-code flow. Both vector arms preserve the
original frozen E/D, no token inputs and unchanged predicted-code feedback.
2,048 updates each, batch32, eight epochs of exactly the same questions/answers;
full-v4 production remains paused1001. Relevant passage is supplied; evidence
span labels are not inputs or auxiliary training labels in this first stage.
Final test answerable exact: raw1/410, deterministic0/410, flow0/410. On53
answerable intervention cases allarms scorezero withcorrect/wrong/blockedmemory;
twoextra flowdraws andtenchangeddatecases alsofail. All64replayedanswers/codes
perarmmatch; Adamcounts/frozencodec/protectedhashespassed. Bothjobsterminalsuccess.
The rawcontrolalsofails, so thisdoesnotisolatecompressionorproveacapacityceiling.
The trainedtoken+memorycheckpointremainsthebestretainedlanguagefoundation.
[Fullresults](grounded_codec_results.md). The broader language-generation goal
is unresolved; no further training was launched.

# Separate conditional head verified: speed gain, no useful text (2026-10-13)

[Completed comparison](conditional_head_flow_results.md): two new 1,088-update
arms retain exactly23,209,216 trainable parameters, original frozen E/D, no token
inputs and direct eight-token-code feedback. Reallocating2,621,440 parameters
from backbone FFNs to two small conditional MLP blocks computes history once
per output code. Cached-input generation of34answers takes23.016/32.922seconds
versus49.250/163.781 for matched shared-head controls at4/16refinement steps.
All outputs remain incoherent, so this is not a quality-adjusted speed win.
Training itself is not faster in these single local runs.

Both arms produce0/34exact answers and0/34end-marker stops; four-draw content
agreement0.1789%/0.8133%. All old-note correct/changed/wrong/blocked conditions
score0. Independent verification reproduces all34answers and codefloats perarm,
checks final/last/Adam1088, exactbudget, originalfrozenD and protectedhashes.
7CPUtests and zero-update full-size GPUparity passed; no originals changed.
Trainer92890/verifier33323 both finishedexit0. Production remains paused1001.

[Research direction](frozen_codec_research_direction.md): task-aligned reading,
real short grounded answers and evidence-selection supervision should be tested
before another full-v4 run. Oracle passage versus learned selection must be
reported separately. SQuAD2 preparation and the initial comparison are above.
Token-output reading or a robustness-trained newcodec are explicit alternatives
that relax existing constraints; neither was silently implemented. No winner.

# Clean-code flow comparison verified (2026-10-13)

[Completed diagnosis and comparison](data_prediction_flow_results.md): two new
matched1,088-update arms keep23,209,216trainable parameters, originalE/D frozen,
no token inputs and direct predicted-code feedback. Velocity/clean-loss versus
clean-code-head four-draw content agreement is0.3590%/0.8065%; both fail all old-note
conditions and produce incoherent complete answers. All34 saved answers and code
floats perarm reproduce exactly; Adam counters, original decoder tensors and
protected hashes pass. No workers remain from trainer44088/verifier14979.

The genuine-code diagnostic still reconstructs99.958% of dev content tokens.
Near-target denoising works even without context, while generation from noise
fails; more solversteps do not rescue it. Loss weighting/headparameterization
alone is not a fix. [Research direction](frozen_codec_research_direction.md)
proposes explicit grounded reading and, for retained direct-vector research,
a separate small conditional diffusion head. That architecture has now been
tested above and failed the quality gate. Production remains paused1001, originals
unchanged, broader goal open; do not launch another fullrun from these scores.

# Full-answer frozen-codec comparison complete (2026-10-13)

[Results and independent verification](full_answer_flow_results.md): three fresh
23,209,216-parameter arms, frozen original codec, same4,352 training dialogues
from17sources, two epochs/1,088updates each. Deterministic MSE+CE and flow+CE do
not produce useful generated language. The fresh raw control also fails the
32old-note cases, so this pilot cannot isolate compression as the retrieval
cause. True-code EOScontrol2 reconstruction failed; a reversible literal[END]
view fixes all1,516dev/test end markers without changing originalv4/codec, but
does not rescue vector generation. All saved answers/codes reproduced on reload;
four flow draws remain poor. Original files unchanged; production paused1001.
No long run recommended. [Research review](frozen_codec_research_direction.md)
separates storage, learned reading, and new-code generation, with a proposed
grounded-reading comparison rather than another blind output-space redesign.

# Reversible latent geometry: bounded follow-up (2026-10-13)

[Expanded frozen-decoder reader result](decoder_reader_flow_results.md):
same23.209M budget/original frozen codec, separate valid position states instead
of compact-code inputs. Two 2,000-update arms finished and passed independent
reload verification. Expanded one-pass/flow agreement is 1.22/0.44%, versus
3.56/1.51% for compact controls on the same new 256 cases; all have zero exact
blocks. Input expansion alone does not fix held-out generation in this pilot.
It also costs more attention work. [Full-answer preparation](full_answer_flow_protocol.md)
retained 2.58M assistant labels across 17 sources and tested shared causal query
computation; its later completed derived-format run is linked above. Repeatedly training the old
16,384 target positions cannot establish a general-language capacity ceiling.

[Preconditioned-flow results](preconditioned_flow_results.md) and
[research direction](frozen_codec_research_direction.md): all dimensions and the
original frozen codec retained; three more 2,000-update pilots. Changing output
geometry did not establish useful generation or a win over the matched one-pass
control. Next isolate decoder-feature reading from output-generation difficulty.
Do not restart a full-v4 run based on these results. Production remains paused
at 1,001, original weights/data unchanged, broader goal open.

# Continuous flow matching: bounded result (2026-10-13)

[Matched flow experiment and independent verification](probabilistic_vector_results.md):
three fresh23,209,216-parameter pilots, original E/D frozen,2,000 updates each.
First8-token held-out agreement is6.35% one-pass,2.15% flow,2.05% flow+CE;
exact blocks4/256,0/256,0/256. Eight-call flow sampling is about7x slower than
one-pass. These small everyday/chat pilots do not beat the saved full-v4 token
model's40.28% on the same eight-token targets; that model has much more training,
so it is a quality reference, not a matched-budget comparison. No full run launched.
Original weights/data unchanged. Position-memory production run is paused at1001
(current status), superseding the older402 status below. Broad goal remains open.

# Position-memory joint: earlier loss and speed check (2026-10-13)

The user's full-v4 run is paused at402. On32 fixed development dialogues,
predicted-code CE15.7409/token accuracy1.63% and MSE1.18977 are poor (constant
training-mean vector MSE1.09469). True encoded targets through the same original
frozen decoder yield CE0.17097/99.61% token accuracy. These are small diagnostic
measurements, not full validation or evidence of an unavoidable model-size limit.
The broad-language transfer is not established by the successful synthetic pilot.

The trainer now packs real constructor queries together and skips masked unused
output-decoder positions. Same FP32,23,209,216parameters, loss and target coverage;
all64 construction positions remain. Matched zero-update batches improved about
2x;8consecutive probes with micro8/query16 took2.02–3.12seconds, with peak allocated
memory5.54GiB. Optimizer steps/evaluation/saves are excluded from those timings.
13CPUtests and real32-dialogue all-gradient parity passed; maximum gradient
difference2.98e-8. Original checkpoint402 and codec weights remain unchanged.
Exact source migration permits resume with intact optimizer/schedule/cursor.
No research optimizer updates were run for this diagnosis. Details, safe resume
command and limitations: [model README](../src/models/position_memory_joint/README.md).
Evidence: `dump/position-memory-speed/`.

# Full-v4 position-memory joint trainer PREPARED (2026-10-13)

[Implementation, verified limits and commands](../src/models/position_memory_joint/README.md).
Exactly23,209,216 trainable predictor parameters, original E/D frozen. V4 lacks
the pilot's correct-record and answer-order labels: this full-language adaptation
uses differentiable soft record selection and a continuous reader state, with
MSE +0.02 decoded CE. Full64 position features are preserved across the memory
mixture. It emits independent8-token codes; it does not implement8→64replacement
or inherit the toy's two-way stochastic plan as a general language prior.

Unchanged v4 tokens/masks/splits,483,862training dialogues,180,678,552assistant
targets/epoch. Twoepochs batch32=>30,242updates; whole-dialogue batches differ
from the token-memory segment run. Training uses true past codes; generation
feeds predicted codes directly back. Output quality under this shift is unproven.
No long training launched. Full data audit, causal/gradient/freeze checks and
synthetic resume tests passed. Read-only CUDA probes support micro16/query8 on
12GB, with measured costs and codec reconstruction limitations in the README.
Old reference/pilot weights and v4files remain unchanged. DB is still future work.

# Current investigation: frozen codec reader versus vector generation (2026-10-13)

[Full-position decoder features and joint recovery](position_vector_constructor_results.md):
matched2k construction pilots reach95.90% exact on new value pairs with64positions
versus35.55% with8averaged groups (correct source/order supplied). A1k joint reader
continuation reaches76.12% autonomous exact vswrong1.17%/blocked0%;94.61% of
previously correct draws follow changed facts. Same23.209M/frozen E/D/direct-vector
output. This remains a fixed8-token,32-word task. Held-out layouts, broad language,
growing8->64feedback/EOS and DB remain open; no full-v4 training restart.

[Varied-record constructor follow-up](slot_vector_constructor_results.md): both
matched 3k-update arms fail reliable answer construction, including training-bank
records. Fresh exact answers 0.49% raw / 0.59% eight-slot decoder features; correct
source selection 91.02% / 92.97%. Codec source reconstruction is exact on the
24,576 checked tokens. Correct-source/plan and text-boundary controls do not
rescue the models. Frozen E/D, same 23.209M trainable budget, direct vector output;
no production or full-corpus change. Broader generation goal remains open.

[Frozen-feature follow-up](feature_vector_planner_results.md): matched3k-update
continuations reach83.11% exact with decoder read-features vs67.72% raw codes;
new question templates47.71vs18.75%. Same23.209M/frozen E/D/direct-vector output.
Both fail new record layouts even with correct-source assistance, while the codec
recovers all32768checked source tokens. Constructor format-generalization remains
open. No production model or full-corpus training change.

[Latest shared probabilistic pilot](shared_vector_planner_results.md): one23.209M
predictor selects memory and samples a learned answer-order plan, then directly
constructs an answer vector. Fresh exact correct/wrong/blocked44.73/.39/0%;
new wording14.36%. Frozen E/D, no token inputs. Correct-source/plan diagnostic100%
shows construction works on this narrow task; robust question reading remains open.

[Curriculum follow-up](vector_curriculum_results.md): expanded composition training
reaches100% on unseen pairs when given the correct record. A shared-weight two-pass
selector/constructor reaches91.6% with eight records, but new wording only4.3%.
Frozen codec, direct new-vector output, unchanged23.209M trainable budget; no
general chatbot or broad-language completion claim.

[New direct-output and probabilistic tests](probabilistic_vector_results.md):
seven-pass denoising did not deliver useful language or a speed benefit. New
word-combination generation remains weak even when the correct source record is
supplied. Both codec components stay frozen; no full-corpus run resumed.

[Latest bounded tests and full evidence](frozen_codec_memory_investigation.md).
Growing-vector training is paused at4595; CE/MSE-only continuations did not rescue
held-out text. Direct vector selection plus frozen decoding learned controlled
retrieval. A frozen-decoder feature reader improves full64-token unseen-wording
retrieval from16.6%to47.7% in a matched pilot, still insufficient for robust chat.
No production model, original weights or data changed. The broader generation
objective remains open; historical preparation notes below are superseded.

# Current experiment: growing vector (2026-10-13)

[Implementation and commands](../src/models/growing_vector/README.md). One active
vector replaces itself at lengths 8,16,...,64, then enters completed memory.
Original encoder and decoder frozen; 23,209,216 trainable predictor parameters.
CE scores each new token once; normalized MSE supervises the entire growing code.
Training uses true prior prefix codes; generation feeds back predicted codes.
This difference and revision of already emitted text remain experimental risks.
Seven tests pass; no long training run launched. Whole-dialogue v4 training gives
30,242 updates over two epochs, so steps are not matched to old segment training.
Retired block/vector source and checkpoints removed; reports and source archives
remain. The historical entries below describe earlier states, not current runs.

# Step 3: causal token prediction with compressed history

2026-10-13 follow-up: [frozen-decoder and growing-vector tests](frozen_growing_vector_diagnostic.md)
completed three bounded300-update real-text pilots. Genuine8-token codes decode
99.80%correct on64examples; fixed8/fixed64prediction remains weak. The user's
single-replaced8->16->...->64vector was tested explicitly: new-token teacher-forced
accuracy1.95%,direct rollout0.59%,old-position revision rate80.44%. No universal
failure claim; no production switch or long run. Existing run stayspaused7096.

2026-10-13: 64-token block-vector experiment (now retired; source archived in dump/retired-block-vector-20261013.zip)
is implemented and tested: 23,192,065 trainable parameters, frozen encoder,
vector-only predictor and trainable parallel decoder, normalized target-vector
MSE plus assistant-token CE. Neither learned forward interface takes token IDs.
The decoder has 64 ordered slots and EOS stopping; actual emitted text is encoded
once per block for feedback. Assistant-aligned spans preserve every v4 target
without concealing the question. No DB or full-corpus training launched.

Matched 300-update synthetic pilots: both CE and CE+MSE get 16/16 fresh colors
with correct memory, 0/16 wrong, 4/16 blocked, and follow changed old facts16/16.
Only8/16 full blocks are exact; unseen names can be wrong. Joint loss greatly
improves vector MSE but has not established a language-quality advantage.
The first pilot had a filler-alignment shortcut; corrected evidence is retained
under `dump/block-vector-pilot-300-v2`. Broad chat remains unproven.

Seven implementation/lifecycle tests pass. Real-data zero-update throughput is
3.09 effective batches/s at micro16,4.81GiB reserved, excluding optimizer steps,
validation and saving. Encoding uses a bounded10k CPU-vector cache, never a full
prefix corpus. The slow per-token `vector_memory` source/tests/helpers were
archived and removed at the user's request; its weights and all reference
weights remain safe. See the new README for limitations and fresh/resume commands.

Follow-up [data/loss and raw-versus-vector diagnosis](v4_grounding_diagnosis.md):
simplecolorfacts8/8recent(shortandlongcontrols),0/8compressed,1/8blocked.
Three-recordvalues1/32rawversus0/32compressed. All2202sampledfacttokensrecoverable
byfrozencodec;32EOSerrorsoutsidefacts. Memorytargets0.70%ofcorpus;locationtokens
only~0.12%estimated. No loss-shift/normalization bug found; focused task/CE-weight
repair proposed, not trained. Model size unchanged.

Latest checkpoint evaluation: [v4 final memory and chat tests](unified_v4_final_evaluation.md).
At48,762updates, correct memory improves held-out natural-history CE1.9916 versus
blocked2.1377. However generated known-fact answers0/103 and changed values0/32;
chat samples remain weak. Useful contextual memory is demonstrated, reliable
retrieval/chat for this fresh checkpoint is not. No new training launched.

Latest data-only extension: [v4 tools and reasoning](chat_dataset_v4.md).
483,862training dialogues; fixed model/encoder unchanged, no research training.
Both reasoning modes retained. Tool annotations do not establish learned DB CRUD.

Latest follow-up: [mixed language/retrieval/chat results](mixed_memory_chat_results.md).
Retrieval retained, ordinary language slightly improved on900additional documents;
general chat generation is still inadequate. Original weights remain unchanged.

Latest dataset work: [unified chat corpus v3](unified_chat_dataset.md), with pinned
public teacher-generated conversation/persona/tool/reasoning sources, explicit
thinking on/off, and educational continuation. One canonical token-memory entry
point; no separate chat architecture. The v2 run is paused at2,168updates and safe.
The previous [chat-first corpus and two-epoch contract](chat_dataset.md) remains
historical evidence.
The parameter cap stays fixed. [The chat trainer](../src/models/token_memory/CHAT_TRAINING.md)
is now ready with exact epoch coverage and same-folder resume; DB heads remain
future work. No v3 research training has been launched.

## Current decision: fixed-size conversational memory and Step 3B (2026-10-12)

Planning only: no new training, database implementation, or checkpoint changes.
This follows the CPU/working-state/persistent-storage idea in
[the canonical goal](research_goal.md). Keep the consumer budget at or below
23,209,216 learned parameters, including any controller/query/key heads. The
9,426,432-parameter frozen encoder remains an additional disclosed system cost.
Reallocate a small part of the FFN budget for heads; count the implementation
before committing to exact dimensions. Preserve existing reference checkpoints.

**What is established:** the model reads supplied old vectors on controlled
retrieval tasks. The mixed run gets 98.83% on new question wording and 86.13% on
new bindings plus wording; blocking memory gives 0%. This does not establish
autonomous storage/retrieval, arbitrary fact recall, or raw-context quality parity.

**Why chat is still weak:** the code prepares only the first user/assistant pair,
not multi-turn dialogues. It trains on 32,000 examples from one shard, with
48,000 example presentations in the mixed run. Conversation gets 0.15 of the
separately normalized objective; ordinary LM gets 0.75 and retrieval 0.10.
Retrieval covers eight fixed record names and sixteen single-token answer words.
These are clear coverage limitations, not proof of a unique failure cause.
Teacher-forced assistant CE improved to 2.12515, but free generation still fails
simple instructions with both greedy and sampled decoding. Parameter capacity
may contribute; no controlled scaling/data study has established a 23M ceiling.
External storage can supply facts, but cannot substitute for learning language,
instruction following, and how to use those facts.

### Proposed first database interface: learned READ / WRITE / NONE

Use architecture outputs, not textual tool calls. From the current causal hidden
state, small heads produce operation scores and a query key. A learned write
gate selects which completed encoder chunks to retain. Software executes the
selected operation; the model chooses it. Initially use a local, inspectable
store with exact key search, before introducing an approximate index.

Store rows containing record ID, learned search key, frozen 256-dimensional
encoder value, source order/group, valid length, and encoder/key-model version.
Keep the original vector precision initially. Reconstruction vectors are values;
their Euclidean distances are not established semantic search distances. Train
a separate key/query matching space. Rebuild keys if the key projection changes;
do not silently search an index produced by incompatible weights.

Retrieved values return directly to the existing memory attention input, with
a bounded combined recent/retrieved budget (initially 32 vectors). Define packing,
ordering, duplicate exclusion and provenance explicitly; arbitrary retrieved
order is a new distribution, not equivalent to the existing chronological input.
The database can grow independently of that working-memory limit. Report actual
storage bytes and retrieval time; 64 tokens per vector is not 64x byte compression.

Reading only after each 512-token rollover is too late for a short question.
Offer a read decision after consuming a user turn and before generating its
answer, plus at rollover. Allow NONE or one bounded read per event initially.
After retrieval, recompute the active prefix with the selected vectors before
answering; count this extra forward pass. Here a block means a temporal 512-token
buffer, not a Transformer layer. Writes initially operate on completed chunks;
short turn-end records need an explicit padding/valid-length path and tests so
facts need not wait for 512 tokens. Never encode future answers into memory.

Token CE alone does not backpropagate through hard database selection. Start
with labelled operations/write decisions and contrastive query/key training
with positive records and hard negatives, alongside answer CE. Train the
consumer with oracle reads first, then test autonomous predicted operations.
Training labels are scaffolding; evaluation must not secretly supply correct
records or actions. Learn/calibrate a no-match result on absent-fact examples.
UPDATE/DELETE, conflicting versions, cache invalidation and deduplication follow
after read/write works; they are not implemented by appending duplicate records.

### Dataset and evaluation before the next run

1. Audit and prepare short, coherent, multi-turn conversations with assistant-only
   loss, consistent roles/EOS, greetings, direct answers, follow-up references and
   explicit unknown answers. Preserve whole conversations across disjoint splits.
   Include simple prose for language fluency; broad instruction examples alone
   are not evidence of suitable difficulty for this model.
2. Build grounded dialogue episodes with facts outside the active window and in
   earlier sessions. Vary names, values, record phrasing and questions; include
   distractors, missing facts and occasions requiring no database access. Use
   variable-length answers, not just the existing sixteen-word answer bank.
   Keep entities/bindings and template families held out; remove duplicates.
3. Evaluate matched questions with (a) facts in recent raw text, (b) oracle old
   vectors, and (c) learned DB retrieval. Failure in (a) points to language/task
   learning; an (a)-(b) gap implicates vector reading; a (b)-(c) gap implicates
   storage/routing. Check action accuracy, write admission, recall@k, absent-fact
   handling, full generated answers and language CE separately.
4. In autonomous rollouts clear active context, reopen persistent storage, and
   ask about stored facts. Replace a DB fact without changing model weights and
   require the answer to follow it; block/shuffle retrieval as controls. Report
   answer accuracy by distance, distractor count and unseen wording, plus latency.
5. Change data first with the existing model as a control, then add the budgeted
   DB controller. Match dataset, target tokens and update budget for new paired
   comparisons; also report compute. Extra continuation cannot be compared as an
   equal-budget win against the old 20k baseline. Existing FFN differences remain
   a confound in raw-versus-memory architecture attribution.

Relevant precedents, not guarantees for this implementation:
[TinyStories](https://arxiv.org/abs/2305.07759) shows that suitably restricted data
can support coherent text in small models;
[Differentiable Neural Computer](https://www.nature.com/articles/nature20101)
studies learned external-memory control;
[Memorizing Transformers](https://arxiv.org/abs/2203.08913) studies key/value
retrieval for language modeling. Our next claim must be measured locally.

The sections below preserve experiment history; this decision and the linked
mixed-run report supersede earlier statements about pending experiments.

## Memory-dependent training experiment (2026-10-12)

User authorized training and evaluation. Continued saved token-memory20k for
2000 specialized updates, fresh AdamW, LR3e-5 decaying to3e-6,warmup50,
batch64/micro16. Architecture unchanged; encoder frozen.128000 unique training
histories, none matching development/test histories. Each history has8 named
records in8 independent64-token chunks (512tokens), random access words from
16 single-token words, and random record order. The recent question contains
the requested record name, never the answer. Loss is full-vocabulary CE at the
single answer position. This is a retrieval diagnostic, not ordinary LM training.

Train reserves4/16 answer words per record name for a held-out binding test.
Tests contain512 examples each, arranged in256 pairs with identical questions
and histories differing only in the queried fact. Wrong memory changes only
that fact to the paired alternative. Test metrics use the reloaded final2000
checkpoint. No hyperparameters were changed based on the test results.

| Test | Correct memory accuracy | Wrong memory, original answer accuracy | Shuffled memory | Blocked memory |
|---|---:|---:|---:|---:|
| New histories, familiar bindings |100%|0%|100%|8.40%|
| Held-out record/word bindings |90.63%|0%|89.84%|0%|
| Unseen question wording |20.51%|3.71%|21.29%|7.42%|

With changed facts, answers followed the replacement word on100% of familiar
binding cases and90.63% of held-out binding cases. Both members of counterfactual
pairs were correct100%/83.59% respectively. This establishes learned reading of
old encoder vectors for this controlled task, not merely query-only prediction
or fixed-slot lookup. Original model answer accuracy was0% in these tests; that
also reflects unfamiliar answer format, so causal memory controls are essential.

Limitations:8 fixed record names,16 words,one history wording,512-token histories;
not arbitrary fact/string retrieval. Unseen question wording generalizes poorly.
Its answer CE6.701 is worse than blocked3.174 despite higher top1 accuracy,
showing overconfident errors out of template. Ordinary development CE worsens
3.01312 to3.24600,accuracy41.30%to38.53%. Retrieval-only specialization is not an
overall language-model improvement. Original checkpoint retained as general
model; experimental checkpoint not promoted. Next candidate is mixed ordinary
LM/retrieval training with varied wording, evaluated on untouched templates and
held-out facts. No additional mixed training launched.

Output:dump/memory-retrieval-2000-20261012. recipe.json,dataset.json,data_audit.json,
before.json,after.json,training.jsonl,development.json,result.json,last.pt and
completion_audit.json retain evidence. Checkpoint SHA
adb95a7745d8f0c40d32ebc95fc3de995953a139b0d8e39321f39e4b40b02a3d.
Verified2000 sequential updates, saved prediction metrics, checkpoint hash,
encoder/parent unchanged and no encoder gradients. Source:tools/train_memory_retrieval.py
and immutable run source.py. About485seconds including validation and final tests.

## Attention diagnostic (2026-10-12, no training)

Same saved20k checkpoint and100dev documents;400history-bearing segments.
Attention mass to memory averaged over all8layers/8heads:
first query81.95%,queries2-16 20.38%,17-64 10.00%,65-128 5.34%,129-256 3.84%,
257-512 2.78%. Uniform-over-available-slots references are93.99%,67.37%,33.32%,
14.31%,7.83%,4.11%; these are descriptive references, not optimal targets.
Memory key/value norms are lower on average than token key/value norms at each
layer; does not identify a causal defect or prove incompatible latent spaces.

Inference-only all-layer memory logit bias log(2)/log(4) retains ordinary causal
mask and positions. History-target CE baseline3.0306624,2x3.0321724,4x3.0385272.
First64 CE baseline3.3718677,2x3.3768413,4x3.3950162. Paired document bootstrap
intervals exclude zero for these degradations. Forcing greater memory attention
does not repair this frozen checkpoint; not a test of training with memory bias.
No parameters changed. Attention reimplementation at zero bias matched exactly.
Source and per-document evidence:dump/memory-attention-20261012.

Encoder accepts1/8/32/64-token inputs and returns one finite256-dimensional
vector; this verifies interface only, not reconstruction or LM quality. Current
LM was trained on complete64-token history vectors and ordinary recent token
embeddings. Replacing recent tokens with encoder vectors requires a new trained
consumer/input adapter. Independent one-token codes remain one-token-per-slot
and can be cached by vocabulary ID for this frozen encoder; recomputing the
growing recent prefix every step is a different, more costly proposal. Prefix
codes in training must not include future targets (encoder is bidirectional).

## Saved-model memory ablation (2026-10-12)

Read-only evaluation of token-memory-512-micro16 last.pt at20,000updates.
Checkpoint SHA14b8457ae5a1cbf3dbdce11477d544f143d09602d63569ed8e5903da415955b0.
Same100 development documents as saved validation,211200 targets;160000 targets
have older memory. No training, optimizer, or weight changes. Correct-memory CE
replayed at3.01310852 (7.15e-6 from saved aggregate, changed batching/reduction).

| Condition | CE on targets with history | CE first64 after rollover | CE later targets |
|---|---:|---:|---:|
| Correct |3.0306624|3.3718677|2.9656709|
| Wrong document, same history length |3.0403938|3.4195648|2.9681707|
| Shuffled vector order |3.0312001|3.3753174|2.9656539|
| Blocked memory keys, same token positions |3.0386164|3.4129506|2.9673146|

Paired document bootstrap,10000 replicates: history-target blocked-minus-correct
CE0.007954,95%interval[0.006829,0.009150]; wrong-minus-correct0.009731,
interval[0.007853,0.011727]. First64 blocked-minus-correct0.041083,
interval[0.035341,0.047317]. Blocking worsens document-aggregated history loss in
95/100documents; first64 loss in97/100. Ordering effect is small, and later-token
shuffle interval includes zero. Model uses older vector contents usefully, but
the measured benefit is modest and concentrated near block beginnings.
Does not establish exact factual retrieval, superiority to raw context, or free
generation quality. Reused development set, one checkpoint and one corruption
draw; uncertainty is conditional on this model and protocol.

Blocked keys are masked inside every attention layer, retaining original RoPE
positions. Verified blocked output invariant to replacement memory, no-history
outputs identical across conditions, all model/encoder tensor hashes unchanged,
and no gradients. Results/per-document statistics and exact evaluation script:
dump/memory-ablation-20261012. Reusable script:tools/evaluate_memory_ablation.py.

## Current decision (2026-10-10)

The user replaced vector-output experiments with direct token prediction. Prior
MSE/parallel-decoder results remain historical; exact prior code/docs are archived
in `dump/superseded-vector-output-source.zip`. Runs,weights and negative evidence
are retained where not explicitly retired. Token-memory completed 20,000 updates
with validation CE 3.013115671 and token accuracy 41.297348%. The hierarchical
decoder alternative completed the same update budget with CE 4.200900109 and
24.643939% accuracy and was retired on 2026-10-12. Its code and weights were
removed, preserving result logs and source archives. Raw baseline and direct
token-memory remain active. The ablation above establishes a small contextual
prediction benefit for this checkpoint; it does not establish raw-baseline parity.

## Architecture

The fresh Branch Sigmoid transformer has23,209,216trainable parameters,8layers,
width512,8heads. Input combines frozen256-dimensional encoder vectors and token
embeddings in the same causal self-attention sequence. Output is a vocabulary
distribution for the next token, trained with CE. No separate decoder or latent MSE.

The active block holds up to512tokens. On reaching512, independently encode its
eight64-token chunks, append8vectors to memory, and clear the active buffer. The
block-start marker is never encoded. Training uses actual preceding blocks/tokens;
generation uses generated tokens. No target/future tokens enter encoder memory.
All older vectors are retained, so attention cost grows with history. Current data
trains at most32older vectors; longer-history quality is untested. No KV cache yet.

## Data

Reuse the downloaded320k/1000document windows of2112tokens. Repack stored IDs into
compact uint16 memory-mapped files without downloading, decoding or retokenizing.
Each document gives5segments of512/512/512/512/64targets, with0/8/16/24/32older
vectors. Every document token is a target exactly once per full dataset pass.
Documents are never joined. Encode only the history required by each microbatch.

## Short prompts and chatbot scope

The old partial64-token prompt restriction no longer applies: short prompts and
partial final512-token blocks stay as raw tokens. Boundary tests cover0,1,511,512,
513-token prompts. This is input handling, not chatbot quality. The mixed follow-up
adds single-turn instruction data, EOS and a transcript-based chat tester with
greedy/sampled decoding. Multi-turn training and tool execution remain untested.

## Verification and limitations

Tests cover future-token masking, padding invariance, rollover target alignment,
generated-token compression, predictor gradients, frozen encoder, checkpoint reload
and tiny synthetic resumed/uninterrupted training equality. No research training
or full dataset conversion was run during setup. Tokenizer/manifest/binary-size
checks remain; repeated expensive full-corpus checks are omitted. GPU memory and
quality are not promised. Public pretrained baselines remain inference-only.

## Step 3B: persistent vector storage (planned)

Future memory may provide create/read/update/delete operations on encoder vectors.
The transformer supplies reasoning/text; the external store supplies retrievable
information. This does not imply neural weights stop storing facts. Retrieval,
memory budgeting, storage versioning and evaluation are specified in the current
plan above; no database work is implemented. Commands: `docs/training_cli.md`.
# Historical: three-way comparison added (2026-10-10)

The token-memory model completed20,000 updates with validation CE3.01311567 and
41.29735% token accuracy. New fresh comparison arms are raw_transformer (23.077M,
GELU,context2560 raw positions) and hierarchical_memory (23.244M,global Branch
Sigmoid plus small causal decoder,conditioning vector and past-state cross-attention).
Both use the same prepared data,seed47,batch32,target-token CE and20k schedule.
The hierarchy compresses completed64-token blocks and computes its global state
once per generated block; the active token-memory arm still rolls over512 tokens.
See [three_model_comparison.md](three_model_comparison.md) for exact architecture,
commands, matching limitations and tests. No new research training launched.
At setup, memory removed/shuffled evaluation was pending. It has since completed
as recorded above; the hierarchical arm was retired. Step3B vector CRUD remains
planned, not implemented.

