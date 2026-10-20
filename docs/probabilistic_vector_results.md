# Follow-up: reversible preconditioning, 2026-10-13

[Full geometry experiment and verification](preconditioned_flow_results.md):
three additional matched 2,000-update pilots retain all 256 vector dimensions,
23,209,216 trainable parameters and the frozen original codec. On another 256
held-out prompts, transformed one-pass/flow/flow+CE score 5.37%/2.05%/2.25%,
versus 6.54%/1.32%/2.05% for their original-coordinate controls. No significant
paired improvement or useful language generation was established. Flow samples
became more similar across independent draws. Full-v4 token-memory quality
reference: 40.58%, with much more training. Production remains paused at 1,001.

# Continuous flow matching: matched real-language pilot, 2026-10-13

**Completed and independently reloaded: neither flow arm beats the matched
one-pass control. No production model was trained or replaced.** The user's
position-memory full-v4 checkpoint is currently paused at **1,001**, as confirmed
by its actual status and unchanged checkpoint hash; older notes saying402 are
historical. The broader generation/baseline-comparison goal remains open.

## Method and controls

This is a new continuous velocity-field experiment, not a repeat of the earlier
seven-level clean-code denoiser below. Following the straight conditional path in
[Flow Matching](https://arxiv.org/abs/2210.02747),
`x(t) = (1-t)*Gaussian + t*standardized_target`, with velocity label
`standardized_target - Gaussian`. Noise and target are sampled independently;
no minibatch optimal-transport coupling is claimed. The predictor's eight
existing phase vectors supply smooth time conditioning; active-state amplitude
is retained rather than RMS-normalized away. This input interface differs from
the older denoising pilot, so the new one-pass arm is the matched reference.

All three fresh arms have exactly23,209,216 trainable parameters, identical
initial weights, batches/data order, AdamW3e-4/weight decay0.01, batch64/micro32,
and2,000 updates. The original encoder AND decoder are frozen. No token IDs
enter the predictor; it outputs a new eight-token codec vector.

- One-pass control: normalized clean-vector MSE, plus0.02 decoded CE after200.
- Flow: velocity MSE only.
- Flow+CE: velocity MSE, plus0.02 CE on the estimated clean endpoint after200.

For flow+CE the training endpoint is `x(t)+(1-t)*predicted_velocity`; its
teacher-assisted CE is not a free-generation result. Inference starts from
Gaussian noise with only past codes supplied. Euler8/16/32 and Heun16 were
compared on development data only. Both flow arms selected Euler8 under the
prespecified accuracy/exactness/fewer-calls rule. Extra solver calls did not
establish a development-quality rescue.

## Actual data coverage

Reuse of the previous bounded prepared data was checked by re-encoding four
spread-out train records and four development records. Training has2,048
dialogues:458 everyday and1,590 chat. Development has256:5 everyday and251 chat.
The final256 held-out dialogues are all chat, selected after the256 test cases
used by the previous denoising experiment. No test/development literal input
prefix or full input/answer pair exactly matches training. Document IDs are
split-disjoint. This does not establish absence of near duplicates.

Only the first eight tokens of the first eligible full assistant span are
predicted, with up to32 prior codes. There are16,384 distinct training target
positions and128,000 example presentations per arm. The test's prior text ranges
from31 to1,566 tokens. This is **not the full17-source v4 mixture**, a full-answer
chat evaluation, a long-memory benchmark, or a test of growing8-to64 feedback.
Training-only mean/std are shared across arms. Genuine target codes reconstruct
2,048/2,048 dev tokens and2,046/2,048 test tokens;254/256 test blocks are exact.

## Results

Single first draw, same256 held-out prompts and2,048 target positions:

| Model | Token agreement with reference | Exact8 blocks | Sample+decode, batch32 |
|---|---:|---:|---:|
| Matched one-pass MSE+CE |6.35%|4/256|19.5ms|
| Continuous flow |2.15%|0/256|139.3ms|
| Continuous flow+0.02CE |2.05%|0/256|140.7ms|

Timings include the frozen decoder but exclude history encoding/preparation;
they are not end-to-end chatbot timings. Eight-call flow sampling is about7.1–7.2x
slower than the one-pass control. Training plus periodic dev/checkpoint saves
took240.4/220.6/245.7seconds respectively; final solver sweeps, test evaluation
and independent verification are additional.

Four-draw mean test token agreement is1.66% for flow and2.23% for flow+CE;
neither gives any exact block over1,024 draws. Both average four distinct
outputs per prompt, which demonstrates diversity rather than useful language.
Paired dialogue-bootstrap95% intervals for first-draw accuracy minus one-pass
are[-6.30,-2.29] and[-6.40,-2.44] percentage points. These describe uncertainty
across these cases, not across training seeds; each arm has one training seed.

On256 checked TRAIN examples, free generation gives99.95% token agreement for
one-pass,10.11% for flow and15.72% for flow+CE. Thus one-pass memorizes the small
training set but generalizes poorly; these flow recipes additionally fail to
sample training answers accurately. Lower velocity/teacher-endpoint loss is
insufficient evidence of successful generation.

The first retained sample has reference ` No, Tom is not shor`; outputs are
`." that is comp thatonsit"` (one-pass), ` out make was the light, orer` (flow),
and ` But be was co that fist is` (flow+CE). All24 ordered examples per arm are
saved, not just this example. Eight BPE tokens can end midword even for the
reference, so truncation alone is not a failure criterion. Low reference agreement
also does not reject every valid alternative continuation; the saved outputs
and weak training-sample results support the negative practical assessment.

Correct/wrong/blocked conditioning token agreement:

| Arm |Correct|Wrong|Blocked|
|---|---:|---:|---:|
|One-pass|6.35%|0.63%|0.54%|
|Flow|2.15%|0.34%|0.20%|
|Flow+CE|2.05%|0.63%|0.29%|

These controls replace/block **the question as well as prior history**, because
all conditioning is encoded. They do not isolate old-fact retrieval. The sampler
is exactly invariant to changes in masked-out code values.

## Existing token model as a quality reference

The unchanged completed v4 token-memory checkpoint at48,762 steps achieves
**40.28% token agreement and57/256 exact prefixes (22.27%)** on these same first
eight targets, generating autoregressively. It receives the token sequences that
were encoded for the pilot, including any historical EOS fill in that pilot's
plan; recent text is raw and older text is encoded by its normal512-token rollover.

This is **not matched training**: that model trained on the full v4 corpus for
two epochs, whereas the new pilots trained on2,048 dialogues. Representation,
conditioning and generation factorization differ. This is a practical quality
reference, not proof of architectural superiority or relative sample efficiency.
Its40.28% free-generation agreement must not be confused with its61.01%
teacher-forced full-development token accuracy. No pilot beats this reference.

## Verification and decision

Two CPU contract tests cover path algebra, known-solution solver convergence,
padding/blocked invariance, seeded sampling, noise dependence, frozen gradients
and shortened-decoder parity. Independent CUDA reload reproduced every saved
generated token and first code bitwise for all three conditioning modes, checked
the original frozen decoder tensors, exact parameter counts, Adam step counts,
final/last checkpoint tensor identity, and real decoder-logit/gradient parity.
Four final test examples were independently re-encoded. All protected checkpoints,
codec files, and all top-level v4 data files retain their original hashes.

A first zero-update verification process was explicitly interrupted after
unusually slow startup without results. A second process, staging prepared
tensors on CPU before CUDA transfer, completed the checks. No training was
repeated. The precise cause of the first startup delay was not established.

Evidence: `dump/flow-vector-real-20261013/{results,verification,data_audit,
coverage_audit}.json`, archived training/support sources, immutable prepared
pilot tensors, per-case generated outputs, and three diagnostic checkpoints.
Scripts: `tools/{flow_vector_common,train_flow_vector_pilot,
verify_flow_vector_pilot,audit_flow_vector_data}.py`.
The pilot trainer supports periodic optimizer/RNG resume with input/source
guards; a complete interruption/replay lifecycle test was not run.

Do not promote these checkpoints or start full-v4 flow training from this result.
The evidence rejects these bounded recipes, not flow matching in general.
A next experiment needs a specific improvement to language conditioning or
latent predictability and must outperform this matched control on held-out
generation before scaling. Broad chat, variable length/EOS, repeated direct-code
feedback, old-fact retrieval and a fair full-training baseline win remain open.

---

# Direct codec-vector generation: earlier bounded tests, 2026-10-13

Follow-up: [a broader composition curriculum and two-pass memory prototype](vector_curriculum_results.md)
now generalize to unseen word combinations on a controlled task. This does not
change the failed real-language denoising results below.

The user selected direct vector output with probabilistic generation. Both codec
components remain frozen, and neither predictor nor decoder receives raw token
IDs. These experiments did not produce a usable generative model. The original
growing-vector run remains paused at update 4,595; no full-corpus run was resumed.
All pilots retain exactly **23,209,216 trainable predictor parameters**.

## Controlled generation, distinct from copying a stored record

Each history contains eight 64-token records. A question identifies a name;
the predictor must construct an eight-token answer such as
`The access words are red and cat.` The answer string is absent from the records.
Four colors and eight nouns give 32 possible answers: 24 combinations are used
in training and eight combinations are held out. Each individual word occurs in
training. Name and question-template holdouts provide additional tests.

Two fresh arms use raw codec codes or mean-pooled frozen-decoder hidden states
as their inputs. They share initialization, examples, optimizer and 1,200 updates.
Both use 200 MSE-only updates, followed by normalized MSE + 0.02 token CE through
the frozen decoder. Target codes are loss labels only. All 32 genuine answer
codes decode exactly. Six answer tokens are constant, so whole-answer accuracy
is more informative than token accuracy here.

| Exact answers, 512 cases per family | Direct codes, eight records | Pooled features, eight records | Direct codes, correct record supplied |
|---|---:|---:|---:|
| Familiar answers in new histories | 58.40% | 0% | 100% |
| Previously unseen word combinations | 1.17% | 0% | 1.56% |
| Unseen names | 42.77% | 0% | 100% |
| Unseen question wording | 9.96% | 0% | 100% |

The last column is an **oracle diagnostic**, not an autonomous retrieval model:
the test harness supplies the correct source record. Its remaining composition
failure separates constructing a new answer code from finding the correct record.
Supplying wrong records drops familiar-answer accuracy to 3.52%; blocking memory
gives 1.56%. Removing distractors also makes names/question wording largely
unnecessary, so the 100% figures do not establish general question understanding.

The pooled reader's earlier retrieval improvement did not transfer to this
generation task. Its correct-record oracle also learned familiar answers, but
scored 0% on unseen combinations. This does not establish a universal limitation
of decoder features or direct vector prediction.

Evidence: `dump/frozen-memory-generation-20261013` and
`dump/frozen-memory-generation-oracle-20261013`. Full per-case outputs, protocols,
source copies, support archives and diagnostic weights are retained. Changed-value
tests can cross the training/held-out combination boundary; do not interpret their
aggregate score as a clean independent composition test.

## Probabilistic generation on real language

`tools/test_probabilistic_vector_generation.py` compares a fresh one-pass predictor
with conditional denoising from Gaussian noise. Each produces **one vector for
eight tokens**, which the original frozen decoder converts to text. Sampling
receives only prior memory vectors and seeded noise, never the target code.

The denoiser replaces the same noisy vector over seven refinement passes. Noise
levels are 1.0, 0.9, 0.7, 0.5, 0.3, 0.15 and 0.05. Training corrupts genuine
target codes and predicts the clean standardized code. Both arms use the same
initial weights, batch order, 2,000 updates, batch 64, AdamW at 3e-4, and the
MSE/CE recipe above. Existing phase embeddings identify noise levels without
adding parameters. Denoising has additional noisy-input supervision; this is an
algorithm comparison, not an isolated loss-weight comparison.

Data consists of 2,048 training dialogues, 256 development dialogues and a fresh
256-dialogue test subset from v4. Selection takes the first eligible full
assistant block per dialogue and predicts only its first eight tokens. Earlier
history is capped at 32 vectors. Codes are prepared once for this bounded subset,
not the full corpus. There are 16,384 distinct training target positions, repeatedly
presented during optimization. Normalization uses training codes only. Selection
is deterministic, not a representative benchmark sample.

| Fresh test, first sampled vector | One-pass prediction | Seven-pass denoising |
|---|---:|---:|
| Token accuracy, correct memory | 7.71% | 8.40% |
| Exact eight-token blocks | 6/256 | 3/256 |
| Token accuracy, wrong memory | 0.83% | 1.51% |
| Token accuracy, blocked memory | 0% | 0.15% |
| Sample + decode, batch 32 | 22.5 ms | 129.9 ms |

The small accuracy difference is not an established quality win. Denoising is
about 5.8 times slower in this local inference probe and produces fewer exact
blocks. Four noise draws produce 3.87 distinct outputs on average, but diversity
does not imply coherent language. Encoder work and data preparation are excluded
from these inference timings. This is not a full-system throughput benchmark.

Example first draws from the fresh denoising test:

| Reference prefix | Generated prefix |
|---|---|
| `You can cook a wide variety of` | `Yar even) very has Fle` |
| `Start by writing down all your t` | `Thensist could could comp. to` |
| `Let's break this problem down:` | `With thatness Nationalin main, rep` |

The deterministic model reproduced all 256 checked training examples exactly;
its held-out performance remained poor. The denoiser did not fit those training
examples during actual sampling: 5.13% token accuracy and zero exact blocks.
Thus this denoising recipe has an optimization/sampling problem in addition to
the deterministic control's poor generalization. It is not evidence that every
probabilistic latent model must fail.

On 64 fresh test examples, denoising inputs containing the true target plus only
0.05 noise yielded 49.8% token accuracy, falling to 5.66% at pure noise. Those are
teacher-assisted diagnostics, not generation results. The true target codes
decoded 2,048/2,048 development tokens and 2,047/2,048 fresh test tokens (99.95%).

Decoder CE conditional on a generated code is recorded for diagnosis. It is not
ordinary autoregressive LM perplexity. The four-draw per-position marginal CE is
also not the joint likelihood of an eight-token text; do not compare it directly
with the original token-memory model's CE 1.8047.

## Verification and decision

CPU sampler checks passed: target independence, repeatable seeded generation,
noise-dependent outputs, blocked-memory invariance, CE gradients into the
predictor, and frozen decoder parameters. GPU reload checks independently
re-encoded four development histories/targets and matched stored data and labels.
Fresh test IDs are disjoint from training/development IDs. Loaded models passed
target-independence and blocked-memory invariance checks. Checkpoint/codec hashes
were verified unchanged, including the original v4 model and paused growing run.

Scripts: `tools/verify_probabilistic_sampler.py` and
`tools/verify_probabilistic_vector_generation.py`.
Evidence: `dump/probabilistic-vector-20261013/results.json` and
`verification.json`, prepared pilot tensors, checkpoint hashes, generated codes,
examples and source archives. These are diagnostic checkpoints, not chatbot weights.

Do not promote this seven-pass generator or launch two full epochs with it.
Future experiments should first pass unseen word-combination generation on the
controlled task, with actual noise-only sampling and correct/wrong/blocked memory.
Then extend to real text and the growing 8-to-64 feedback path. The current tests
do not validate growing-prefix consistency, EOS handling, long rollouts or a chat
interface. Useful memory reading has been demonstrated separately; dependable
new-text generation in the frozen codec space remains unresolved.
