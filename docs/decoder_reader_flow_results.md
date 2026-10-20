# Separate frozen-decoder positions: independently verified bounded result

Both bounded training arms and independent reload verification are complete.
Neither model beats its compact-code control on this held-out test.
The full-v4 production model remains paused at step 1,001. Original weights,
codec and data are protected by the previous experiment's SHA manifest.

## What changes

The compact-code controls already completed 2,000 updates each. Two new arms
use the same initialization, selected examples, noise streams, objectives and
23,209,216 trainable parameters, but replace each input code with its separate
valid frozen-decoder position states. Records are not averaged together. The
original encoder and decoder stay frozen; the predictor receives no token IDs
and outputs an eight-token codec vector. This differs from the production
position-memory model's pooled reader and soft mixture of record positions.

The new arms are one-pass MSE+0.02CE and flow velocity-MSE+0.02 endpoint CE.
CE begins at update 201. Each uses batch64, eight microbatches of8, sorted by
input length inside the effective batch. The old controls used microbatch32.
Targets, sampling order and effective-batch normalization are preserved; FP32
grouping and allocator/kernel choices are not claimed bitwise equivalent.

This deliberately expands attention: the training mean is 193.85 positions
instead of about3.98 codes. It tests input readability, not a compression-speed
improvement. Inference caches causal history K/V once, then updates only the
active query during flow integration. CPU tests and a 23M GPU preflight check
that this factors the same forward computation; it is not an approximate reader.

## Data and limits

The pilot still uses 2,048 train dialogues (458 everyday,1,590 chat),256 dev
dialogues and only the first8 target tokens of one eligible assistant span.
The new held-out256 examples are eligible test indices768:1024, after previous
test cohorts. They contain143 chat and113 stories; stories are absent from the
pilot's training set, so final results must also be separated by source.

IDs are disjoint; no literal test prefix equals a training prefix. Near
duplicates have not been excluded. Target codes never enter the sampling
interface. The genuine-code oracle reconstructs all2,048 new-test target tokens.
This finite result does not establish a universally lossless codec.

Correct/wrong/blocked controls in this pilot alter the whole history, including
the question. They cannot establish old-fact retrieval independently of question
reading. No full answers, EOS, self-fed history, broad chatbot quality, or fair
comparison against the much-longer-trained full-v4 reference is established.

## Completed generation results

Every arm has 2,000 retained optimizer updates. Compact controls were already
trained and were re-evaluated on this new test without additional updates.

| Input / objective | First-draw token agreement | Exact eight-token blocks | Sampling calls |
| --- | ---: | ---: | ---: |
| Compact codes / MSE + CE | 3.5645% | 0/256 | 1 |
| Separate decoder positions / MSE + CE | 1.2207% | 0/256 | 1 |
| Compact codes / flow + CE | 1.5137% | 0/256 | 8 |
| Separate decoder positions / flow + CE | 0.4395% | 0/256 | 32 |

Solver/step counts were chosen on development data, not this test. Four-draw
agreement for the flow arms is 1.7700% compact versus 0.4517% expanded, with no
exact blocks in either arm's 1,024 draws. This is reference agreement on short
continuations, not a complete language-quality benchmark. The generated samples
also contain broken words and incoherent fragments.

The expanded one-pass model generated all 256 checked training answers exactly,
but none of the test answers. Its training-sample decoder CE is 0.000961 versus
39.2557 on test. The flow model has only 2.3438% agreement and zero exact blocks
on the checked training sample. It has not demonstrated even the same training
fit, despite its partially revealed-target denoising loss decreasing.

Expanded correct/wrong/blocked agreement is 1.2207/0.2930/0.1953% for one-pass,
and 0.4395/0.1953/0.4883% on the first flow draw. These low scores do not establish
useful retrieval. The controls also change the question, so they must not be
presented as isolated old-memory tests.

Training plus periodic development checks/checkpoint saves took 1,298.922 seconds
for expanded one-pass and 1,318.453 seconds for expanded flow (about 21.6 and 22.0
minutes). The successful launcher invocations together took about 49.1 minutes,
including additional final evaluations, but excluding preparation, the discarded
attempt, and independent verification. Peak allocated memory was 7.94 GiB and
reserved memory 8.54 GiB across those invocations. These are this pilot's costs,
not a full-v4 runtime prediction.

The evidence does not support expanding every stored code as a useful generation
improvement under this small training protocol. It does not establish that a
task-trained reader, more representative supervision, or all latent generation
must fail. The [full-answer protocol](full_answer_flow_protocol.md) addresses the
supervision limitation; it is preparation only and has not been trained.

## Independent checks and source breakdown

`verification.json` reproduces saved token IDs and first-draw vectors exactly.
It checks original frozen decoder tensors, 23,209,216 trainable parameters,
all Adam counters at 2,000, final/last weight identity, archived source hashes,
protected original weights/data, four freshly re-encoded inputs/targets, target
mutation invariance and blocked-input invariance. Cached versus full-forward
sampling differs by at most 3.58e-6 (one-pass) and 1.91e-6 (flow) in code
coordinates, with identical decoded tokens on the checked eight cases. That
comparison uses the identical noise tensor, avoiding CUDA RNG shape differences.
The one-pass 256-example training fit was independently reloaded as well;
`train-fit-reload.json` confirms 256/256 exact answers with zero optimizer updates.

| Input / objective | Chat agreement (143 cases) | Stories agreement (113 cases) |
| --- | ---: | ---: |
| Compact / one-pass | 6.3811% | 0% |
| Expanded / one-pass | 2.1853% | 0% |
| Compact / flow + CE | 2.4476% | 0.3319% |
| Expanded / flow + CE | 0.6993% | 0.1106% |

The expanded-minus-compact paired agreement differences have 95% bootstrap
intervals of [-3.9063, -1.0254] percentage points for one-pass and
[-1.8066, -0.4395] for flow. These quantify variation across these test dialogues,
not across training seeds. The stories source was absent from training, but the
chat subset also fails to show an improvement.

For the fixed 32-prompt timing group, sampling plus output decoding took
20.8 ms compact one-pass, 143.2 ms expanded one-pass, 135.3 ms compact flow and
2,226.7 ms expanded flow. Expanded inputs run as four microbatches of eight;
compact inputs use one batch of 32. Expanded timing includes building the history
KV cache each repeat. All timings exclude input encoding/frozen-feature creation,
and flow uses the separately selected 8 versus 32 Euler steps. These are disclosed
inference settings, not a pure fixed-step hardware comparison or full-system speed.

The existing full-v4 token model scored 44.3848% eight-token agreement and
73/256 exact blocks on these prompts (chat 38.4615%, stories 51.8805%). It had
48,762 full-v4 updates and raw recent tokens: a useful quality reference, not a
matched-training-budget baseline. No superiority or capacity claim follows from
comparing its score directly with these tiny pilots.

## Runtime correction and discarded work

The first attempt reached at least300 updates but Windows reported approximately
11.19GiB dedicated and2.52GiB shared GPU memory for the worker. Reported training
speed had fallen to0.61updates/s. Sending terminal Ctrl+C terminated the process
before its Python handler saved a checkpoint. There is no retained model from
that attempt; its exact discarded update count is unavailable. This is wasted
compute and must be included as a limitation of the overall experiment cost.
Its source archive and initial protocol remain in `interrupted-sources.zip` and
`interrupted-results.json`; `interrupted-attempt.json` records the event.

A zero-update allocator probe then ran12 consecutive effective batches with
Adam-sized buffers, plus the eight longest histories. It preserved every initial
model tensor. Per-batch times were0.531–0.937seconds, excluding optimizer steps;
peak reserved8.40GiB, longest-case allocated7.81GiB. These are capacity/throughput
probes, not a measured whole-run ETA.

The new attempt uses the unchanged trainer through
`tools/run_decoder_reader_limited.py`, with PyTorch memory fraction0.75 and
garbage collection threshold0.75. It starts from the same original initialization
and seeds. At update400 it reported1.59updates/s; a Windows counter sample showed
8.71GiB dedicated and no process shared-memory allocation above100MB. This is an
observed improvement, not an isolated hardware benchmark or a new objective.

The first bounded invocation used `--stop-after 501`: update500 validation and
checkpoint were followed by a successful programmatic stop after501. Saved
Adam counters were all501, source/protected hashes passed, and the original
decoder tensors were unchanged. CheckpointSHA was
`3395fc5570d4c64b883816213db054ab5fba5a5e3326596a7cafdf89634b5bab`.
It was resumed and both arms reached the predeclared 2,000 updates.
At500, dev agreement was1.171875%,0/256exact,decoderCE32.6811 and normalized
MSE1.3870. This is preliminary poor validation, despite falling training loss.
At update1,000, training-batch CE was0.734 and MSE0.02295, but dev agreement
remained1.171875% with0exact,decoderCE43.153 and MSE1.32778. This shows poor
transfer in the current subset; the completed results above confirm poor transfer.
The terminal
interrupt route has not been verified as safe on this host. Subsequent calls
must use `--resume`; source hashes, optimizer and random streams are restored.

## Reproduction and verification

Evidence: `dump/decoder-reader-flow-20261013`. Preparation is complete; do not
re-encode it. Two CPU cache/mask/label tests pass. The completed experiment need
not be retrained; independent verification uses the same allocator budget:

```powershell
& D:/Git/latent-text-compressor/.venv/Scripts/python.exe -B -u tools/run_decoder_reader_verification.py
```

The verifier runs only after both 2,000-update arms finish and release the
worker lock. It reloads models, reproduces outputs, re-encodes fresh records,
checks frozen-codec identity, evaluates matched old controls on the same new test,
reports source-specific scores and uncertainty, and measures inference time.
These are bounded research commands, not a recommendation to restart full-v4.
Verification/allocator/probe sources are separately archived in
`verification-sources-20261013-171554.zip`, with hashes and a successful outcome in
`verification-launch-20261013-171554.json`. The production run remains paused.
