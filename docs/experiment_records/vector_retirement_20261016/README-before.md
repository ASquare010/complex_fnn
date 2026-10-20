# Transformer memory experiments

Ready for a user-launched full run: [four learned readouts per growing code](src/models/growing_token_memory_readout4/README.md).
One256-number code maps to four512-wide attention positions. Same23,209,216
parameters, fresh47 and original two epochs; existing prefix cache reused.
Twenty-four CPU checks and the real-data GPU parity/capacity check pass. Micro4
uses4.195GiB peak allocation with Adam-sized reserves. Measured forward/backward
is0.878s versus0.446s for one-position growing; memory KV is4x larger. Quality is
untested. Full command is in its README; no training was launched. Existing
query-reader run is preserved at its user-paused step6,834.

Previous experiment, now paused at6,834 updates: [query-conditioned growing reader](src/models/growing_token_memory/README.md).
Two reader layers keep one position per code, unchanged KV shapes and23,209,216
trainable parameters. Fresh47, effective32 and the original two-epoch schedule
are retained. Microbatch4 passed the idle-GPU benchmark and maximum-layout
capacity check. Measured cached forward/backward takes0.727s versus0.506s for the
old growing model; equal parameter/cache sizes do not mean equal compute or
quality. At6,000 updates, dev CE5.28943 versus5.27547 original growing shows no
early advantage. Its checkpoint is preserved. [Details](docs/growing_query_reader_setup.md).

Completed and evaluated: [raw64 gated memory cross-attention](docs/token_memory_64_crossattn_results.md).
All48,762 batches, labels and learning rates match the original. Test CE/accuracy
are2.24979/53.01%, versus2.21695/53.51% plain64 and1.80695/61.00% original512.
All three recover0/103 exact old notebook answers and0/32 changed facts. Hybrid
correct missing responses improve to18/25 but false missing responses rise to
78/103 answerable cases. No demonstrated factual-recovery improvement. Evaluation
preserves trained gates and independently reproduces prior baseline results;
models/checkpoints/cache remain unchanged. [Implementation](src/models/token_memory_64_crossattn/README.md).

The full decoder-expansion pilot was rejected and removed at the user's request
(2026-10-16). Expanding every code into64 positions defeated the intended sequence
compression; it also failed at its own memory cap after two updates. It has no
quality result. The subsequent query reader above preserves compact processing.
Historical original growing code/checkpoints, codec and caches are retained.
Removal audit: `dump/rejected-unpacking-removal-20261016.json`.

Evaluated: [64-token attention residuals](docs/token_memory_64_attnres_results.md).
The user deliberately stopped at37,045updates and requests no further training.
At matched37,000-step validation, token accuracy improves52.72% to52.96% over
plain64. Available-checkpoint test accuracy is original51261.00%, plain6453.51%,
attention residual6453.04%; the older models have48,762updates. All three score
0/103 exact notebook answers and0/32 changed facts. Attention residuals correctly
abstain22/25 missing cases but falsely claim missing information on87/103
answerable cases. No demonstrated factual-recovery gain; original512 remains
the strongest retained language foundation. Read-only checks pass; no training
was resumed. [Implementation](src/models/token_memory_64_attnres/README.md).

Completed: [64-token rollover comparison](docs/token_memory_64_results.md).
The user finished both epochs/48,762 updates. All training batches, labels and
learning rates match the original512-token baseline. Test CE is1.80695 original
versus2.21695 new; token accuracy61.00% versus53.51%, with all17 sources worse.
Both recover0/103 notebook answers exactly and0/32 changed facts. Missing-fact
answers improve13/25 to17/25, without reliable factual reading. Retain the
original language foundation. The [separate implementation](src/models/token_memory_64/README.md)
and its verified4.756GiB cache remain available. Read-only comparison checks
pass; no additional training or cleanup was performed.

Implemented and independently verified: [token-output growing-prefix pilot](docs/growing_token_memory_pilot.md).
The frozen encoder rebuilds one active vector from committed tokens after each
prediction, committing it at64tokens. Both arms retain23,209,216trainable parameters
and start from the same trained baseline, with1,024 matched updates each and
pre-encoded training prefixes. Answerable old-fact scores are baseline16/204 versus
growing6/204; both score0/204 on correctly following paired changed facts. Real
reading is1/209 versus0/209. No improvement or baseline replacement. Authorized
cleanup freed1.91GiB from redundant completed-pilot optimizer snapshots; original
baseline/codec and paused production remain intact. Sources, checkpoints, outputs
and independent verification are retained in `dump/growing-token-pilot-20261013`.

Completed and independently verified: [real grounded-answer comparison](docs/grounded_codec_results.md).
Three fresh 23.209M arms trained for 2,048 matched updates on 8,192 SQuAD2
questions. Held-out answerable exact scores are raw 1/410, deterministic vector
0/410 and flow vector 0/410. Correct/wrong/blocked controls do not establish
useful reading. True test answer codes decode correctly, but the learned models
do not generate them reliably. The raw control also fails, so this does not
isolate compression as the cause. The existing trained token + memory checkpoint
remains the best retained language foundation. Full-v4 production remains paused;
original weights/data are unchanged. See the [research recommendation](docs/frozen_codec_research_direction.md).

Completed and independently verified: [separate conditional diffusion head](docs/conditional_head_flow_results.md).
Two matched 23.209M pilots compute history conditioning once, then refine only
a small output head. Both preserve the original frozen codec and direct predicted
vector feedback. Sampling is faster, but output remains incoherent: zero exact
answers and zero end-marker stops in 34 free generations per arm; both fail all
32 old-note control conditions. Four-draw content agreement is 0.179% / 0.813%.
All saved answers/code floats reproduce exactly, and protected files are unchanged.
Both jobs are finished. No full-v4 restart or quality win is justified. The
[research recommendation](docs/frozen_codec_research_direction.md) now specifies
evidence-supervised reading and real grounded-answer construction as the next
bounded gate; preparation and the initial supplied-passage comparison are above.

Completed and independently verified: [clean-code versus velocity flow](docs/data_prediction_flow_results.md).
Two further matched 23.209M pilots preserve the original frozen codec and direct
vector feedback. Four-draw content agreement is 0.36% and 0.81%; both fail the
old-fact controls and generate incoherent answers. Genuine target codes still
reconstruct at 99.96% on the diagnostic dev sample; near-target denoising and
more sampler steps do not establish generation. All 34 saved answers/codes per
arm reproduce exactly, and originals are unchanged. No full-v4 restart or
quality/speed win is justified. The [research review](docs/frozen_codec_research_direction.md)
separates task-trained memory reading from new-code generation and describes
the smaller conditional diffusion-head alternative, now tested above.

Completed: [full-answer raw/vector/flow comparison](docs/full_answer_flow_results.md).
Three fresh 23.209M arms trained for 1,088 updates each on the same two epochs of
4,352 dialogues. Neither vector arm produced useful language; the raw control
also failed the old-note retrieval test. Independent reload reproduced all saved
answers, checked the frozen codec/Adam counters and unchanged protected files.
Original EOS control token2 was not recoverable by the codec; a reversible
literal end marker fixed true-code stopping representation but did not fix code
generation. No full-v4 restart or quality/speed win is justified. Production is
still paused at1,001. See the [research recommendation](docs/frozen_codec_research_direction.md).

Completed: [separate frozen-decoder position inputs](docs/decoder_reader_flow_results.md)
did not improve held-out generation over compact-code controls after two matched
2,000-update pilots. Independent reload, causal sampling and protected-weight
checks passed. Both retain 23.209M trainable parameters and the original frozen
codec. The expanded reader was slower; no full-v4 restart is recommended.
The [research review](docs/frozen_codec_research_direction.md) now distinguishes
SONAR-LLM's autoregressive decoder/re-encoded feedback from our stricter design,
and records the very limited language coverage of the small flow pilots.
[Full-answer selection and causal query packing](docs/full_answer_flow_protocol.md)
are recorded in the preparation protocol; the completed derived literal-marker
experiment is linked above. Original v4 data were not edited.

Latest [reversible latent-geometry test](docs/preconditioned_flow_results.md):
three additional matched pilots preserve the frozen codec and 23.209M budget.
The change did not produce useful held-out generation or beat the one-pass
control. [Research findings and the next reader diagnostic](docs/frozen_codec_research_direction.md)
separate recoverable information from a predictor's ability to use it.
Original checkpoints/data remain unchanged; no long training resumed.

Latest [continuous flow-matching comparison](docs/probabilistic_vector_results.md):
three matched2,000-update pilots retain23.209M parameters and the frozen codec.
Neither flow nor flow+CE beats the one-pass control on held-out eight-token
generation; no new full-corpus run is recommended. The production position-memory
run remains paused at1,001. Reference weights and prepared data are unchanged.

The [full-v4 position-memory joint trainer](src/models/position_memory_joint/README.md)
is prepared: exactly23,209,216 trainable parameters, original encoder/decoder
frozen, direct eight-token vectors, CE+MSE, bounded feature caching and same-folder
resume. V4 has no toy source/order labels, so this adaptation learns soft memory
selection through the answer loss. Data/causality/resume checks passed; no long
run was launched and broad-language quality remains unproven.

Latest [full-position frozen-feature pilot](docs/position_vector_constructor_results.md):
keeping all decoder positions improves direct answer-vector construction to95.9%
on unseen word pairs with the correct source supplied. Joint reader/construction
training reaches76.1% autonomous exact answers versus1.2% wrong-memory and0%
blocked-memory. Same23.209M trainable budget, original codec frozen. New record
layouts, broad language and growing-vector feedback remain unresolved.

Earlier [varied-record construction test](docs/slot_vector_constructor_results.md):
two matched 3,000-update continuations retain frozen encoder/decoder and direct
vector output, but neither learns reliable answer construction from varied
formats. Eight ordered groups of decoder features do not fix it. No long-run
training recipe has been promoted.

Earlier [frozen-feature vector pilot](docs/feature_vector_planner_results.md):
frozen encoder/decoder, direct vector output, 23.209M shared parameters. Reusing
decoder features improves question reading; unseen record layouts still fail.
General language generation remains unresolved.

The reference models and current experiment are:

| Model | Entry point | Context |
|---|---|---|
| Raw baseline | `src/models/raw_transformer/training.py` | Up to 2,048 older raw tokens + 512 current positions |
| Token memory | `src/models/token_memory/training.py` | Frozen encoder vectors + 512 current token positions |
| Growing vector experiment | `src/models/growing_vector/training.py` | One active vector grows 8→16→…→64; original frozen decoder; CE + MSE |
| Position-memory joint | `src/models/position_memory_joint/training.py` | Shared reader/constructor, frozen position features, independent eight-token codes |

Use each model's `--help` for arguments. Training starts only when you run the
command. All support named runs, checkpoint resume and microbatch controls.
See [training commands](docs/training_cli.md) and the
[growing-vector experiment, tests and commands](src/models/growing_vector/README.md).
The per-token re-encoding `vector_memory` implementation is retired; its source
archive remains under `dump`. Superseded vector/block-vector checkpoints were removed;
logs and diagnostic results remain. See `dump/growing-vector-cleanup-20261013.json`.

## Prepared chat data

See [the unified data recipe](docs/unified_chat_dataset.md) and
[the resumable chat trainer](src/models/token_memory/CHAT_TRAINING.md).
`token_memory_chat` was a redundant entry point, not another architecture; it has
been consolidated into `token_memory`. The older v2 run is preserved at update
2,168. The new v3 recipe adds public distilled chat, personas, tool calls,
reasoning controls and educational text. The DB variant remains future work.

[The v4 expansion](docs/chat_dataset_v4.md) adds42,210eligible tool/reasoning
dialogues, including more xLAM and published Qwen3 examples. It is audited and
ready at `dump/data/unified-chat-v4`; select it explicitly with `--data`.
Existing runs and the trainer's v3 default remain unchanged.

## Results and retired experiments

Latest controlled follow-up: [composition curriculum, two-pass memory and sampled
vector plans](docs/vector_curriculum_results.md). These retain the frozen codec
and 23.21M trainable budget. Familiar-format memory generation improves, while
unseen question wording and general language generation remain unresolved.

Current investigation: [frozen-codec direct vector generation and probabilistic
sampling](docs/probabilistic_vector_results.md). The bounded denoising pilot
remains poor on new text; it is not a recommended long-run training recipe.

Latest: mixed language/retrieval/conversation training retained controlled memory
retrieval and slightly improved held-out language loss. Conversation loss improved,
but generated replies still fail basic chat tests. A runnable experimental chat
CLI is available; it is not a ready general assistant.
See [mixed-training results and command](docs/mixed_memory_chat_results.md).

Token-memory completed 20,000 updates: validation CE **3.0131**, token accuracy
**41.30%**. The hierarchical decoder experiment reached CE **4.2009**, accuracy
**24.64%** at 20,000 updates and was retired. Its active code and weights were
removed; logs and archived sources retain the evidence. SwiGLU was also retired.
Read-only memory ablation confirms a small benefit from correct older vectors,
strongest just after rollover. Raw-baseline quality parity and exact factual
retrieval remain unproven; see the Step 3 report.

- [Current Step 3 design](docs/step_3_memory_llm.md)
- [Historical three-model comparison](docs/three_model_comparison.md)
- [Earlier repository notes and Step 1/2 results](docs/legacy_repository_readme.md)

The former `comparison` implementation now lives inside `raw_transformer`.
`tools/compare_saved_runs.py` reads historical results. The
`position_compressor` and `branch_sigmoid` modules include dependencies required
by the frozen encoder and token-memory model. They are not disposable copies.
Prepared datasets, the pretrained encoder, raw/token-memory checkpoints and
historical result logs remain local under `dump`. Fresh clones do not include
these artifacts. No root launcher or PowerShell training wrapper is required.


## Full-corpus growing token-prefix training

The per-token growing-prefix reader now has a separate two-epoch full-v4 trainer. See [the command and cache requirements](src/models/growing_token_memory/README.md). It uses a frozen encoder, 23,209,216 trainable parameters, and a resumable FP32 prefix cache on `I:\growing-token-cache-v4`. Full training has not been launched as part of setup.
