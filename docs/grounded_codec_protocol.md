# Real grounded-answer preparation

2026-10-13. Preparation and independent verification passed. Preparation commands
perform no optimizer updates. A separate bounded comparison is specified below;
the full-v4 production run remains paused. This is the next data/measurement
gate after the [separate conditional head](conditional_head_flow_results.md)
reduced sampling work without producing useful text. The broader goal remains
useful direct-vector language generation against a matched raw-token reference.
A smaller reading task is a diagnostic stage, not a redefinition of that goal.

## Source and scope

[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) provides natural passages,
questions, annotated answer spans and deliberately unanswerable questions. The
official files are distributed under CC BY-SA 4.0; retain attribution to the
dataset authors (Rajpurkar, Jia and Liang) when distributing a derivative.

Downloads are pinned by SHA256:

- `train-v2.0.json`: `68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002`
- `dev-v2.0.json`: `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`

The prepared subset has 8,192 training questions, 512 internal development and
512 exploratory test questions. Ten percent of official training article titles
are reserved for internal development, before selecting questions. Official dev
supplies the exploratory test; it is public and must not be called a hidden or
previously unexposed benchmark. Article titles, normalized full paragraphs and
normalized questions are disjoint across the selected splits. At most two
questions share one passage inside a split.

The source files contain 118,613/11,706 questions in the proposed train/internal
dev article groups and 11,873 in official dev. The eligibility rules retain
109,716/10,536/11,036 before bounded selection and overlap exclusion. They reject
long inputs or answers rather than cropping around the answer: full rendered
passage <=384 original-codec tokens, question/prefix <=96, first answer <=24.
The selected answerability mix is approximately 80% answerable and 20% missing;
this is an explicit experimental mix, not the natural source distribution.

A bounded reserve is scanned against all original v4 JSONL splits. The scan
casefolds strings and checks literal complete questions and first/middle/last
96-character passage snippets in both common JSON Unicode encodings. Any matched
reserve question is excluded. This is a conservative literal-overlap check,
not semantic/paraphrase decontamination. All source text hashes are recorded and
checked after scanning. The original corpus is not modified or duplicated.

## Exact input/output contract

```text
Passage:
<complete supplied passage>
Question: <question>
Answer: <short answer or Not stated.>[END]
```

The relevant full passage is supplied by the dataset. This means oracle passage
retrieval, not oracle intra-passage reading: the consumer still has to locate
the fact and connect it to the question. No gold answer span or source index is
an inference input. Evidence token/block indices are labels and diagnostic
metadata only. Any later oracle-evidence-chunk view must be explicitly labeled
and supplied equally to the raw and vector arms; its score is not autonomous QA.

Passage and question are tokenized/chunked separately, into independent <=64-token
codec blocks. This lets wrong/blocked-passage controls preserve the complete
question's token IDs and code values. The raw control receives the exact same
concatenated token sequence; the explicit BPE boundary applies to both arms.
Original frozen encoder and decoder are used. Metadata preserves actual lengths
and chronological order.

The answer is encoded in independent eight-token blocks; the final block is
space-padded for the encoder only. CE labels exclude padding. The ordinary-text
`[END]` marker avoids the earlier unsupported control-EOS2 failure. All answer
and end-marker labels occur once. Future answer codes are targets only for the
current query; only earlier genuine answer codes may be teacher-forced during
training. Free generation must append its predicted codes unchanged and stop at
the decoded marker. Neither predictor nor frozen decoder receives input token
IDs in the vector arm.

For answerable examples, all supplied reference answers are retained for scoring;
the first source annotation is the training target. No example is selected on
decoder reconstruction success. Preparation reports actual passage, evidence,
answer-body, exact-answer and end-marker reconstruction errors separately.
Inaccurate codec targets must remain visible in the report instead of silently
being removed to improve model scores.

## Frozen bounded comparison protocol

Preserve the 23,209,216 trainable budget in every learned arm, with codec cost
reported separately. Compare the raw control, deterministic new-code prediction
and conditional flow on identical source information, question order and answer
exposure. Train each arm fresh for 2,048 updates, batch 32, exactly eight passes
over the 8,192 selected questions. This gives 629,296 assistant-label exposures,
including 367,152 answer-body labels, per arm. This is a small reading experiment,
not a full-language training recipe. Raw microbatch is eight; vector microbatch
is 32. All models use FP32, seed 47, AdamW (weight decay .01), gradient clipping
at one, 50-update warmup to 3e-4 and cosine decay to 3e-5. Batch shuffle seed is
71410 plus epoch. Sources and data are hashed and archived before training.

Give informative answer-body supervision an explicit allocation rather than
letting short-answer end markers dominate; apply the same token weighting to
raw and vector CE comparisons: .9 times mean answer-body CE plus .1 times mean
end-marker CE. These coefficients are declared starting choices, not validated
optima. Vector arms use normalized code MSE for the first 200 updates, then add
.02 times the weighted CE through the frozen decoder. Deterministic prediction
receives a zero query. Flow predicts the clean code from a linear noise/target
interpolation at a uniformly sampled time. Both share the same conditional-head
architecture and initialization. The raw control has the same parameter count
and backbone family, but token embeddings and a different allocation of FFN
width; this is not an identical-architecture comparison.

This first comparison supplies the relevant passage but does not add evidence
selection CE. Span labels are retained for a subsequent supervised-reader
diagnosis. Successful source selection alone would not be a correct answer.

Every 512 updates, evaluate free generation on a fixed 64-development-example
subset. Final evaluation uses all 512 development and 512 exploratory test
questions. Flow uses eight Euler steps, fixed before observing results. Every
arm stops at the literal marker or 32 generated tokens; all reference answers
fit this limit. Report answerable and unanswerable scores separately, so a
constant missing-answer response cannot appear to demonstrate reading.

The actual save/resume boundary is exercised after one optimizer update. The
trainer restores model, Adam, RNG, next batch and the original horizon; periodic
checkpoints are every 256 updates. It never resumes the production model.

## Verified preparation

The overlap scan excluded 2,156 reserve questions. Independent source/offset,
split, label-count, causal-visibility and fresh sampled re-encoding checks passed.
Dev true-code reconstruction recovered 5,114/5,116 answer tokens (510/512 complete
answers); exploratory test recovered 4,849/4,849 (512/512). All 2,667/2,404 annotated
dev/test evidence tokens and all 1,024 end markers were recovered. Errors in
other passage tokens and two dev answers remain recorded; the codec is not
claimed universally lossless.

The weighted-objective full-batch-32 zero-update probes passed for all three
arms. Maximum measured allocation was 1.815 GiB for raw and .520 GiB for vector
arms, including Adam-sized moment buffers. These probes exclude optimizer steps,
evaluation and saving, and are not a whole-run speed or worst-case guarantee.

Report answerable EM/F1, missing-answer accuracy, stopping, actual generated
answers, and full predicted-code feedback. Include correct, changed, wrong and
blocked passage conditions with an unchanged question. Natural-text replacements
must preserve consistent evidence/answers and cannot be arbitrary word swaps.
Reordering or removing passage blocks changes positions; disclose that instead
of claiming a representation-only intervention. Also separate an oracle-evidence
construction diagnosis from learned evidence selection.

The dataset is short extractive reading, not general chat, open-ended creative
generation, multi-hop reasoning or a memory database benchmark. Success must
later transfer to complete new-sentence answers and ordinary language before
the broader goal can be considered achieved. Do not equate vector-decoder CE
with autoregressive token-model NLL.

## Commands and artifacts

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\prepare_grounded_codec.py
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\prepare_grounded_codec.py --prepare
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\verify_grounded_codec.py
```

The first command downloads/checks source identities and creates a bounded
selection; the second encodes it under the shared GPU-worker lock; the third
independently checks original source correspondence, split separation, packing,
fresh sampled re-encoding and matched full-size zero-update model execution.
Completed outputs are refused rather than overwritten. These are preparation
commands, not model-training commands. Evidence is stored in
`dump/grounded-codec-data-20261013` with source archives and hashes.

The separate bounded trainer is `tools/train_grounded_codec.py`. Its completed
`--preflight` performs zero optimizer updates. `--stop-after 1` starts and pauses
the first arm; `--resume` continues the declared comparison. Checkpoints and
results are under `dump/grounded-codec-20261013`. Training and independent
verification are now complete; see [results and limitations](grounded_codec_results.md).
No arm establishes useful held-out factual answering, including the raw control.
