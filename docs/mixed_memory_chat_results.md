# Mixed language, retrieval, and conversation experiment

Completed 2026-10-12. **Memory retrieval was retained, ordinary language improved
slightly, but generated conversation is not yet good enough for a usable general
chatbot.** Original weights and the retrieval-only checkpoint remain unchanged.

## Training

Continued the retrieval-trained copy for 3,000 updates, preserving the23,209,216
parameter architecture and frozen encoder. This follows20,000 original LM updates
and2,000 retrieval-only updates; it is not a from-scratch or equal-budget baseline
comparison. New AdamW, LR3e-5to3e-6,100warmup. Separately normalized loss weights:
75%ordinary language,15%conversation,10%retrieval. Per-update batch sizes32/16/32,
microbatch8. Those weights are not percentages of tokens.

Ordinary language continued the original shuffled sampler after640,000 segments.
Conversation data:32,000 short examples plus256development/256test examples from
[HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
pinned revision f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc. Only one shard downloaded.
Questions are hash-disjoint across splits. Assistant-only CE includes EOS; first
assistant answer per conversation, complete sequence at most512tokens.

Retrieval:96,000 unique histories, eight records, sixteen answer words,
512history tokens compressed into eight vectors. Sixteen question templates train,
four development, four final test; record/word binding holdouts retained.
Final checkpoint selected by a prespecified development score, not test scores.
Selected step3000. Training plus periodic/final evaluation took about43.7minutes.

## Ordinary language

| Evaluation | Original CE | Mixed CE | Original accuracy | Mixed accuracy |
|---|---:|---:|---:|---:|
|100 development documents used in selection|3.01312|3.00493|41.30%|41.49%|
|900 additional documents excluded from selection|3.02685|3.02012|41.20%|41.33%|

The900-document comparison scores1,900,800 identical target tokens. Paired
candidate-minus-original CE=-0.006737, document-bootstrap95%interval
[-0.007401,-0.006128].712/900documents improved. This supports a small improvement
on this corpus, not universal improvement or parity with the raw-token baseline.

## Retrieval tests (512 examples each)

| Test | Correct vectors | Shuffled vectors | Blocked memory | Follows replacement fact when memory changes |
|---|---:|---:|---:|---:|
|New histories, familiar wording/bindings|100%|100%|0%|100%|
|Unseen question wording|98.83%|99.02%|0%|98.83%|
|Unseen bindings and wording together|86.13%|85.16%|0%|86.13%|

Wrong-memory accuracy against the original answer was0%/0.20%/0.20%, respectively.
Counterfactual pairs change only the requested old fact, keeping the question
identical. These are full-vocabulary next-answer-token predictions. They establish
controlled retrieval, not arbitrary names, strings, distances or conversations.
The new wording tests differ from the previous retrieval-only experiment's test;
do not present the two accuracies as a matched before/after comparison.

## Conversation: better loss, failed readiness check

On256held-out conversations, assistant-token CE improved from the original model's
2.74096 to2.12515; accuracy47.25%to55.39%. The retrieval-only starting copy had
CE2.97796, which is a different reference from the original model.

Despite this, greedy replies looped or failed simple instructions. For example,
"Hello!" produced repeated "get a chance" text; "What is two plus three?" did
not answer5. Sampling at temperature0.7,top-p0.9,repetition penalty1.15 and a
repeated4-gram restriction reduced loops but still produced irrelevant or false
content. The fixed prompt set and all outputs are retained; no success claim
is based on selecting a favorable sample. No additional training was launched
after inspecting these final tests.

The local chat front end runs, maintains a transcript, stops at EOS, and supports
512-token memory rollover. It is an **experimental model tester, not a ready
assistant**. Conversation training here was single-turn; multi-turn quality is
not established. Neither extra memory attention nor decoding controls alone
have established usable general chat quality.

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\tools\chat_memory.py --run D:\Git\complex_fnn\dump\token-memory-mixed-chat-3000-20261012
```

Use `/reset` or `/quit`; `--temperature 0` reproduces greedy decoding.
The tool refuses conversations beyond its tested context range instead of
silently dropping history. The command was executed successfully after training.

## Evidence and next decision

Run:dump/token-memory-mixed-chat-3000-20261012. Selectedbest.pt SHA256:
e84721d648f3b2b7b61205df65f6580543b5046344641e2f9a28a89286c56b68.
recipe.json,training.jsonl,development.json,before.json,after.json,result.json,
fresh_language_comparison.json/.npz,samples.json,sampled_chat.json and
completion_audit.json retain the evidence. Frozen training source copies are
inside the run.3,000sequential updates, source bytes, selected checkpoint hash,
saved prediction metrics, unchanged parent/encoder checks all passed.

The memory experiment succeeds at retaining retrieval without a measured
language regression. The chatbot requirement remains unmet at the model-quality
level. Next work should target conversational capability and free-generation
evaluation, with mixed replay retaining language and retrieval. Do not change
memory architecture or claim a general chatbot based solely on lower CE.
