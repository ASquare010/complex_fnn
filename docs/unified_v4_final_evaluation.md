# Final v4 checkpoint: memory helps prediction; retrieval and chat remain weak

Read-only evaluation of `dump/encoder-memory-unified-v4/last.pt`, update48,762.
SHA256 `e17b9a8469b5a8c996cc64ff0677eb1c8f52410af52c1c179fd317263b4d9e2d`.
Saved final development CE1.80471036, token accuracy61.01325%. These final
development figures are saved-run metadata, not a full development replay here.
No optimizer, training, model selection, parameter changes or source migrations.

## Held-out natural-context ablation

Seed47014 selected200non-memory test dialogues that have compressed history.
All432supervised segments from those dialogues were evaluated in each condition:
121,217assistant targets overall,71,562with history. Wrong memory comes from
another test dialogue with exactly the same history length. Blocked memory masks
memory keys in every attention layer, retaining slots and RoPE positions.

| Memory condition | History-target CE | Token accuracy |
|---|---:|---:|
| Correct | 1.99159 | 58.18% |
| Wrong donor | 2.11925 | 56.37% |
| Blocked | 2.13766 | 56.09% |

Wrong-minus-correct CE0.12766, paired document bootstrap95%CI[0.11636,0.13957].
Blocked-minus-correct CE0.14607, CI[0.13547,0.15722],5,000resamples.
On13,599scored positions in the first64positions after rollover, correct CE2.39149
versus blocked2.80566; accuracy51.11%versus45.35%.
All-target CE is1.93537correct/2.01074wrong/2.02161blocked; it includes positions
without history. This sample is deliberately history-bearing and excludes the
synthetic memory source; it is not the full-test aggregate or directly comparable
to the mixed development score.

Conclusion: useful causal contribution from old vectors to contextual prediction
on this sample. Not proof of arbitrary factual retrieval or superiority to raw
attention. Original natural-language ablation used a different corpus/checkpoint,
so differences in effect size are not a matched longitudinal comparison.

## Actual generated memory answers

128seeded test memory dialogues were decoded greedily, maximum64new tokens, with
correct/wrong/blocked memory. The reference answer was never supplied during
generation. Facts lie entirely before the active512-token window.

| Cases | Correct memory | Wrong memory | Blocked memory |
|---|---:|---:|---:|
| All128: exact whole answer | 13/128 | 13/128 | 11/128 |
| Fact present103 | **0/103** | **0/103** | **0/103** |
| Fact missing25 | 13/25 | 13/25 | 11/25 |

Ignoring answer wording still gives0/103correct location substrings in every
condition. Changing an old stored value while retaining the question gave0/32
exact answers and0/32correct new-value substrings. These are controlled synthetic
episodes with held-out names/templates, not a universal retrieval benchmark.

Example: expected “Rina keeps the notebook in the red cabinet in the hallway.”
Correct-memory output: “Rina keeps the notebook in the black cabinet in the bedroom.”
Replacing the stored value with “in the green drawer in the kitchen” still
produced the original incorrect black-cabinet answer in this example.

Low source CE can reward predictable answer wording while missing the variable
fact. The old specialized/mixed checkpoints previously passed different retrieval
tests. This is a fresh broadly trained model and has not reproduced that ability;
do not claim the prior checkpoint lost its skills or that compression cannot work.

## Fixed greedy chat prompts

All outputs are retained without rerolls in `chat_samples.json`.

| Prompt/test | Observed behavior |
|---|---|
| Greeting/help | Grammatically fluent but irrelevant privacy/technical-support refusal |
| Explain melting ice | Circular “a type of ice” repetition;192-token limit |
| Exactly three desk-tidying tips | Three identical irrelevant map-app suggestions |
| Mira green bicycle, Omar blue; ask Mira's color | Incorrectly answered blue |
| Three apples plus two, reasoning off | Invalid arithmetic; did not give correct answer |
| Lost dog story | Story-like prose about Lily, weak prompt adherence;192-token limit |
| Name Lina/favorite purple, first turn | Echoed the user |
| Follow-up favorite color | Recalled purple in recent raw-token context |
| Three apples plus two, reasoning on | Think-tagged repetition;256-token limit, no final answer |

The model has learned some language forms and short-context copying. It is not
a reliable chatbot. No tool calls were executed, no general benchmark rank is
claimed, and these few greedy examples are qualitative rather than a full study
of sampling strategies. Correct raw-context bicycle binding also failed, so the
weakness is broader than compressed-memory retrieval alone.

## Verification and evidence

- Model/encoder tensor hashes and checkpoint/encoder file hashes unchanged.
- All model and encoder gradients absent; blocked outputs invariant to memory.
- Empty-history controls identical across conditions.
- Test binary, loss mask, dialogue/index files and tokenizer matched manifest.
-32training-prefix versus generation-prefix checks: matching serialized tokens,
  identical first-token logits (maximum difference0),32/32argmax matches.
- The initial verification harness accidentally passed an already-mode-tagged
  stored dialogue through a helper that adds the mode; corrected in the harness.
  Main evaluation used exact stored prefixes; chat prompts had no pre-added mode.
  No evaluation predictions or model sources required changes.

Evidence: `dump/unified-v4-final-evaluation/report.json`, `natural_per_document.npz`,
`retrieval_predictions.json`, `changed_facts.json`, `chat_samples.json`, and
`generation_verification.json`. Source: `tools/evaluate_unified_memory.py` and
`tools/verify_unified_evaluation.py`. Reproduce in a new output directory; the
verification helper currently targets the named evaluation directory.

Next decision: evaluate improvements against exact factual binding and generated
answers, not only aggregate token accuracy. Do not infer a65%capacity ceiling or
extend training blindly. Prioritize focused data/objective diagnosis before
adding a database or replacing the token representation. No further training
is launched by this report.
