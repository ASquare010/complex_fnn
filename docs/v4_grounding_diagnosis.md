# Why v4 predicts fluent patterns but retrieves facts poorly

Read-only diagnosis of final48,762-step checkpoint, SHA256
`e17b9a8469b5a8c996cc64ff0677eb1c8f52410af52c1c179fd317263b4d9e2d`.
No optimizer, training, model changes, data changes, or parameter increase.
All outputs are in `dump/unified-v4-grounding-diagnosis`.

## Same simple fact in recent tokens versus old vectors

Eight facts cross two names(Amina/Rina) with four colors(blue/green/red/purple).
The same fact and question are used in all conditions. Greedy generation,64-token
cap; correct means the answer mentions only the expected color among the eight
recognized color words. Extra contradictory colors fail this criterion.

| Condition | Unambiguous correct color |
|---|---:|
| Short prompt; fact in raw tokens | 8/8 |
| Long filler first; fact still in raw tokens | 8/8 |
| Same filler after fact; fact only in old vectors | 0/8 |
| Old memory blocked | 1/8 |

Example fact: “Amina's favorite color is blue.” Recent-token answer: “Amina's
favorite color is blue.” Compressed-memory answer starts “Amina's favorite color
is red” and continues irrelevantly. Correct and blocked long conditions often
guess red; chance matches do not establish retrieval. This strict full-output
color score should not be confused with first-mentioned-color accuracy.

Long recent/compressed arms have the same messages/filler but reordered fact and
filler turns. Fact placement is checked against the actual512-token boundary.
Ordering, distance and token positions necessarily differ; the experiment is a
small diagnostic, not a representation-only randomized benchmark. Nevertheless,
the long recent control rules out prompt length alone as the explanation here.

## Three-record notebook retrieval is also difficult in raw text

32fresh generated cases(indexes outside the original40k), crossing familiar versus
held-out name pools and familiar versus held-out wording,8per combination.
Each case has three notebook facts and a three-part(color/container/room) value.

| Condition | Correct full location substring |
|---|---:|
| Short raw fact prompt | 1/32 |
| Long prompt with facts recent | 1/32 |
| Facts compressed | 0/32 |
| Compressed memory blocked | 0/32 |

Familiar names plus familiar wording did not solve this(0/8in every arm).
Thus the issue is not solely test names/templates or solely compression. The model
also struggles to bind one entity to a multi-part value among distractors.
Substring scoring is less strict than exact answer formatting.

## Does the encoder lose the facts?

The matching frozen decoder reconstructed all2,202fact-body tokens in the
three-record diagnostic correctly. The exported encoder and full codec produced
identical vectors(maximum difference0).

Across the complete64fact-containing chunks,4,064/4,096tokens were correct;
only32/64chunks were fully exact. All32errors were EOS token2 outside the fact
bodies, not colors/names/locations. Consequently do not call the codec completely
lossless on chat input. EOS/turn-marker representation deserves a separate check
before any codec-interface change. It does not explain loss of fact information
in these sampled chunks: those fact tokens were recoverable from the vectors.
This is a diagnostic use of the original decoder, not a decoder added to the LM.

## Loss implementation versus loss allocation

Reviewed `src/chat_data/dataset.py` and `src/chat_data/training.py`:

- Input shift uses block-start then preceding tokens; labels predict next tokens.
- User/system/tool content is attention-visible, with labels-100.
- Summed CE for each microbatch is divided by the supervised-token count of the
  entire effective batch. Accumulated gradients implement that batch objective.
- Frozen encoder outputs feed a trainable memory projection and transformer.
- Prior verification found identical generation/training-prefix logits on32cases.

No off-by-one, hidden-user-attention or accumulation normalization bug was found.
This is not a proof that every possible implementation defect has been excluded.

Independently scored128seeded memory dialogues each from train and test. Critical
tokens are BPE pieces overlapping color/container/room words in the final answer;
fixed “in the” wording is excluded. Remaining supervised tokens include names,
acknowledgments, fixed answer wording and EOS. This is teacher-forced accuracy,
not autonomous full-value accuracy.

| Sample | All memory CE / accuracy | Location-token CE / accuracy | Other-token CE / accuracy |
|---|---|---|---|
| Train128 | 0.1582 /93.22% | 0.8464 /63.35% | 0.0189 /99.27% |
| Test128 | 0.2244 /91.38% | 0.8175 /63.94% | 0.1109 /96.64% |

Only693of4,116supervised training-sample tokens are variable location words.
Whole-corpus memory examples account for0.6997%of scored training tokens.
Combining those gives an estimated0.12%of scored tokens directly teaching these
location components. This is a token-budget estimate, not a measured optimizer
gradient share or exact coefficient: batches are normalized independently.

Additional data-design constraints:

- Memory uses just12training names, one notebook relation, and four training
  question templates, always with three records and long repetitive filler.
- About25.4%of scored corpus tokens belong to reasoning-labelled sources; teacher
  traces teach style as well as correctness. Their factual validity was not fully
  independently verified. This is substantial competition for a23M learner.
- TinyStories examples use four generic story prompts. They teach story language,
  but not obeying arbitrary requested characters, plots or constraints.
- The previous mixed/specialist checkpoint's success was not inherited: this v4
  consumer started fresh. Its old results are not evidence this checkpoint has
  mastered the same skills.

These findings support a task/data/objective mismatch. They do not uniquely prove
capacity is sufficient for general chat or establish a65%accuracy ceiling.

## Proposed fix at the same parameter count (not executed)

1. Establish elementary grounded-answer competence: many fresh names and values,
   one fact first, then two/three facts and distractors. Include color, location,
   preference and simple ID relations; vary wording. Pair each task across recent
   tokens and compressed history, spanning several rollover distances. Change
   values for the same name so answering from a prior association cannot succeed.
2. Keep token CE, but give retrieval an explicit separately normalized loss share
   rather than hoping0.7%of tokens suffices. A10%retrieval share is a reasonable
   starting experiment, not an established optimum. Score informative answer
   values; omit/downweight acknowledgment boilerplate. Include full-sentence
   examples separately to retain natural output formatting.
3. For chat repair, prioritize short, verifiable grounded conversations and simple
   follow-ups, corrections and missing-information answers. Keep a language
   preservation component; avoid adding still more long reasoning data before
   basic answers work. A later bounded repair can use reasoning-off examples
   without deleting the original mixed corpus or changing the architecture.
4. Select improvements using generated value/answer accuracy, prompt adherence
   and repetition rates, alongside ordinary held-out CE. Include familiar and
   unseen entities/templates, correct/wrong/blocked memory and changed facts.
   Expand beyond these8simple facts before making strong claims. Use copied run
   outputs, retain finalv4as the baseline, and fix the evaluation before training.

No database, larger FFN, recurrent/vector-output redesign or extra training was
implemented. Adding a database alone would not repair the observed inability to
read the correct supplied memory. The first milestone should be elementary
grounded answers, not a general-purpose large-model chatbot.

Scripts: `tools/diagnose_v4_grounding.py`, `tools/diagnose_v4_single_fact.py`
(plus `--long-control`), `tools/inspect_v4_codec_errors.py`.
Evidence: `report.json`, `contrasts.json`, `single_fact.json`, `single_fact_long.json`,
`codec_oracle.json`, `codec_errors.json` in the diagnostic output folder. Original
checkpoint and frozen encoder file/tensor hashes unchanged; no gradients/updates.
