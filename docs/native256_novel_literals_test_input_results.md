# Reserved final literal-test inputs verified

18 October 2026. Input construction and independent audits are complete. There
are zero student predictions and zero optimizer updates. This is not quality,
losslessness or convergence evidence. It complements the full original test.

| Group | Four-way units | Cases |
|---|---:|---:|
| Known names and values |64|512|
| Known names, novel literals |64|512|
| Novel names and literals |64|512|
| Missing facts, full answers |32|128|

Positive units include full replies and proper known-name/relation completions.
Each asks both names against both swapped memories, with a third distractor.
Across swaps, the source wordbag, statement/value coordinates, token lengths,
query phase and known prefix are controlled. All questions are fully raw, facts
are old, and every correct answer including EOS fits before rollover. Histories
use8/16/32 old blocks plus32 raw tokens. No structural rejection occurred.

The707 selected qz/Qz identifiers are arbitrary stored labels in the original
notebook grammar. Independent checks prove their exact whole words and actual
leading-space BPE sequences are absent from the entire original training corpus.
The subword vocabulary remains known. These are synthetic novel literals, not
unseen tokenizer tokens or broad world-knowledge questions.

Three construction tests passed. The independent audit checked all1,664 cases,
224 units, actual messages/questions/answers/source coordinates, sparse block
indices, all counterfactual and prefix/no-roll conditions, missing facts and
nonce exclusions. Original input hashes match the canonical original training
input hashes already used in the source-occurrence experiment.

There are835 deduplicated genuine full64-token blocks. The separate frozen
9,426,432-parameter encoder produced raw FP32 semantic codes without decoder or
student filtering. All835 independently re-encoded CPU codes match exactly;
all835 GPU comparisons give maximum absolute error1.907349e-5. Encoder weights
and buffers remain exact, with no updates. Encoder loading checks its exported
architecture-source hashes. Literal-ID packing remains a later RAM-only step.

Artifacts: `dump/native256-novel-literals-test-20261018`, including
`independent-verification.json` and `encoder-inputs/independent-{cpu,cuda}-verification.json`.
Sources and inputs are bound by the two manifests. See the
[prospective protocol](native256_novel_literals_test_protocol.md).

Student scoring remains reserved for actual step48,762 with a verified
361,357,104-label/two-original-epoch training audit. The final student evaluator
still needs implementation and must enforce that gate. No pilot score from this
test has been computed or used to choose an architecture/training objective.
