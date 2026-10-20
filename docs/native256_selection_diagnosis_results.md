# Selective-reading diagnosis after the balanced 512-step pilot

Evaluation and independent results verification passed. This is a
zero-update diagnosis of the completed pilot, not a new training run. Native256
and both saved 512-step arms remain unchanged. The broader quality goal is unmet.

The [balanced pilot](native256_balanced_pilot_results.md) established useful
reading of eight familiar answer words from one compressed vector. This follow-up
shows that the success depends strongly on the trained fact layout. Trainable
packing does not resolve the measured position or entity-selection failures.

## Controlled setup

Use the exact frozen/packing checkpoints from
`dump/token-memory-64-balanced-fixed-pilot-20261017`. Both were initialized from
the same copied native step8750 checkpoint with fresh Adam, then trained for512
updates with language replay. This is additional diagnostic training, not fresh
two-epoch language-model training. No completed run was extended here.

New diagnosis outputs are under
`dump/native256-selection-diagnosis-20261017`. Each condition has64 aligned units
and256 cases. Every16 cases balance the eight values, two examples per value.
Each source history contains47 body tokens, all facts are inside the first64
tokens, and the raw query begins at77. Every case supplies exactly one code;
the answer is one token plus EOS. Neither decoder features nor reconstructed
tokens enter reader attention. Prefix-only decision logits contain no answer
tokens. Production attention shape, parameter count and KV entries are unchanged.

Clean shifts preserve the full source token word bag and raw query. Distractor
and two-record memory swaps preserve both value words, changing their assignments.
Two-record cases ask both entity names against both memories; the reversed-order
condition also preserves the word bag. No case is filtered by model success.

## Complete free answers

All counts require the correct value and EOS, with unrestricted vocabulary.

| Condition | Frozen packing | Trainable packing |
|---|---:|---:|
| Familiar value at trained position24 | 235/256 | 238/256 |
| Same fact shifted8 tokens, value32 | 35/256 | 38/256 |
| Same fact shifted16 tokens, value40 | 22/256 | 23/256 |
| Fact at24, spare-value distractor after it | 222/256 | 222/256 |
| Spare-value distractor before fact shifted8 | 37/256 | 32/256 |
| Two records, A then B | 116/256 | 112/256 |
| Two records, B then A | 115/256 | 111/256 |
| Eight unseen answer words at24 | 0/256 | 0/256 |

With memory blocked, each familiar-value condition scores32/256, the constant-word
baseline. Fixed clean cases score0/256 with shuffled memory. This confirms useful
code dependence at the trained location. Fixed clean all-four success is52/64
and54/64; shifted, two-record and unseen-word groups have **zero all-four success**.
The fixed-fact/distractor condition retains48/64 and49/64 all-four successes.

Passing the spare-word test does not establish entity binding: the requested
fact still occupies the trained slot. In the two-record conditions, changing the
question changes only7/128 and5/128 frozen outputs, or5/128 and4/128 packing
outputs. First-decision query-by-memory interactions are small: .01843/.03343
frozen and .01648/.03555 packing for the two orders; blocked interactions are
exactly zero. Output changes or positive interactions alone are not correctness.
Across both orders, outputs repeat the FIRST record's value448/512 times frozen
and443/512 packing, versus only7/512 for the second record in either arm. This
supports a fixed-layout reading shortcut rather than successful name selection.

## Unseen-word ranking and encoded content

Restricting the decision to the same eight unseen words for every case gives
17/256 frozen and19/256 packing correct, versus38/256 and42/256 with memory
blocked. These are forced-choice diagnostics, not free generation. They do not
support the explanation that removing familiar-word output competition alone
would solve unseen-value recovery.

The original frozen reconstruction decoder independently reconstructs **all
65,536 source tokens and all1,024 blocks exactly** from both arms' codes, including
the moved facts, distractors, two-record assignments and unseen words. This
demonstrates recoverability by that decoder on this suite, not a globally lossless
codec or an ability already possessed by the reader. The decoder is never a
reader side channel. Packing's median relative code drift is .96873%; frozen
cached-packing recomputation differs by about3.79e-7 from original live codes.

## Verification and interpretation

Four CPU tests passed: balancing/positions/word bags, source disjointness, actual
answer-free prefix construction, and independently computed scoring/prior
cancellation. Independent data audit reserialized all2,048 cases and re-encoded
every one of the1,024 source blocks exactly. Evaluation asserts bitwise unchanged
reader and packing tensors, no optimizer updates and1,154 unchanged protected
hashes. Independent results audit passed checks of saved output IDs, full and candidate
logits, exact/pair/all-four scores, blocked invariance, interactions, source values,
reconstruction and drift. Full language validation is not repeated for unchanged
weights; its verified result remains in the balanced pilot report.

This is strong evidence of layout dependence and failed query/entity selection
for these checkpoints. It is not proof that a23M reader cannot learn either task,
or that a new attention architecture is required. The frozen encoder already
supports the narrow fixed-position task; this packing recipe adds no clear
generalization benefit.

The next separate curriculum test should first vary fact positions while keeping
the familiar values balanced and language replay intact. After position
generalization, add same-word-bag two-record swaps with both questions, then
broaden answer words and introduce balanced missing-fact supervision. These
training changes retain one attention/KV position per64 old tokens. No further
training is started or prescribed as an automatic extension of the finished arms.

Final adoption still requires broad held-out reading, new values, binding,
reasoning and language preservation in a fresh matched two-original-epoch
comparison. The historical61.013% reference uses raw512 plus older compressed
memory; it is not a pure all-raw1:1 reference.

Protocol: [prospective diagnosis](native256_selection_diagnosis_protocol.md).
Implementation: `tools/native256_selection_diagnosis.py`; independent audit:
`tools/verify_native256_selection_diagnosis.py`; tests:
`tests/test_native256_selection_diagnosis.py`.
