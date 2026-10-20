# Reserved final notebook literal-generalization test

18 October 2026, before preparation or any student prediction. Complement the
full ORIGINAL test split; do not replace it or use these scores for pilot/model
selection. Student scoring is allowed only at actual step48,762 with verified
361,357,104 original assistant labels/two original epochs. Input preparation and
source audits may run now on CPU, without touching any live training source/run.

Use the original textual notebook statement, question templates and answer format.
Three groups have64 positive four-way units each: known names/known values;
known names/novel literal values; novel names/novel literal values. Every memory
contains three distinct records. Ask two names against both memories obtained
by swapping their values; the third record is a distractor. Include full answers
and proper known-name/relation completion prefixes. Add32 missing units with
two absent names and both swapped memories, full format only. No model/decoder
success filtering. Any filtering is solely token length/serialization structure.

Novel fields use deterministic synthetic identifiers beginning qz (Qz for names).
Keep the grammatical frame "in the COLOR CONTAINER in the ROOM"; identifiers
are arbitrary stored labels, not a claim about real colors/rooms. Each selected
identifier must be absent as an exact whole word from ORIGINAL TRAIN JSON and
its actual leading-space BPE sequence must be absent from original TRAIN token
bytes. Subword pieces themselves are known vocabulary; call this novel literals,
not an unseen tokenizer vocabulary. Report exclusions for preexisting strings.

Within each quartet, name BPE lengths and value BPE lengths are fixed. Source
statement/field coordinates, source wordbag, query phase and common answer prefix
must match across the appropriate counterfactuals. All facts are old. Completed
prefix length is8/16/32 blocks plus32 raw tokens, balanced by unit index. Entire
questions remain raw and all correct target tokens including EOS fit before the
next rollover. No facts or target values enter a completion prefix; construct it
from the queried name/relation alone and independently prove exact BPE-prefix
consistency for both swapped targets and unrelated probe values.

Deduplicate exact raw64 prefix blocks into an input bank, with per-case block
indices. Any later semantic code preparation uses the same separate frozen
encoder, preserves all blocks without reconstruction filtering, and records exact
weights/buffers/hashes and zero updates. Student literal-ID packing remains the
normal RAM-only representation. No inference architecture is introduced here.

Independent ALL-case audit must re-encode actual messages, parse actual questions
and source statements, derive answers/prefix/coordinates, verify all block indices,
missing cases, no future evidence, all four-way length/wordbag/raw-query/phase/
prefix/no-roll conditions, exact nonce exclusions and all input/source hashes.
No decoder or student is loaded for data construction/audit. Score complete text
plus EOS, both swapped pairs and all-four; disclose synthetic finite-suite limits.
Lower teacher-forced loss, correct one-question swaps or gold-name completion
alone cannot establish broad free factual recovery or completion of the goal.
