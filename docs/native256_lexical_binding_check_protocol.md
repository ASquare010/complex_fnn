# Evaluation-only four-way notebook binding check

Prospective18October2026, while fresh lexical64 training is live and before any
trained quality evaluation. No model, training loss, corpus, code cache, checkpoint
or original run change. Zero additional training examples/epochs.23M reader and
one global position per64 old words remain unchanged.

Derive up to64 units from the ORIGINAL held-out development notebook dialogues.
Each unit retains its original background and query wording, selects two source
records and swaps their complete locations. Ask both people about both memories.
The four inputs have equal length/phase, fixed coordinates and the SAME source
token multiset; facts are entirely in old completed blocks. Changes between
memories occur only in old fact text; each query's raw tail is identical. Select
only by these structural properties, never decoder success or model output.
Declare the number of eligible/selected units. This is a correlated additional
control on held-out documents, not a new independent dataset or training-fit test.

Evaluate ordinary full replies and conditional completions. Completion prefixes
contain ONLY the queried name and fixed notebook relation, split at actual BPE
token boundaries BEFORE any value content. No reconstructed history or answer
is provided. Targets are independently derived from source statement regexes;
check all prompt serialization/query bindings/wordbags/positions/prefix offsets.
Block all fact-containing vectors for the first16 units with original physical
counts/coordinates preserved. Both swapped memories must yield identical blocked
outputs for a fixed question. The1:1 control blocks the same64-word ranges.

Report strict value/text+EOS, both-swaps success per question and all-four success
per unit, separately for full replies/completions and enabled/blocked source.
One correct output or changed output is insufficient. A constant answer or a
query-independent last-record selector cannot pass all four. First-token accuracy
is diagnostic only. Independently reaggregate all scores and public-forward
replay source-selected enabled/blocked cases from the copied immutable evaluation
checkpoint. Do not claim broad success from a partial1000-update checkpoint.
This supplements the full original-dev/natural/legacy/historical/reasoning suite;
it never replaces those checks or the full-budget matched comparison.
