# Frozen decoder features improve reading — 2026-10-13

Reusing the frozen decoder's internal representation improves memory selection
and transfer to new question phrasing. It does **not** fix answer construction
from unfamiliar record layouts. The broader direct-vector language-generation
goal remains unresolved.

## Matched experiment

Both variants start from `dump/shared-vector-planner-20261013/predictor.pt` and
receive 3,000 additional updates, the same examples in the same order, batch 64,
AdamW cosine LR 1e-4→1e-5, and the same objective:

`normalized MSE + 0.02 decoded CE + 0.25 source-index CE + 0.25 plan-distribution CE`

Each uses exactly **23,209,216 trainable parameters** and two passes through the
same predictor. The original encoder and decoder stay frozen. The common parent
has 8,000 updates in its composition/planning lineage; these are continuations,
not models trained from scratch for 3,000 updates.

The control reads raw codec vectors. The feature variant runs each supplied code
through the original frozen decoder up to its final normalized hidden states,
then mean-pools valid positions to one 256-dimensional vector. The predictor
reads those vectors to select the record and predict the order distribution.
No token IDs or decoded strings enter the predictor. Known chunk lengths are
used for pooling; no answer information is used.

Both variants still construct the new answer vector from the **original selected
record code**, plus the sampled order, and decode it through the original frozen
decoder. Thus this experiment changes the reader's representation only. The
sampled plan chooses between two predefined word orders; it is not an arbitrary
language planner. Frozen codec costs are additional to the trainable count.

The task remains eight 64-token records, 32 possible words, two-word values and
eight-token answer sentences. Training holds out 240 word pairs and 18 names,
and expands question wording with 24 extra base templates and four extra
instructions per order category. There are 40 training base templates total,
eight prefixes and three suffixes. Tests retain the previous held-out templates
and add eight new question templates and three new instruction phrasings.
Only the original two record formats are used during training.

## Independent checkpoint reload

The main tests have 512 fresh cases each, with four generated draws per case.
New-record-format tests have 256 cases each. Exact accuracy requires the entire
eight-token answer, correct values and an allowed order.

| Test | Raw codec reader | Frozen-feature reader |
|---|---:|---:|
| Unseen word pairs, familiar phrasing | 67.72% | **83.11%** |
| Unseen names | 55.52% | 57.28% |
| Previously held-out question templates | 25.15% | **38.09%** |
| Eight entirely new question templates | 18.75% | **47.71%** |
| Previously held-out order instructions | 32.71% | **48.14%** |
| Three entirely new order instructions | 35.50% | 34.77% |
| New record wording | 0% | 0% |
| Reordered record fields | 0% | 0% |

Paired case-level bootstrap 95% intervals for feature-minus-raw accuracy are
**10.74–20.17 percentage points** for familiar phrasing and **23.88–33.84 points**
for the eight new question templates. The intervals include zero for unseen
names and the three new order instructions; those are not established gains.

For familiar phrasing, the feature reader selects the correct record 90.63% of
the time. Its exact-answer score falls from 83.11% to 0.34% with wrong memory and
0% with blocked memory. Changing the queried stored values gives 81.98% exact
replacement answers. Removing the question gives 8.06%. This is controlled
memory use, not a general chatbot benchmark.

Given the correct source and requested plan as an oracle diagnostic, both
variants construct the answer correctly in essentially all cases using known
record layouts. Both reach **0% even with that oracle assistance** on the new
record layouts. The two new layouts express the same facts:

```text
Two access words were stored for {name}: {first}, followed by {second}.

Archive entry. Second access word: {second}. Owner: {name}.
First access word: {first}.
```

The feature reader still selects the right record in 39.45% and 29.30% of these
cases, versus 6.25% and 12.11% for the raw reader. Its constructor then fails.
The frozen codec reconstructs all **32,768/32,768 source tokens** in 512 checked
records across these layouts. These facts are recoverable in the tested codes;
the trained constructor has learned too much dependence on the original formats.
That diagnosis does not establish a universal model-size limit or prove all
codec inputs are lossless.

## Cost, verification and next step

Training plus periodic development evaluation took 455.8 seconds for raw codes
and 499.9 seconds for features—about 9.7% more time for equal updates. At 1,000
updates the feature model had 62.21% development exact accuracy versus 39.16%
for the control. These timings are for this small synthetic task, include
on-demand feature computation during training, and are not full-v4 throughput
claims. This pilot does not cache the derived decoder features. Stored-record
features could be cached in an eventual inference implementation.

Verification checks include exact trainable count, original frozen decoder
tensors, encoder freeze, target-table mutation invariance, blocked-memory
invariance and decoder-feature parity with a hook on the actual decoder forward.
Both checkpoints completed exactly 3,000 Adam updates. Original v4/growing-vector
checkpoints, codec files, parent checkpoints and v4 manifest remain hash-identical.
No production model code or original training data was changed.

The next experiment should address both exposed weaknesses: use the frozen
decoder representation when constructing an answer as well as selecting memory,
and train on varied record formats/field orders while reserving untouched layouts
for evaluation. A single pooled vector may discard useful ordering, so preserving
some position structure is a hypothesis to test, not an established fix. General
text generation, growing-vector feedback, EOS and DB operation still need proof.

Evidence and reproduction:

- `tools/train_feature_vector_planner.py` — matched bounded training; `both` runs
  the variants sequentially on the GPU.
- `tools/verify_feature_vector_planner.py` — fresh paired cases and layout tests.
- `tools/audit_feature_vector_planner.py` — codec oracle, statistics, source and
  protected-checkpoint hashes.
- `dump/feature-vector-planner-20261013/` — per-arm source archives, checkpoints,
  protocols, progress, results, independent verification and completion audit.

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\train_feature_vector_planner.py --output D:\Git\complex_fnn\dump\feature-vector-planner-repeat --representation both --updates 3000
```

This is a reproduction command for a bounded research experiment. It refuses
overwriting a run, saves periodic checkpoints, and has no resume CLI. It is not
the long-running full-corpus trainer and has not been promoted to one.
