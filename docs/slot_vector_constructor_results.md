# Varied-record vector construction remains unresolved — 2026-10-13

Neither of the two matched continuations learned reliable answer construction
from varied record layouts. Reusing eight ordered groups of frozen decoder
features lowers the training losses, but does not establish an exact-answer
improvement over the original codec-vector input. These checkpoints are research
artifacts, not a recommended full-corpus training initialization.

## What was tested

Both arms start from `dump/feature-vector-planner-20261013/features/predictor.pt`
and receive 3,000 updates with batch 64, identical examples, sampled plans and
cosine LR 1e-4→1e-5. The common parent already has 11,000 updates in its synthetic
composition/planning lineage. Each continuation therefore has 14,000 total
lineage updates; this is not a fresh 3,000-update model.

Both retain exactly **23,209,216 trainable predictor parameters**, the original
frozen encoder and decoder, and direct codec-vector output. The shared predictor
runs twice: it selects a record and samples an answer-order plan, then constructs
an eight-token answer code. Gold source and allowed plan labels supervise training;
inference uses the predicted source and sampled plan. The plan distribution has
two predefined choices, original/reversed word order. It is not a general
language planner or an unrestricted probabilistic distribution over codec space.

The reader uses one mean-pooled frozen-decoder feature vector per source record
and query in both arms. The construction inputs differ:

- **Raw:** the selected original 256-dimensional codec vector.
- **Eight slots:** eight ordered 256-dimensional vectors, each averaging eight
  adjacent normalized hidden states from the original frozen decoder.

No token IDs or decoded strings enter the predictor. Stored memory is still one
codec vector per record; deriving the eight construction features adds frozen
decoder work and expands the selected record during computation. Codec costs
remain additional to the trainable parameter count.

The objective is unchanged:

`normalized MSE + 0.02 decoded CE + 0.25 source CE + 0.25 plan CE`

Training adds sixteen record layouts, both field orders and four fact offsets.
A deterministic bank contains 20,480 records: 512 per training name, with pair,
layout and offset selected independently. It covers all 768 training word pairs.
Evaluation uses 240 held-out nonidentical pairs, with reversals also excluded from
training, plus held-out names, questions and six held-out record layouts.

## Independent checkpoint reload

Each row below uses 512 fresh cases and four generated draws per case. Exact
accuracy requires the complete eight-token sentence with the correct two values
and an allowed order. Checking only the two variable word positions gives the
same correct-memory scores, so punctuation or fixed answer wording does not
explain these failures.

| Test | Raw construction | Eight-slot construction |
|---|---:|---:|
| Unseen pairs, trained layout/wording families | 0.49% | 0.59% |
| Unseen names | 0.20% | 0.29% |
| Unseen question templates | 0.20% | 0.44% |
| Previously failed, held-out record layouts | 0% | 0% |
| Four additional held-out record layouts | 0% | 0% |
| New layouts and new question templates | 0% | 0% |

The paired case-bootstrap 95% interval for eight-slots minus raw on unseen pairs
is **−0.68 to +0.88 percentage points**. This does not establish a benefit.
All-zero observed results in other families are not proof of a zero population
success probability.

On familiar layout/wording families, record selection is **91.02% / 92.97%** for
raw/eight-slot construction; sampled plans obey the request **91.55% / 91.89%**.
But even supplying the correct record and requested plan as an oracle yields only
**0.39% / 0.78%** exact answers. Both oracle scores are zero on unseen layouts.
The construction step is therefore a major bottleneck independent of retrieval.

Wrong and blocked memory both give zero exact answers in the main unseen-pair
test. Changing the queried values gives 0.98% / 0.29%; shuffled record order gives
0.49% / 0.59%. Correct-memory accuracy is too low to call this a useful retrieval
and generation system.

An additional 256 cases assembled from the actual training record bank give
0% / 0.49% exact answers. Correct-source-and-plan assistance gives 0% / 1.17%.
The failure is not confined to previously unseen value combinations: this bounded
recipe has not fitted construction reliably even for records available in training.
That does not establish that more training, another representation or a different
objective could never work.

The original frozen codec reconstructs all **24,576/24,576 source tokens** in the
384 checked records across trained and held-out layout families. The same records
were checked for both arms; they should not be counted twice. This establishes
recoverability for these inputs, not losslessness on arbitrary chat text.

## Text-boundary artifact and controls

The bank prepends token slices of filler before the record. Some shifted examples
join words without whitespace, such as `wasFile` or `forFile`. This is a data
quality limitation. The pilot and its archived sources were preserved rather
than silently changing its training data after launch.

Read-only controls used the same 256 cases per family, with either a separating
space before the record or no filler prefix at all:

| Trained-layout control | Raw exact | Eight-slot exact |
|---|---:|---:|
| Original shifted prefixes | 0.20% | 0.39% |
| Space inserted at the join | 0% | 0.39% |
| All records start at position zero | 1.37% | 1.17% |

All three controls remain zero on the held-out layouts. The codec reconstructs
all 4,096 checked tokens in each boundary/family condition. These inference
controls do not rescue the models; they do not prove that correcting the training
bank would have no effect. Any next training dataset should use clean text joins.

## Cost and verification

One-time encoding and frozen-feature preparation took **7.73 seconds** for this
small synthetic bank; `records.pt` occupies 189,251,941 bytes. Training plus
periodic development checks took **462.8 seconds raw / 457.1 seconds eight-slot**.
These are single runs with precomputed source features, excluding final tests.
They are not a full-v4 speed estimate or evidence that the larger construction
input is inherently faster.

The GPU preflight checked gradients reach the predictor while the decoder stays
frozen, fixed trainable count, blocked-memory invariance, and feature parity with
the actual decoder forward hook. The independent verifier also mutated the
target-code/label tables and confirmed generation is unchanged. Train-pair,
layout and offset coverage were checked. Both checkpoints have exactly 3,000
Adam updates; saved predictor and final checkpoint tensors match.

Source hashes, the original v4 and paused growing-vector weights, original
encoder/decoder, all parent checkpoints, the record bank and v4 manifest remain
unchanged. No production model code or prepared v4 data was edited.

## Interpretation and next experiment

The previous fixed-layout success does not transfer automatically to varied text
formats. Sampling a valid answer order avoids averaging two mutually exclusive
orders in a narrow task, but does not teach the constructor to read arbitrary
formatted records and synthesize the corresponding code. Lower MSE/CE in the
eight-slot arm is insufficient evidence of usable text generation.

Before another long run, a useful bounded diagnostic would preserve all frozen
decoder position states for the selected record instead of averaging groups of
eight, use clean text boundaries, and first check whether construction can fit
and generalize with the correct source supplied. This keeps the codec frozen and
the predictor's output a codec vector, but increases temporary compute. It is a
hypothesis, not a demonstrated fix. General-language generation, growing-vector
feedback, EOS handling and the DB goal remain unresolved.

## Evidence and reproduction

- `tools/train_slot_vector_constructor.py`: matched pilot and periodic resume.
- `tools/verify_slot_vector_constructor.py`: fresh paired cases and invariants.
- `tools/audit_slot_vector_constructor.py`: training-bank and boundary controls,
  optimizer/source/protected-artifact audit.
- `dump/slot-vector-constructor-20261013/`: bank, per-arm source archives,
  checkpoints, protocols, logs, results, independent verification, paired
  statistics and completion audit.

The following reproduces this failed bounded recipe for research; it is not a
recommendation to launch a long run. It requires the preserved parent checkpoint
and composition task artifacts described above, and refuses an existing output:

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\train_slot_vector_constructor.py --output D:\Git\complex_fnn\dump\slot-vector-constructor-repeat --construction both --updates 3000
```

For an interrupted arm, `--resume <run-root>\raw` or `--resume <run-root>\slots`
restores its last periodic checkpoint, optimizer, RNG and saved horizon with
source/input checks. A partial interrupted update is discarded. Full
interruption/replay equivalence has not been separately tested for this pilot.
