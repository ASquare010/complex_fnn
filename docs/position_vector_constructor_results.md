# Preserving decoder positions improves direct vector generation — 2026-10-13

The predictor can learn to construct useful new codec vectors from compressed
memory on this controlled task. Keeping all 64 frozen decoder position features
works much better than averaging them into eight groups. Jointly training the
shared reader and constructor then restores useful autonomous memory generation.
Unfamiliar record formats remain weak, and this is not yet a general language
model or a working growing-vector chatbot.

## Matched construction experiment

Both variants start from the previous eight-slot constructor checkpoint, use
exactly **23,209,216 trainable parameters**, and keep the original encoder and
decoder frozen. The common parent has 14,000 synthetic-task updates in its
lineage. Each receives 2,000 additional updates, batch 64, identical examples and
sampled allowed orders, AdamW LR 1e-4→1e-5, and:

`normalized MSE + 0.02 decoded token CE`

This phase supplies the correct source record and answer order to isolate
construction. Those oracle scores must not be presented as autonomous retrieval.

The only construction-input difference is eight means of eight adjacent decoder
states versus all 64 separate states. These are continuous hidden features from
the original frozen decoder; no token IDs or decoded strings enter the predictor.
Its output is a new 256-dimensional codec vector. The original decoder converts
that code into the eight-token answer.

The source bank contains 20,480 clean 64-token records across sixteen layouts,
both field orders, four complete text prefixes, forty training names and all 768
training value pairs. The previous `wasFile`/`forFile` join artifact is removed
by tokenizing the complete joined text. The 240 held-out nonidentical pairs and
their reversals remain absent from training. Source features are computed once
and reused; targets remain loss-only.

Independent reload tests use 512 fresh cases per family. Both allowed plans are
evaluated when either order is permitted; scores average the allowed plans per
case. Exact matching requires the complete sentence, including both variable
words. Variable-word-only scores agree with exact sentence scores.

| Oracle-source construction test | Eight averaged groups | All 64 positions |
|---|---:|---:|
| Records from the training bank | 39.94% | **99.22%** |
| Unseen value pairs | 35.55% | **95.90%** |
| Unseen names and value pairs | 29.00% | **92.09%** |
| Two previously failed, held-out layouts | 0.39% | **27.34%** |
| Four additional held-out layouts | 0.10% | **8.01%** |

For unseen pairs, the paired case-bootstrap 95% interval for full-position minus
eight-group accuracy is **55.86–64.65 percentage points**. The improvement is
large under this matched budget. It does not establish that averaging destroys
all recoverable information or that the weaker variant could never learn.

Construction-only training also changes the shared reader. Autonomous exact
accuracy after this phase is only 17.58% with full positions, despite the high
oracle construction result. This exposed the need to retain reader supervision.

## Joint reader and constructor continuation

A separate 1,000-update continuation starts from the full-position checkpoint:
same shared 23.209M predictor, frozen codec, batch 64, LR 3e-5→3e-6. The objective
is:

`normalized MSE + 0.02 decoded CE + 0.25 source CE + 0.25 plan CE`

The final model has 17,000 updates in its synthetic composition/planning lineage.
Gold source/order labels supervise training, but are not passed into generation.
At inference the model selects a record by its learned scores and samples a
learned two-way original/reversed-order plan. It then constructs the answer code.
This is a predefined two-choice plan distribution, not unrestricted probabilistic
language generation.

The read pass uses one pooled decoder-feature vector per record and query.
The construction pass preserves all 64 positions of the selected record. Both
passes reuse the same trainable weights. Stored memory remains a codec vector;
the construction step temporarily expands the selected record's representation.
Frozen codec parameters and its compute are additional costs.

Independent final tests use 512 fresh cases per family, four sampled draws each:

| Autonomous generation test | Construction-only parent | Joint final |
|---|---:|---:|
| Unseen value pairs, familiar layout/wording families | 19.24% | **76.12%** |
| Unseen names | 15.67% | **53.42%** |
| Unseen question templates | 15.72% | **50.88%** |
| Two held-out record layouts | 3.76% | **11.87%** |
| Four additional held-out layouts | 1.66% | **4.69%** |

Parent and final predictions use identical fresh cases and sampling seeds.
The paired bootstrap interval for the new-pair gain is **52.29–61.47 percentage
points**. These are continuation gains, not equal-total-training comparisons
against the original token-memory model or a language-model benchmark.

For the main new-pair family, final exact answers are:

| Memory condition | Exact answer |
|---|---:|
| Correct memory | **76.12%** |
| Only the queried stored values changed | **78.71%** |
| Record order shuffled | **76.56%** |
| Wrong records from another case | 1.17% |
| Memory blocked | 0% |
| Question removed | 8.64% |

Among draws correct before a fact replacement, **94.61%** produce the correct
replacement answer after only that stored fact changes. This supports actual
dependence on the stored content. It is not merely retrieving a prewritten answer
vector: the input is a 64-token record, and the output code represents a different
eight-token answer sentence. Target-table mutation leaves generation unchanged.

An example from the saved fresh cases, chosen as the first original-order case
correct before and after replacement:

```text
Question: Please help. What are the two access words for Oliver?
          Return them in the order stored.

Original stored values: river, house
Generated: The access words are river and house.

Replacement stored values: rain, car
Generated: The access words are rain and car.
```

The final model selects the right record in 86.13% of main-family cases, and
sampled plans obey the requested order in 91.50% of draws. Supplying the correct
source and plan still gives 95.70% exact answers, so selection/planning remain
significant limitations. On the two held-out layout families the final oracle
scores are only 28.22% and 9.57%; those require better construction generalization
as well as better retrieval.

## Cost, correctness and limits

Preparing the 20,480-record FP32 feature bank took 25.14 seconds and stores
1,363,656,869 bytes (about 1.27 GiB). That expanded feature cache is an additional
storage cost; it is not the size of the compressed vectors alone. Training plus
periodic development checks took 133.9 seconds for eight groups, 155.3 seconds
for all positions, and 154.9 seconds for the 1,000-update joint continuation.
These timings exclude final tests and are for cached synthetic records, not
full-corpus throughput estimates. Full-position construction was about 16%
slower than the eight-group control for equal updates.

The batch-64 preflight peaked at 1.93 GiB allocated for full-position
forward/backward, excluding the full record bank and optimizer state. No optimizer
updates were made in that preflight. It checked predictor gradients, frozen
decoder gradients, blocked-memory invariance, target-table independence and
feature parity with the original decoder's forward hook.

The independent verifier re-encoded sampled bank records and checked stored
codes/features, verified train/held pair and reversal separation, and checked
the original frozen decoder tensors. It reconstructed all 8,192 checked tokens
in 128 fresh source records and all 8,192 target tokens in 1,024 possible answer
codes. These finite checks are not a blanket losslessness claim for the codec.

Saved predictor tensors match the final training checkpoints. Adam states show
exactly 2,000 updates in each matched arm and 1,000 in the joint continuation.
Source hashes, all protected parent/reference checkpoints, original encoder and
decoder, paused growing-vector weights, and v4 manifest remain unchanged. There
are no production model or v4 dataset edits.

The current evidence establishes a useful direct-vector memory operation with
fixed eight-token answers and known vocabulary. It does not yet establish broad
chat quality, arbitrary future-text sampling, variable output lengths/EOS,
8→16→…→64 replacement feedback, direct reuse of generated codes as later input,
DB operations, or a speed/quality win over a matched raw-token language model.
The broader goal remains active. The next work must test those requirements;
more successes on this small answer template alone would not complete it.

## Artifacts and reproduction

- `tools/train_position_vector_constructor.py`: matched construction pilots.
- `tools/preflight_position_constructor.py`: GPU correctness/fit checks, no updates.
- `tools/verify_position_vector_constructor.py`: independent construction and
  autonomous tests, paired statistics and artifact audit.
- `tools/train_position_memory_joint.py`: bounded joint continuation.
- `tools/verify_position_memory_joint.py`: fresh parent/final memory comparisons,
  counterfactuals and integrity checks.
- `dump/position-vector-constructor-20261013/` and
  `dump/position-memory-joint-20261013/`: protocols, source archives, checkpoints,
  outputs and completion audits.

These are research reproduction commands, not the full-v4 training command:

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\train_position_vector_constructor.py --output D:\Git\complex_fnn\dump\position-vector-repeat --positions both --updates 2000

& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\train_position_memory_joint.py --output D:\Git\complex_fnn\dump\position-memory-joint-repeat --updates 1000
```

The second command uses the preserved original full-position parent and record
bank; it does not automatically use the first command's new output. Both scripts
refuse existing fresh outputs and support `--resume` from their periodic
checkpoints with source/input, optimizer, horizon and RNG restoration. Full
interruption/replay equivalence has not been separately tested for these pilots.
