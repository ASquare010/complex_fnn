# Shared probabilistic vector planner — 2026-10-13

Follow-up: [frozen decoder features improve reading](feature_vector_planner_results.md)
in a matched continuation, but both variants fail on unseen record layouts.

The predictor can construct new codec vectors and sample between two valid answer
orders. Combining that with autonomous retrieval still leaves a major question-
understanding problem. This is a controlled memory experiment, not a chatbot or
a successful general-language replacement for the token-memory baseline.

## What ran

Exactly **23,209,216 trainable parameters**, reused for two passes:

1. Eight stored 64-token record vectors plus one encoded question → source scores
   and a learned distribution over original/reversed answer order.
2. The selected original record vector plus the sampled order → a newly predicted
   eight-token answer vector. The original frozen decoder reconstructs its text.

The second pass receives a fixed active vector and the order phase; it does not
receive the question again. Both encoder and decoder stay frozen. Neither pass
receives token IDs. No target vector, true source index or correct answer enters
inference. The ordering modes are explicitly designed; this does not learn an
arbitrary language-planning space. Record selection is argmax; order is sampled.

Training starts from the previous sampled-plan checkpoint, which already has
4,000 updates in its composition/planning lineage. This integration adds 4,000
updates, batch 64, AdamW cosine LR 1e-4→1e-5, with:

`normalized MSE + 0.02 decoded CE + 0.25 source-index CE + 0.25 plan-distribution CE`

The true source and an allowed order are supplied to the construction pass during
training only. Three requests specify original, reversed or either order. The
last uses a [0.5, 0.5] plan target. The plan logits reuse two existing head outputs
in read phase 3; construction reuses phases 1/2, without extra parameters.

Data covers 768 training word pairs and 240 held-out non-identical pairs, with
all 32 individual words known. Reversals of held-out pairs also remain held out.
Training uses 40 names, 16 base question templates, eight prefixes, three suffixes
and four phrasings for each order request. Eighteen names, four base templates
and three order phrasings are held out. This is deliberately narrow synthetic
data, not the v4 language corpus. The held-out answer vectors never enter training
losses or normalization. Having all target codes in an evaluation table does not
give the model access to them.

## Independent saved-checkpoint results

Each family has 512 fresh cases and four generated draws per case. “Exact” means
the whole eight-token answer has the correct facts and an allowed order.

| Test | Exact valid answers |
|---|---:|
| New word pairs, correct memory | 44.73% |
| Change only the queried old fact; score the replacement | 45.56% |
| Shuffle record order | 46.73% |
| Wrong memory from another example | 0.39% |
| Blocked memory | 0% |
| Remove the question vector | 7.57% |
| New pairs and new names | 37.55% |
| New base question phrasing | 14.36% |
| New order-instruction phrasing | 17.53% |

The paired, case-level bootstrap 95% interval for correct-minus-wrong exact
accuracy is **40.19–48.63 percentage points**. Among originally correct draws,
86.24% follow the changed fact correctly. Thus memory is used, but retrieval is
not reliable. The separate first test draw was 41.31% correct; the table above
reports the independently reloaded verification, not a selected best checkpoint.

For familiar phrasing, source selection is 45.70% and sampled orders obey the
request 97.90%. Supplying the true source as a diagnostic raises exact answers to
98.24% with the predicted argmax plan. Supplying both true source and requested
plan gives **100%** across all four fresh families. These oracle diagnostics are
not autonomous generation scores. New order phrasing reduces plan compliance to
54.79%, illustrating template dependence as well as retrieval difficulty.

On “either order” questions, the mean learned original-order probability is
0.514. One correct example generated both:

- `The access words are sun and light.`
- `The access words are light and sun.`

All 1,024 true answer vectors decode exactly. An additional 512 encoded questions
reconstruct **16,238/16,238 tokens**, including the held-out wording families.
The information was recoverable in these tested inputs; the trained predictor
has not learned to interpret it robustly. This does not prove all codec inputs
are lossless, that 23M is insufficient, or that extra updates cannot help.

## Verification, cost and saved artifacts

Training plus development evaluation took 597 seconds. Batch-32 generation
through two shared predictor passes and the decoder took about 41 ms, excluding
encoding. These timings concern eight-token synthetic answers and are not a
speed comparison against full v4 dialogue training.

The verifier checks unchanged frozen decoder tensors, frozen encoder parameters,
the exact trainable count, held-pair separation, invariance to replacing the
target-code/label tables, and invariance to masked-out memory values. The original
v4 and growing-vector checkpoints, codec files and parent checkpoint remain
hash-identical. All 4,000 Adam updates and archived source hashes were checked.
The faster decoder projection matches full projection argmax, with maximum
logit difference 4.58e-5 and gradient difference 2.43e-8 in preflight.

Files:

- `tools/train_shared_vector_planner.py`: bounded pilot; refuses fresh overwrite.
- `tools/verify_shared_vector_planner.py`: independent reload and ablations.
- `dump/shared-vector-planner-20261013/`: protocol, source archive, progress,
  resumable `last.pt`, model-only `predictor.pt`, results, verification and audit.

The pilot saves at step 0 and every 500 updates, restoring optimizer, schedule,
RNG and deterministic example/plan draws on `--resume`. It discards a partially
completed update on interruption. The CLI is a research pilot, not a replacement
for the full-corpus trainer; no full interruption/replay lifecycle claim is made.

To reproduce the bounded run in a new directory:

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe -B -u D:\Git\complex_fnn\tools\train_shared_vector_planner.py --output D:\Git\complex_fnn\dump\shared-vector-planner-repeat --updates 4000
```

To resume that run after interruption, use `--resume` with its directory. The
stored total-update horizon remains fixed. The completed existing run has no
remaining training updates.

The next bottleneck to investigate is question-to-record selection and wording
generalization. Repeating a long v4 run with this prototype is not justified by
these results. General language, variable-length/EOS handling, growing predicted
vector feedback, multi-record reasoning and DB operation remain unresolved.
