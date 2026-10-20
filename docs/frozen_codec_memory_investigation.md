# Frozen-codec memory investigation — 2026-10-13

Follow-up: [direct generation and probabilistic sampling results](probabilistic_vector_results.md).
Fresh one-pass and seven-pass denoising pilots are now complete; neither produces
useful held-out language. Original weights remain safe and no long run was resumed.

The growing-vector run stopped at update4595. Its full saved state remains at
`dump/growing-vector-23m-8to64-v4/last.pt`, SHA256
`80822ed7c4156c2f36c94807704783a778d4bf77fd01afbad0d0a74b850930f3`.
The user authorized bounded alternative experiments. No long run was restarted.
Both pretrained codec components, the original token-memory v4 checkpoint, and
the v4 corpus remain unchanged. No production architecture or resume code changed.

## Diagnosis of the stopped run

Saved development CE at3000/4000 was14.5865/14.5918; accuracy1.256%/1.090%.
MSE improved without improving text. The frozen decoder is a reconstruction
decoder; its reliability on exact encoder codes does not establish reliability
on predicted codes or their average across several plausible future texts.

On32 development blocks, genuine target codes decoded99.902% of the2048 newly
scored tokens, CE0.0526. Replacing each code with a point30% along the line toward
the predictor's code reduced accuracy to24.02%; at the predicted code it was0.78%.
This is a measured sensitivity in this sample, not a universal latent-domain bound.

Three300-update continuations used the identical4595 initialization,512 training
dialogues,128 disjoint development dialogues, one complete64-token assistant
block per dialogue, eight growing prefixes, batch32 and fresh AdamW1e-4. Memory
was bounded to the previous32 completed chunks. Codes were prepared once for
these small pilots. The original checkpoint and corpus were untouched.

| Objective | Final development CE | New-token accuracy | Normalized MSE |
|---|---:|---:|---:|
| Starting checkpoint on this subset |14.9364|0.793%|0.6490|
| MSE only |47.1419|0.452%|0.5547|
| CE only |16.6515|0.732%|0.7867|
| MSE +0.01 CE |21.8820|0.464%|0.5393|

All final arms had0/128 exact64-token blocks. Training fits improved more than
development, so these are not evidence for running longer on that small subset.
At one initial batch, head-gradient norms were CE9.65 versus MSE0.632; their
cosine was+0.629. Equal coefficients are not equal gradient influence, but these
gradients were not opposed at that measured point. These tests do not exhaust
loss weights, fresh initializations, stochastic generators or training budgets.

Evidence: `dump/frozen-loss-diagnosis-20261013/results.json`, three diagnostic
weights and `tools/diagnose_frozen_vector_losses.py`.

## Alternative: select an existing vector, then decode it

The predictor learns a query vector, scores eight stored record vectors and
chooses a record index with CE. Its output selects an existing code unchanged.
The frozen decoder recovers the selected record. Both the query and records are
encoded; the predictor receives no token IDs. The model retains23,209,216 trainable
parameters. This is supervised retrieval, not next-block language modeling or
open-ended generation. The correct record index is a loss label only.

Record order and values vary. Training excludes held-out name/value combinations,
names and question templates. Tests contain512 episodes per family. Changed-value
tests replace only the queried record's value. Shuffling preserves labels under
the permutation. Wrong-memory and blocked-memory numbers measure selection;
blocked selection does not itself establish memory use because all scores tie.
Counterfactual and query-removal controls supply additional evidence.

| Pilot | New histories | Held-out value bindings | Unseen names | Unseen wording |
|---|---:|---:|---:|---:|
| Short records,4 training templates,600 updates |98.44%|98.24%|96.09%|38.67%|
| Full64-token records,20 templates,1000 updates |94.92%|86.52%|71.68%|16.60%|
| Same full64 data, frozen-decoder pooled features,1000 updates |97.27%|95.70%|76.17%|47.66%|

The first and later pilots differ in both record length and question set; they
are not a causal length comparison. The last two use the same data, initialization,
training order and budget; only the predictor's input representation changes.
All evaluated target records reconstructed exactly from their true encoder code,
so selection and exact-record accuracy coincide in these particular tests.

The feature reader uses the frozen decoder's normalized hidden states, mean-pooled
over valid positions, for both query and candidate records. After selection it
decodes the original stored code. It neither generates a replacement latent nor
changes codec weights. The bounded record/query bank was precomputed: training
loop timing omits that preprocessing and is not a general language throughput claim.

## Independent reload verification

Fresh256-example families re-encoded records and questions and reloaded each
checkpoint. The feature verifier extracts states through the actual decoder
forward hook rather than the training helper's manual path. Per-case outputs and
labels are saved. Blocked-memory predictions are invariant to memory contents.

| Reader on full64 records | Familiar wording | Unseen names | Unseen wording |
|---|---:|---:|---:|
| Direct compressed codes |91.41%|71.48%|17.19%|
| Frozen-decoder pooled features |97.27%|72.66%|46.88%|
| Cosine similarity of those features, no trained predictor |47.66%|53.13%|38.67%|
| Feature reader with query removed |13.28%|12.11%|13.28%|
| Feature reader with wrong memory |13.28%|14.45%|16.02%|

Chance for eight records is12.5%. The trained feature reader improves familiar
retrieval substantially; unseen wording remains too weak for a reliable general
memory interface. Structured access-word records and fixed filler are a limited
domain, not a chatbot benchmark, natural-document QA or missing-information test.

Evidence directories:

- `dump/frozen-memory-router-20261013`
- `dump/frozen-memory-router-expanded-20261013`
- `dump/frozen-memory-reader-20261013`

Each contains protocol/results, archived pilot script, support-source hashes/zip,
weights and fresh verification. Active tools are `test_frozen_memory_router.py`
and `verify_frozen_memory_router.py`. Original/reference weights are preserved.

## Decision and remaining work

Changing CE/MSE weights alone did not rescue the current generator in these pilots.
Selecting valid stored codes is a working controlled retrieval path, and using
the frozen decoder as a reader improves transfer to different question wording.
Do not promote these specialist checkpoints as language models or replace the
original v4 checkpoint. Do not restart the4595 run as if the failure were fixed.

The broader objective remains open: reliable reading across varied records and
questions, then generation of new text conditioned on retrieved memories. Future
tests need unseen record layouts, variable values/lengths, missing facts and
multi-record answers, alongside ordinary language preservation. A memory read
success cannot substitute for proving future-vector generation.

For generative latent alternatives, [Large Concept Models](https://arxiv.org/abs/2412.08821)
studies regression, diffusion and quantized prediction; its sentence embedding
space is different from this exact-reconstruction codec. That is research context,
not evidence that diffusion will fix this model. The subsequent bounded denoising
pilot is documented in the follow-up linked above.
