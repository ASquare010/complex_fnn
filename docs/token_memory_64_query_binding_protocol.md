# Question/fact binding diagnosis

2026-10-16. The previous retrieval-only experiment is complete and independently
verified. Its raw specialist solves every familiar single-record positive case,
but two-record errors are overwhelmingly the other record's value. The code
specialist additionally favors the first record. This protocol distinguishes
extracting a value from associating it with the question's entity. No optimizer
or architecture change is part of this diagnostic.

For every eligible positive two-record counterfactual pair, ask both entity
questions independently against each of its two memories:

| Input | Question A | Question B |
|---|---|---|
| Memory A=x, B=y | x | y |
| Memory A=y, B=x | y | x |

Use all eligible existing dev/test pairs and the first32 eligible training pairs
as a clearly labeled fit probe. Require equal prompt lengths, unchanged raw64
coordinates, the entire question in the raw suffix, all facts in old codes, and
the same answer prefix up to the first differing token. Filter only these input
properties, never a model score or reconstruction success. Report coverage and
all exclusions. Save all eligible training quartets for a possible later matched
training experiment; their existence does not mean training is authorized to run
without the normal preflight and independent source/label checks.

Use the already completed raw and code retrieval specialists, with their original
respective inference paths. Compare full-vocabulary first-decision accuracy,
complete free answers with EOS, both questions correct, all four correct, output
changes after each intervention, and wrong-record versus outside-record values.
One preferred record can achieve50% per-question accuracy but zero correct
question pairs. Code-blocking controls preserve counts/coordinates. Existing
public/reference parity, masked-content invariance, immutable checkpoint/source
guards and independent scoring are required. No full decoder or decoded source
is given to either reader.

Record a training-only candidate contrast, without treating it as quality:
let d_mq be logit(value x)-logit(value y) at the common first divergent token;
C=(d_00-d_01-d_10+d_11)/4. Additive query-only, memory-only and shared value priors
cancel algebraically. A later matched objective may test softplus(1-C) alongside
full-vocabulary answer CE and original-language replay. This does not prove that
the loss will teach broad retrieval, and it adds no inference parameters, attention
positions or KV entries. CPU checks cover cancellation, actual model gradients,
data causality and a fixed-record scoring counterexample.

This is a post-hoc mechanism diagnostic on previously inspected groups, not a
new held-out quality claim. The finite values, same-partition name co-occurrences
and short synthetic records remain limitations. Unseen-value generation,
historical tasks, language preservation and the eventual fresh two-epoch quality
comparison remain necessary. The goal is unchanged and unmet.

External context: [Zoology](https://arxiv.org/abs/2312.04927) distinguishes simple
synthetic recall from multi-query associative recall. Its results motivate testing
several bindings; they do not establish the cause or solution for this codec.

## Prespecified follow-up training comparison

The completed diagnosis found zero all-four successes in every group for both
specialists. The independent input/output audit passed. There are3620 eligible
training quartets (14480 examples); all are used, without model-based selection.

Two new copies of the completed **code retrieval specialist at2048** start with
fresh Adam. This deliberately builds on its existing value-reading ability; it
is not fresh initialization or a repeat of the original language two-epoch run.
The specialist and every original checkpoint remain immutable. Each arm receives
two passes of the new quartets:1810 updates,16 original language segments plus
four complete quartets (16 reading examples) per update, matching orders/labels.
Loss is .75 language CE + .25 answer CE; the binding arm additionally receives
.25 mean softplus(1-C). The ordinary arm sees the exact same four-way examples.
Thus the comparison isolates the extra objective rather than data exposure.

Use original legacy positions/model,23209216 parameters, FP32, AdamW.01, clip1,
LR3e-5 cosine3e-6, warm30, language seed4701660, reading seed4701661+epoch.
Microbatch8 preserves whole quartets. No missing examples are introduced into
the new positive-only quartets; original-language replay still includes its old
missing-information dialogues, and missing-answer regressions must be evaluated.
This is a limitation, not evidence that abstention has been fixed.

Before training require actual mixed-loss gradient/microbatch/serialized-resume
checks, fullsize independent public-forward gradients, heavy-batch capacity with
Adam-equivalent scratch, source guards and a first actual saved-update audit.
After training require all-step order/label/LR/Adam verification, exact free
four-way answers, original complete retrieval groups (including missing/unseen
values), full original language dev and historical transfer. No quality claim
from contrast or teacher accuracy alone; no automatic adoption. Production
inference has zero extra operations, parameters or KV entries for this loss-only
comparison. These data have been inspected, so results remain diagnostic.

## Additional read-only checks during training

A CPU-only gradient diagnosis used the first actual reading batch from the
immutable initial specialist: quartets1034/1439/2317/752,100 answer labels. Answer
CE gradient norm was5.101955, binding-loss norm1.605903 (ratio0.314762), cosine
0.183362. Both reading objectives have coefficient.25 in the binding arm; this
check excludes the language replay gradient. Every inspected layer group,
embedding and memory projection receives a nonzero binding gradient. Thus the
near-zero initial interaction is not a zero-gradient wiring failure on this
batch. This is not convergence or retrieval evidence. No optimizer update was
performed. Artifact: `dump/token-memory-64-query-binding-gradient-diagnosis-20261016.json`.

Related primary-source reading suggests separating codec reconstruction from
reader alignment. [ICAE](https://arxiv.org/html/2307.06945v4) combines autoencoding
and language-model objectives and reports its main results at4x compression with
Llama7B; its multi-span experiments also require training on concatenated spans.
[AutoCompressors](https://arxiv.org/abs/2305.14788) adapt pretrained language models
to use summary vectors in subsequent language modeling. Neither establishes
arbitrary64:1 factual reading in this independently trained23M reader.

[LoMA](https://arxiv.org/html/2401.09486v2) trains memory states through both
language prediction and masked repetition. Its reported4x/8x Llama2-7B transfer
tests have99.38–99.84% token reconstruction but35.30–71.56% completely correct
zones (Table2). The word "lossless" in its title is not a guarantee of perfect
record retrieval. These studies motivate testing alignment if the present
binding objective fails; they do not authorize replacing the codec, expanding
every code into64 attention positions, or claiming a solution before measuring it.
