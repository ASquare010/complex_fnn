# Training-only code-content diagnostic, 16 October 2026

This is a bounded continuation experiment, not a new production architecture,
fresh initialization comparison, or completion of the language/retrieval goal.
The user's latest constraint allows modest compute/VRAM overhead while retaining
the main compression advantage. This experiment still changes inference by zero.

## Question and interpretation

Can the unchanged 23,209,216-parameter plain64 model learn to extract original
content at requested offsets from a single genuine compressed code when given
dense direct supervision? Size alone is not established as the failure cause:
the trained parent answers103/192 conditional raw-text completions correctly,
but0/192 corresponding distant compressed completions. These are conditional
completions, not broad independent question answering.

Synthetic associative recall can be learned by small transformers; the
[Zoology repository](https://github.com/HazyResearch/zoology) includes a two-layer
transformer example. Its [paper](https://arxiv.org/abs/2312.04927) finds a70M
attention model outperforming a1.4B gated-convolution model on recall. These
results motivate testing access and supervision, not a claim that this particular
23M compressed reader must succeed. Training only on retrieval would be a useful
capacity diagnostic but would not by itself preserve general language quality.

## Fixed protocol, written before corpus training

Data:4096 training,512 development,512 test full64-token blocks, source-balanced
across the17 original sources. Exact-block deduplication within/across selected
splits. Original corpus splits and tokenization retained. Train/dev code tensors
are exact copies from verified original FP32 caches; test blocks are encoded by
the frozen original encoder. No selection on decoder success.

Frozen original decoder is a diagnostic only. It reconstructs261843/262144 train,
32728/32768 dev and32735/32768 test tokens. All train/dev content tokens (>2) are
correct; test has one content error. This is not globally lossless compression.
Independent audit verifies all5120 source blocks, cached vectors and counts.

Two exact copies of the completed plain64 parent, fresh Adam,512 updates each:

- `blocked`: one code stored but masked; original query positions remain1..64.
- `code`: same data, positions, targets, batches and schedule; code is visible.

Each update:16 original language segments (micro8) plus16 reconstruction blocks.
Objective .5 language mean CE + .5 reconstruction mean CE. Each of two passes
presents every block once: one exposure uses64 BOS placeholders (no source raw
tokens), the other supplies a strictly causal target prefix. Half the524288
auxiliary target exposures are therefore position-only. This is deliberately
denser supervision than earlier fact-loss pilots, while retaining language data.

Original model, legacy coordinates, one stored position per code, no additional
KV or inference parameters; the64 training predictions are ordinary query slots,
not64 expanded memory features. No decoder features or outputs enter the LM.

FP32, AdamW .01, clip1; cosine3e-5 to3e-6 over512 with first30-update warmup.
Both arms share language seed4701604, auxiliary order seed4701623+epoch. Original
data, models, checkpoints and caches are read-only. Unique checkpoint format and
source/data/parent hash guards prevent accidental original-trainer resume.

## Prespecified evaluation

On parent and both trained arms, measure position-only and causal-prefix
reconstruction on dev/test and a fixed training subset. For each, use correct,
blocked and cyclically shuffled codes with the same queries/positions. Save
token predictions, per-token NLL, per-position/source/content accuracies and
whole-block exactness. Position-only output should follow donor content when
codes are swapped; target-label agreement alone may be guessing. Teacher-prefix
accuracy alone does not establish code reading. There is no decoder-derived
trainable input and no decoding-to-raw-history production path.

Evaluate all7831 original dev segments plus free factual answers on the existing
paired dev/test/train-fit/blocked suites and historical312 cases. Reuse unchanged
parent generation/language outputs only after checking checkpoint and provenance.
Report language damage, abstention and changed-fact paired correctness separately.
A reconstruction gain without factual transfer is not the user-goal success.

Four CPU tests pass: label isolation/two-mode coverage, independent full-gradient
causal/blocked reference, actual microbatch objective, and actual serialized
two-epoch tiny Adam resume. Full-size zero-update GPU preflight passes, unchanged
weights,2.238GiB allocated/2.535GiB reserved with Adam/scratch equivalents. Three
update-body timing samples are not an end-to-end speed result.
