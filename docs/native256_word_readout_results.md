# Broad oracle-position word readout: verified component results

All four new readouts completed exactly two passes/512 updates. Independent data,
first-update, full training and results audits passed. Original reader/encoder
weights and native256 were not trained or changed. This is a word-decoding
component experiment, not language-model accuracy, free QA or two original
corpus epochs. The full goal remains active and unmet.

Run: `dump/native256-word-readout-pilot-20261017`.

## Training and controls

Use4096 genuine64-token original corpus blocks with one256-dimensional frozen
encoder code each. All code arms receive the correct position as an ORACLE.
Only the raw control sees the source token embedding directly. The codec's
normalized frozen word table is shared by the classifiers; no trainable token
table. The~395K residual MLP is freshly initialized with seed47 in every arm.

Exclude all301 occurrences of eight held BPE IDs from positive training labels:
light/dark/sun/star/book/house/school/road. Each arm receives261,843 labels/pass,
523,686 total, covering3936 supervised IDs. Batch1024/micro256, identical two
permutations, FP32 AdamW.01, clip1, warm10, LR3e-4cos3e-5. The final batch has723
labels. All Adam states reach512; frozen prototypes remain unchanged. Pretrained
codec resources already know the vocabulary, so held labels are not words unseen
by every resource. No decoder-success filtering.

## Held-out word decoding

All following positions are supplied correctly; these are SINGLE WORD-token
classifications, with no EOS generation or question-to-position selection.

| Readout input | Test correct /32,768 | Macro ID accuracy | Held IDs /52 |
|---|---:|---:|---:|
| Trained unpack feature | 13,955 (42.59%) | 6.15% | 0 |
| Unpack feature plus original code | 11,170 (34.09%) | 3.18% | 0 |
| Original code plus sinusoidal position | 1,078 (3.29%) | .037% | 0 |
| Raw source embedding oracle | 32,768 (100%) | 100% | 52 |

There are2884 target IDs in test. The macro metric exposes the frequency skew;
the42.59% result is not broad vocabulary mastery. All trained code students have
zero exact test blocks and zero success on the64 never-supervised-ID occurrences.
The selected-feature student's actual training subset scores3409/8192, similar
to held-out accuracy, indicating substantial underfitting at this budget. Adding
the global code does not help this particular student/optimizer; it does not
prove global code information is irrelevant.

On the prior balanced fact suite, selected-feature student oracle word hits are
4/256 fixed,4/256 shifted8,2/256 shifted16,2/256 spare-after,10/256 spare-before,
6/256 and7/256 two-record orders, and0/256 held answer words. The raw oracle
scores256/256 every group. These are not free-answer or binding scores.

## Important zero-update decoder reference

The original codec has **one decoder block**, not a deep contextual decoder.
Running that block on only the selected feature gives32,728/32,768 dev tokens
and32,735/32,768 test tokens correct. Non-special accuracy is32,728/32,728 dev
and32,735/32,736 test, essentially the full decoder's performance. This changes
the next action: reuse the trained nonlinear decoder instead of learning a new
small word readout from scratch or expanding64 positions into LM attention.

The previous unpack-only/word-head oracle scored0/2048. Its missing nonlinear
transformation matters; the failure does not establish missing encoded words.
The next selective-reader preflight verifies this reference independently and
on all fact values, without training more students.

## Verification and cost

Four CPU tests passed: actual microbatch gradients/frozen prototypes, exact
two-epoch serialized Adam resume, selected/full unpack equivalence and single-key
decoder algebra. GPU preflight gradient error7.45e-9, empty Adam and unchanged
weights; peak.209GiB allocated/.234GiB reserved for this component workload.
All saved token predictions, loss vectors, held-label masks, source-derived fact
positions and optimizer/order counts were audited. Independent CPU recomputation
checked1024 test samples per arm/mode. All1524 protected old hashes are unchanged.

Component-only median latency before training, batch1/16: selected MLP path
.947/.956ms, original one-position decoder1.209/1.233ms, full64 decoder2.953/2.976ms.
The one-position timing starts from its selected feature. These are not complete
LLM, address-selection, cache-ingestion or end-to-end speed measurements. The
unpack map has2,080,000 frozen parameters, prototypes1,048,576 values and student
394,753 trainable parameters; eventual model integration must pay for them.

See [protocol](native256_word_readout_protocol.md) and
[verified selective-reader follow-up](native256_selective_reader_results.md).
No completed run should be automatically extended.
