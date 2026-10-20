# Reading compressed codes: evidence and next hypotheses

This note complements the position-varied pilot. It is not evidence that the
23M/two-epoch/64:1 objective has been achieved and does not authorize extending a
finished pilot. Native256 and prior checkpoints remain unchanged.

## Verified update: the trained pointwise decoder is sufficient

The broad readout experiment and selective-reader follow-up changed the next
action. The codec has ONE decoder block. After selecting one position's feature,
that block can operate with just one key, so attention reduces to its value/output
linear maps. Fusing them and keeping the trained FFN/norm/word head matches the
unfused pointwise decoder on every dev/test prediction. Non-special word recovery
is32,728/32,728 dev and32,735/32,736 test; oracle fact values2,048/2,048, with zero/
shuffled controls0/2,048. Independent source/decoder/guard verification passed.

Thus full64-position decoder attention is unnecessary for these word-recovery
tests. The nonlinear trained readout was missing from the failed unpack-only
combination. A new~395K student trained for two small-data passes reaches only
42.59% test word accuracy and fails all held-label words; relearning that inverse
is not justified when the trained pointwise map already works. These observations
do not establish learned question-to-address selection or LM parity.

The next safe action is actual LM/cache integration and address-supervised
question retrieval, not more oracle word benchmarks or extension of the completed
student. Frozen components are in `token_memory_64_selective`; integration must
use a NEW sibling package because that package's source inventory is frozen in
the export. All readout weights must count inside the chosen23M budget.

## New oracle-position component check

CPU diagnosis `dump/native256-selected-unpack-diagnosis-20261017` and independent
recomputation passed. With the correct fact-token position supplied as an ORACLE,
selecting just that position's125 rows of the trained unpack matrix gives the
same256-dimensional pre-decoder feature as the full first unpack. This selected
map uses64,000 multiply-accumulates rather than4,096,000 for all64 positions;
these counts exclude address prediction, normalization, the word head and all
decoder blocks. The map has2,080,000 stored parameters. This is algebra/component
evidence, not a measured complete-reader speed or KV claim.

Passing those features straight to the existing post-decoder norm/word head gets
**0/2,048 oracle value tokens correct**; the full decoder gets2,048/2,048. The
independent script derived positions from source text and recomputed predictions
through the full first unpack, with no selected-weight implementation reuse.
Consequently, copying the unpack map alone is not an immediate retrieval fix.
This does NOT prove its features lack information or that a trained local readout
cannot use them: that head was trained after the decoder's contextual blocks.
There is no trained address selector or new readout in this check, and gold
positions must never be counted as reader performance.

A possible separate architecture probe is an address-conditioned selective
reader with a TRAINED bridge from these features to the language-model states,
plus direct address supervision derived from source facts during training only.
It would require held-out address/value/binding tests, parameter reallocation
for the2.08M unpack map, and an implemented/benchmarked cache layout. No extra
raw-history or decoded-token cache is permitted. Whether the selector can work,
whether one selected feature suffices, and whether the complete path is faster
remain unproven; do not implement a full decoder expansion as a shortcut.

## What the current evidence separates

The original codec decoder reconstructs every block in the selective-reading
suite, while the reader fails shifted facts and novel words. Encoded content is
recoverable by that decoder on those examples; successful reconstruction does
not establish access by the language model. Fixed-layout eight-word success is
real but does not establish entity binding or general copying. The position
curriculum tests whether varied supervision alone repairs the layout failure.

Current packing is a learned global map of64 position features into256 numbers.
Its trained decoder has separate learned unpacking weights for each position.
There is no established shift-equivariance of the latent code. Thus moving a fact
can change how the reader must recover it even when the full token word bag is
unchanged. This is an implementation-based hypothesis, not proof that all shifted
facts are unrecoverable or that encoder retraining is required.

## Primary research, with limits

[Compressive Transformers](https://arxiv.org/html/1911.05507) train compressed
hidden memories with an auxiliary loss that matches attention outputs from old
and compressed memories. Their experiments favor this over reconstructing the
entire old representation. The relevant idea is preserving information the task
can read, rather than merely information a separate decoder can reconstruct.
This paper is not evidence for our target: it uses per-layer hidden memories,
retains ordinary memory, tests compression rates2–4 in the cited comparison,
and uses much larger training budgets. Our unrelated frozen text codec and
one-slot-per64 block do not inherit those results. Task-conditioned readout
distillation would need a new controlled experiment and a measured inference
cost; no such architecture is implemented here.

[Pointer-generator networks](https://aclanthology.org/P17-1099/) combine generation
with source-word copying to improve reproduction of details. Their source words
are directly available, unlike our opaque compressed code. Adding a copy head
alone cannot supply identities that this reader has not recovered. Our unseen-word
candidate-ranking failure also argues against assuming output-vocabulary bias is
the only problem. Any future copy mechanism must get its information from the
same one code per64 tokens, without an extra raw-history or token-ID cache, and
must be tested on unseen values and counterfactual entity swaps. The paper does
not establish64:1 compression,23M parameters or two-epoch parity.

For intuition only: attention over one memory key by itself returns its fixed
value for every query. In the full model, raw-token attention, multiple heads,
layers and nonlinearities can still implement query-dependent code reading.
Therefore the single-key observation motivates an explicit conditioned-reader
hypothesis; it is not a capacity impossibility proof. Any new reader must retain
the attention-position/KV budget, fit the parameter budget, and pass a speed
preflight before training. Repeating the retired growing-vector experiment is
not justified by this argument.

## Practical next decisions

If the new curriculum generalizes only to trained positions, distinguish a
finite set of learned layouts from continuous position transfer before adding
more architecture. If position transfer improves but two-record selection still
fails, train query-to-record selection with identical-word-bag swaps and both
questions, while preserving language replay. If novel words remain poor, broaden
the supervised value space and test code reading separately from output priors.
These are hypotheses and gates, not automatic extensions or quality claims.

For frozen-encoder full training, final256-dimensional code caches can avoid
re-reading64x256 prefix features and recomputing packing. The current matched
pilot deliberately retains its control's exact packing path. A future fast path
needs numerical/logit/gradient parity and measured throughput, not a speed claim
from array-size arithmetic. Do not alter a live guarded trainer to add it.

A fair final reference is still missing: the retained raw-only transformer has
23,077,376 parameters and an older dataset/recipe; historical61.013% belongs to
raw512 plus older compressed memory. Final parity needs a genuinely matched
fresh all-raw reference, the same corpus/exposures/objectives, two original epochs,
broad factual/reasoning evaluation and full language metrics. Diagnostic512
continuations cannot replace that comparison.
