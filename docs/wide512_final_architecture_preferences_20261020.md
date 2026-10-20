# User preferences and pending comparisons — 20 October 2026

The user wants the final memory architecture to remove the explicit memory-width
projection while retaining one global attention/KV position per 64 old tokens.
Preferred candidates are a native width256 reader with width256 token embeddings,
or matching encoder and reader widths. The suggested width500 encoder plus 12
position channels is a proposal, not an implemented or verified improvement.
Do not change the completed wide512 models/checkpoints or retrain them to test this.

The current adapter is applied at memory prefill/append, rather than on every
cached decode step. Its 256-by-512 matrix costs 131,072 multiply-accumulates per
new block, or 2,048 per old token amortized over a 64-token block. This arithmetic
does not measure its fraction of request time. Removing it has no demonstrated
quality/speed benefit yet. A separate future model/ablation is required.

The current local reader scores all accessible old token IDs, not only the latest
64-token block. For N old tokens it has N/64 global positions and N literal
candidates. Fixed-weight inference vocabulary-table reuse is a candidate exact
optimization; block routing/top-k reading is a quality-changing hypothesis.
Neither is implemented here or asserted to preserve quality without verification.

A concrete low-risk candidate is the query selector's fixed-weight table:
`select_query` normalizes the4096-by512 embedding and projects it to8 vocabulary
scores every inference call. That projection is16,777,216 multiply-accumulates;
the resulting8-by4096 FP32 table occupies128KiB. During inference it depends only
on fixed model weights, so precomputation can be tested without changing KV/history
contents. This is an arithmetic opportunity, not a measured latency win. Numerical
logit/output parity and an isolated timing comparison are required before adoption.

One targeted QUALITY hypothesis concerns the local reader's return value rather
than new encoder channels. In `token_memory_64_lexical/reader.py:read`, source
probabilities are aggregated by token ID before the embedding readout. Two source
occurrences of the same token have identical value embeddings, so this returned
feature does not explicitly retain which occurrence was read. The scoring path
does use causal predecessor context, source offsets/recency and query sentence
bias, and the global path also supplies information; this is not a proof that all
position information is absent from the model.

A future controlled test could add a small weighted source-position feature to
the SAME local readout/state, deriving token offsets from the existing payload
order. Compute its weighted aggregate directly, without a persistent per-token
feature bank or new global/KV positions. This adds some local arithmetic and
possibly a few parameters (reallocate within the budget). It may help multi-piece
value continuation but is UNVERIFIED; new held-out cases are required for later
model selection. No such implementation, training, or zero-update model ablation
has been performed in this task.

Token order in the current literal payload is implicit: token i occupies the low
four bits of vector components 4i through 4i+3 (zero based). No token is dropped
to insert position channels; only low semantic float precision bits are replaced.
The format requires 256 FP32 carrier numbers for 64 16-bit payload slots. Reducing
the carrier size or changing precision requires a new format, not a simple cast.

Both wide512 arms are now saved at 48,762 updates / cursor(epoch2,batch0).
Preliminary original-dev document reaggregation: compressed64 1,158,824/1,859,388
(62.32287183%), CE1.726689592; raw1 1,184,606/1,859,388 (63.70945709%),
CE1.652498788. Difference -1.38658526 percentage points. Raw1 has the same custom
copy/query reader and uncompressed history; it is not a vanilla Transformer.
Full training/source/Adam audits and free-fact/binding/efficiency evaluation are
pending at the time of this note. Teacher-forced accuracy is not free fact recall.

Subsequent progress in the same task: both complete training audits passed. Comp
main GPU evaluation and independent aggregate/public CPU verification passed;
60 deterministic cases replay exactly across the saved groups. Its final repeated
dev pass differs from training validation by one argmax hit(1,158,823vs1,158,824),
with CE difference about1.4e-9; use canonical evaluation documents for final paired
comparison. Comp notebook independently verified:66/256 full answers/allfour7/64;
158/256 prefix-assisted/allfour18/64;blocked0/64 in each form. Transfer independently
verified:0/224 original and0/224 prefix-assisted exact. Raw comparison, novel-literal
evaluation and request timings are still pending; these comp results are not a win.

The user agrees to wait for raw1 completion before the matched factual comparison.
No new training run is authorized by this comparison. Preserve the original encoder,
all old/native256 runs, source/cache/corpus files, and the retired growing families.

The longer-term approximately100M model beating a10B model is a research target,
not a verified capability. Keeping FP32 is an allowed design choice, but its memory
and compute costs still apply at100M. Current data cannot establish broad parity.

Final comparison and efficiency verification now PASS. Full results and limits:
`docs/wide512_completed_comparison_20261020.md`. Comp67/285 original exact facts
versus raw27/285; notebook full66/256/allfour7/64 versus39/256/allfour0/64.
Raw has better language63.709511% versus62.322818% and prefix transfer56/224
versus0/224. Both have0/1664 on the larger literal suite, including known controls.
Comp request medians are0.10–3.68% slower across12 measured cases: no latency win.
At2048 old tokens/batch1, KV3.094MiB comp versus72.003MiB raw, including reserves.
At2048/batch4, peak allocated590.62MiB versus744.96MiB. All study workers finished;
no current model/code/encoder/checkpoint/cache/corpus was modified or retrained.
