# H040: frozen correction-only full-training repeat

Written after H038 passes its stricter numerical and memory gates, before this
backend trains a language model. Local global-gradient relative L2 is3.95e-8,
logits and full validation NLL match, allocated peak717.31MiB. These local results
alone do not establish trajectory fidelity. The earlier whole-product backend
failed its200-step repeat and remains a separate retained failure.

Repeat only the selected native rational BlockShuffle recipe: d384/L8/H6,
hidden2048,groups8,B16,context128,vocab4096,seed17,200steps,LR0.0012,BF16,
same pinned WikiText data and all322,688 validation targets. Retain initialization,
factor LR calibration, gate checkpointing, optimizer groups, decay, clipping,
data generator, warmup/cosine schedule and all pointwise casts. Only compile the
rational correction, with native SiLU/product forward and backward. No new
learning-rate selection, shape initialization or dataset.409,600training tokens.

The explicit backend is `inductor_rational_correction`. A zero-token shape warmup
performs forward/backward with no optimizer step and no data RNG consumption;
verify every parameter unchanged, then clear gradients. Record compilation
warmup time/peak separately. Native layer diagnostics temporarily bypass the
compiler and restore it. Validation uses the correction-only compiled execution.

One fresh worker,900-second timeout, fresh fixed run name, no overwrite.
Stop and archive any failure. Verify source/config/data hashes, exact training-loop
AST, initial validation NLL within1e-7, unchanged common optimizer assignments,
all finite recorded diagnostics, final checkpoint step and parameter counts,
and final sampling RNG equality with the native source run.

Acceptance: absolute relative final NLL difference <=0.2% versus the native
rational checkpoint AND all original short-screen parameter/quality/memory gates:
>=70%FFNweight reduction, <=1%NLLincreaseagainst BOTH full controls, beat selected
narrow, >=0.2%better than selected unmodified BlockShuffle, and allocated training
peak <=1.1BOTHfull references. No best-intermediate checkpoint selection.
A pass establishes this one seed/short execution repeat only; it does not prove
new architecture quality, independent replication, convergence or speed.
