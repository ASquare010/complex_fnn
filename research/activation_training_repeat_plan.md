# Frozen compiled rational training repeat

H036 passed the separate isolated memory/derivative audit. Before increasing
training duration, repeat the selected rational BlockShuffle recipe from its
original initialization with the explicit `inductor_rational` execution backend.
This is one fresh seed-17, 200-step trajectory, not an independent seed or a new
hyperparameter sweep. No original checkpoint or result is overwritten.

Keep d384/L8/H6/context128/vocabulary4096, hidden2048/groups8, BF16/B16,
peak LR0.0012, original factor-specific optimizer treatment and checkpointed
activation/product. Reuse the frozen WikiText cache and exact sampled token
stream. Final validation covers the same 322,688 targets. The full config is
[here](../configs/wikitext2_blockshuffle_rational_compiled.json).

The shared trainer now has an explicit backend field, defaulting to eager.
Compilation uses the audited product implementation, with precision-cast
emulation and no CUDA graphs. A zero-token forward/backward compiles its
training shape before initial evaluation; it performs no optimizer update,
consumes no sampling RNG and leaves all parameters unchanged. Record warmup
seconds and allocated peak separately. The normal training peak starts after
initial validation/diagnostics, as in the existing protocol. Activation
statistics use the native pointwise reference temporarily, so Python hooks
cannot cause graph breaks; restore compiled execution immediately afterward.

The original architecture screen and isolated audit keep their original
source archives. All prior variants must still pass mathematical/model tests.
The old training loop remains unchanged apart from the optional compilation
setup and its recorded metadata; eager remains the default.

## Required outcome

A useful repeat must satisfy all finite/count/data checks; remain within 1%
relative NLL of both selected full controls; beat selected calibrated narrow;
improve at least 0.2% over selected unmodified BlockShuffle; and use no more
than 1.10 times the training peak of either selected full control. It must also
stay within 0.2% relative NLL of the selected native rational result. Report
any drift, including improvement; do not call one repeat bitwise deterministic.
Initial full-validation NLL must match the native rational initialization.

Run one bounded worker (900 seconds), preserving source, config, checkpoints,
histories, optimizer/RNG state, diagnostics, and failure information. No
additional rates or step counts are chosen after the result. A pass can earn
an independently specified replication/longer-budget cohort. It does not
establish multi-seed, convergence, transfer or novelty claims.
