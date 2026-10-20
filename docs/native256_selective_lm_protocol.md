# Selective-reader LM integration gate

This is a new package and zero-update preflight, not an extension of a completed
pilot. Native256, the frozen selective component package, all completed runs,
original corpus and caches are immutable. A subsequent bounded experiment uses
its own prospectively recorded recipe; passing this gate is not a quality claim.

One original FP32 code per64 source tokens occupies one ordinary memory K/V
position at each layer. The final layer stores that original code in its existing
V allocation. Other layers use ordinary projected V. There is no separate retained
code bank, expanded memory attention sequence, or reconstructed raw history.

The controller computes64 temporary scalar addresses per visible code and NULL.
It selects one address, invokes the frozen trained one-position decoder, and mixes
that word distribution with the ordinary tied language head. Copy mass is the
selected address probability; all remaining mass falls back to generation. NULL
selects generation. Training must teach address selection explicitly; no gold
position may be supplied to forward or inference. Source-derived address labels
are permitted only as training targets. This gate has no optimizer updates.

Reader-budget mode has exactly23,209,216 stored reader parameters including the
3,981,312 frozen decoder weights and36,864 controller weights, and uses24619 FFN
rows across21 layers. Frozen external encoder adds9,426,432 as in prior budget
accounting. Optional system-budget mode counts that actual encoder too and uses
6208 FFN rows. These are architecture allocations, not demonstrated equal quality.
Core is initialized fresh seed47. Frozen decoder is pretrained; report this.
The user confirmed reader-only23M accounting, encoder separate, on2026-10-17;
reader-budget mode is the selected architecture. System mode is a fixture option.

Checks before training: CPU packed/public/isolation/all-parameter gradients,
serialized optimizer parity, incremental-vs-packed outputs, prefill and token
steps, raw64 rollover with BOS excluded from the compressed block, exact original
code view in existing V, actual cache allocation, frozen decoder isolation and
actual full-size parameter counts. Run a full-size FP32 GPU zero-update timing
gate with learned/untrained controller outputs (no oracle addresses), include
prefill, decode, rollover encoder, memory allocation and forward/backward; compare
native reader under the same measurement harness. Report exclusions and raw
samples. No quality or end-to-end training speed claim from random weights.
