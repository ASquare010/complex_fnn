# Atlas: native256 token memory

The single active compressed-memory model is
[token_memory](src/models/token_memory/README.md): native256 tokens and memory,
23,209,216 trainable reader parameters, raw64 rollover, and one global position
per64 old tokens. It keeps the FP32 low-bit payload and query-conditioned local
reader. No256-to512 memory projection is present.

Use the [PowerShell training command](src/models/token_memory/README.md#train-in-powershell)
for fresh seed47 training on two original unified-v4 epochs. Progress and periodic
full-development evaluation print readable summaries. Training has not been
started by the cleanup.

The frozen [position codec](src/models/position_compressor/README.md), its
branch-sigmoid dependency, original datasets/cache, completed checkpoints and
[raw baseline](src/models/raw_transformer/README.md) are retained. Old compressed
model variants and their dedicated experiment helpers were archived and removed.
[Cleanup record](docs/token_memory_native256_cleanup_20261020.md).

The [completed matched wide512 comparison](docs/wide512_completed_comparison_20261020.md)
used23M readers and two original epochs. Compressed64 had67/285 familiar exact
fact answers versus27/285 for its raw1 control, and23.27x smaller total KV at2,048
old tokens. Language accuracy was62.323% versus63.710%, transfer was weak, and
requests were0.10–3.68% slower. This establishes useful compact storage and some
reading, not perfect recovery or a universal quality/speed win. The new native256
model still needs its own training and evaluation.
