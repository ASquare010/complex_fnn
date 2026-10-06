# Selected position compressor: results

The 5,000-update model has 4,147,200 parameters. The exported encoder alone has
3,225,856 parameters. It uses four encoder blocks, one parallel decoder block,
Curve-Wide FFNs and one 256-feature vector per eight input tokens. Exact length
metadata is included; the decoder receives no source tokens or correct prefixes.

| Evaluation | Exact reconstruction | Token accuracy |
| --- | ---: | ---: |
| Full original development set | 1,000/1,000 | 100% |
| Fresh natural-language confirmation | 1,000/1,000 | 100% |
| Fresh Unicode/formatting confirmation | 256/256 | 100% |
| Independent repeated-pattern sequences | 511/512 | 99.9569% |
| Independent uniform-token sequences | 256/256 | 100% |

Fresh corpus NLL: 0.004085; zero token and character edits. Documents are from
beyond the original preparation range, with exact duplicates excluded, one
32–256-token paragraph per document. The tokenizer and dataset revision are fixed.
These bounded results support near-lossless recovery, not universal exactness.
On 64 development examples, correct memory recovers 64/64; zeroed/shuffled memory
recovers 0/64. The repeated-pattern failure remains recorded.

## Size and resources

The last 1,000 training updates took 223.47 seconds and measured 241.58 MiB peak
allocated GPU memory. The encoder export is 13,026,887 bytes including its tokenizer
and metadata in the original artifact; the training checkpoint includes optimizer state.

Fresh corpus: 96,245 tokens -> 12,471 vectors -> 3,192,576 numerical features.
That is about 7.72 times fewer positions/features than width-256 input embeddings.
BF16 vectors and lengths occupy 6,393,152 bytes, versus 342,453 UTF-8 bytes or
384,980 int32 token-ID bytes. This is not raw-text storage compression.

## Use and cleanup

[Inference notebook](../../../notebooks/paragraph_compressor_inference.ipynb).
Current source-compatible artifacts: `dump/position-pattern-5000-clean-v1/last.pt`
and `encoder.pt`. The cleanup only moved shared code and removed superseded
implementations. Model identity, encoded vectors and outputs match exactly; old
encoder exports remain compatible. Original checkpoints and protocols are preserved.
Historical source files are in ignored `dump/step2-cleanup-v1/`.

Training history, including failed baselines, remains in
[Step 2 results](../../../docs/step_2_results.md). Evidence:
[training](../../../records/position-pattern-5000-v1.json),
[fresh corpus](../../../records/position-fresh-confirmation-v1.json),
[Unicode confirmation](../../../records/position-confirmation-5000-v1.json),
[final verification](../../../records/position-final-verification-v1.json),
[cleanup parity](../../../records/step2-cleanup-v1.json).

Compatibility with another LLM, semantic concept discovery and universal lossless
recovery are not established. Long text uses independent 256-token windows.

## Higher-compression study

The unchanged architecture supports larger spans. A bounded transfer study found
**32 tokens per vector** to be the strongest tested passing setting: 1,000/1,000
exact development paragraphs, 256/256 uniform sequences, 512/512 repeated-pattern
sequences, 256/256 Unicode/formatting strings, and 1,000/1,000 new corpus-confirmation
paragraphs. Full model: 7,292,928 parameters; exported encoder: 4,798,720.

Span 16 also passed all development suites. Span 64 failed at the same 2,000-update
transfer budget (31/1,000 exact paragraphs); it is not a usable replacement under
this protocol. Larger span maps increase parameter count, and the parent already
had 5,000 updates. These are transfer results, not matched-total-compute superiority.

The inference notebook keeps the original 8-token default and now offers 16/32
through `TOKENS_PER_VECTOR`. Encoder-only exports and saved-vector recovery for
both candidates passed; notebook workflows were executed. See the
[full compression protocol/results](../../../docs/latent_memory_compression.md)
and [reconstruction table](../../../docs/step_2_results.md). No whole-LLM speedup,
reasoning over these codes or semantic retrieval has been demonstrated.
