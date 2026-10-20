# Lexical LM: implemented, CPU integration verified, untrained

Latest update: the full-size reference/optimized GPU gates and independent
audits now pass. Speed is mixed: about2.9x faster batch4 forward and25.2x smaller
total cache at2496 old words, but about20% slower cached decode than conventional
1:1. See [complete numerical/memory/timing report](native256_lexical_gpu_results.md).
No full-budget quality result exists. The earlier CPU implementation record follows.

18 October 2026. This is concrete progress toward the full goal, **not a quality
result or final model**. Full-size GPU verification, independent cache/math audit,
complete benchmarking, data/trainer and broad quality evaluation are outstanding.
No lexical LM training has begun. Native256 and completed models are unchanged.

New code lives in `src/models/token_memory_64_lexical` and
`src/models/token_memory_64_lexical_lm`, using the separately verified word-bit
storage component. It retains the old learned code's256 semantic coordinates,
with a small FP32 precision tradeoff, and adds exact source IDs in their low bits.
This is deliberately added lexical information, not proof the old encoder is
lossless. The compressed main transformer still receives one position per64 old
tokens and a raw64 window. Word candidates/keys are temporary local computations;
they add compute and activation memory without adding persistent global KV slots.

The decoder uses shared LM word embeddings to score causal nine-word lexical
contexts, rather than relearning the frozen codec's inverse or classifying seven
names. Two query-conditioned read units, after layers7 and14, return weighted
word embeddings through gated residuals. A soft vocabulary pointer mixes with
the ordinary generator; NULL probability returns to generation. Recency uses
actual old-word offsets. No fixed fact template/parser, gold name/span/answer,
finite answer list or extra reconstructed raw-history input is supplied to the
compressed main transformer.

Actual parameter count: **23,209,216, all trainable**.559,616 weights belong to the
lexical reader, and the21-layer/256-wide/4-head core uses FFN total31,374. The
frozen encoder remains separate; no frozen word decoder or new embedding table
is added outside the reader budget. The1:1 control mode has exactly the same
fresh initial tensors but one ordinary embedding/global KV position per old
word. Its local lexical features and read probabilities match the compressed
mode for identical word streams and visibility. Its larger global attention/KV
cost is intentional and must be measured.

Eight component tests and seven actual LM integration tests passed. The LM tests
activate nonzero read/recency/mixture gates, avoiding a vacuous test of disabled
read paths. They check:

- Full model budget, all trainable weights, exact fresh64-vs1 state identity.
- Public versus cached prefill/token outputs, payload recovery from both actual
  read-layer V allocations, exact cache array inventory and byte count.
- Packed windows versus independently isolated causal query gradients for every
  parameter, plus future source-code and actual invalid-raw-key isolation.
- Finite empty/blocked-memory losses, gradients and generator equivalence.
- Actual64-generated-token rollover excluding BOS: one new encoded code for the
  compressed mode,64 ordinary word slots and no encoder call for the1:1 mode.
- Actual microbatch gradients and exact serialized8-step Adam resume.

Metadata is reinserted after V projection into existing allocations at both read
layers. No derived source keys/values/IDs are stored on the cache. Cache attributes
are only K/V arrays, memory/raw validity, current raw64 IDs and scalar cursors.
Old `cache.codes()` access is rejected because the last-layer V is not an original
code bank; callers must explicitly use a read-layer payload view. These are CPU
implementation checks, not yet independent full-size GPU evidence or measured
production efficiency. The bit payload requires FP32 storage and a new source
cache format; original code-cache files cannot be silently treated as word-bit
inputs.

Full-size CPU tests count the actual23M model and compare both initial states.
Output/gradient/cache tests use a two-layer small-FFN core with the actual lexical
reader. This is not full-size numerical verification. The final component
factors context maps into query-to-vocabulary projections and scalar source-ID
gathers, with a nonlinear query map. It forms no dense per-old-word feature bank;
weighted read vectors use vocabulary-mass aggregation. An independent explicit
causal-key calculation matches the factored scores in the eighth component test.
This reduces the history-dependent arithmetic on paper; complete GPU timing and
training-memory measurements remain required, and no speed claim is made.

Commands actually passed:

```powershell
& .\.venv-cuda64\Scripts\python.exe -m unittest discover -s tests -p test_native256_lexical_reader.py -v
& .\.venv-cuda64\Scripts\python.exe -m unittest discover -s tests -p test_native256_lexical_lm.py -v
```

Protocols: [reader component](native256_lexical_reader_protocol.md),
[LM/cache integration and required GPU gates](native256_lexical_lm_protocol.md).
Do not start training until the full-size independent and complete speed/cache
gates pass. Final acceptance remains broad retrieval/reasoning/language parity
under matched fresh two-original-epoch/23M-reader/64:1 constraints, with a strong
genuine uncompressed baseline. The goal remains **ACTIVE / UNMET**.
