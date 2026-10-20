# After selective word access: the unresolved address problem

The completed selective512 pilot must not be extended automatically. Native256,
original corpora/caches and completed components remain immutable. The new actual
23M reader/cache is integrated, not a fixture. Frozen one-address decoding works,
including answer IDs held from QA, but same-template counterfactual tests show
essentially no entity-conditioned selection. This is underfitting even on actual
positive training units; it does not prove23M capacity impossible or convergence.

Any new experiment first needs the completed language/results audits. Keep the
64:1 GLOBAL attention/KV ratio and reader-only23M budget, encoder separate. Do not
substitute word-rendering/roundtrip demonstrations for full QA or reasoning.

Two distinct hypotheses are available, neither implemented here:

1. **Teach an entity anchor before the value offset.** Existing address logits
   use a linear selected-unpack representation and a learned offset. Source-derived
   supervision can first point the unshifted score at the queried name's BPE
   positions, then teach the existing offset/value address. This changes only
   training, with no new inference parameters, attention positions or KV. It
   targets the factor currently missing, rather than merely increasing answer CE.
   A trained-name-anchor test, held names, both questions/both memories/both record
   orders and full language checks are needed. Do not assume this will fix binding.

2. **Test explicitly structured lossless token codes in a NEW branch.** A4096-ID
   vocabulary needs12 bits per token. Four base8 digits can represent each ID;
   assigning four FP32 coordinates to each of64 source positions fits one256-vector.
   Positions are then explicit coordinate ranges. Selecting one position recovers
   one ID arithmetically, without decoding64 words or adding memory attention/KV
   positions. Query-to-source lexical matching can use these canonical keys,
   rather than relearning the codec's nonlinear word inverse. The old encoder is
   unchanged; this would be a different, parameter-free packing scheme requiring
   new LM training and explicit query-routing design.

   This is64:1 **attention-position/KV compression**, not compression versus a
   compact token-ID byte stream:256 FP32 numbers use1024 bytes, while64 uint16 IDs
   use128 bytes. The useful comparison is transformer K/V states, not ID storage.
   Do not call this a learned semantic summary or assume a roundtrip means QA.
   A full vocabulary/BOS/rollover exactness proof and actual cache check precede
   any training; then matched fresh controls, held entities/values/relations,
   broad reasoning/language and total speed must be measured. Raw query tokens
   may be used only from the existing recent64 window, never from old history.

Decoding all64 nonlinear word features for address scoring is another possible
linear-in-history compute tradeoff, but it must not quietly become a64-position
global memory sequence or a retained expanded feature/KV bank. Its total compute
could outweigh saved attention; benchmark before considering training.

## More efficient structured-code candidate (unimplemented)

Four digits per ID are not necessary in FP32. For vocabulary4096, one scalar
`(token_id-2048)/2048` exactly represents each ID. A NEW256-coordinate format
could therefore reserve64 coordinates for the64 literal IDs and192 coordinates
for a projection of the old encoder's semantic code. This is a positional
residual path inside the EXISTING vector, not an extra ID/KV bank. It retains
one attention position/code, but trades64 semantic dimensions for exact literal
access. A train-only PCA192 or fixed orthogonal projection needs measurement;
do not assume language quality survives this tradeoff.

The original code must occupy an existing unmodified FP32 V slot, as in the
integrated cache. The reader can compare recent-query BPE spans against the64
literal coordinates directly, then decode only a selected word ID arithmetically.
A learned recent-raw64 span-start/length selector is still required. Full span
and boundary checks matter: matching only a name's first BPE or a prefix such as
Ana within Anastasia is insufficient. Multiple fields/mentions require relation
selection too. Gold query/entity/source spans may be training targets, never
inference inputs. No hardcoded answer or entity parser may replace the model.

This format is incompatible with the old frozen word decoder and old caches;
it needs a new namespace, format guard, fresh LM and matched controls. Removing
the3.98M frozen word decoder frees reader budget for the backbone/controller,
but no quality/speed result exists. Literal coordinate views/scalar scores are
temporary and must not expand into64 global memory attention/KV positions.
Token-ID byte compression is still not claimed. This candidate is not implemented
or roundtrip-verified; it remains a possible next architecture if the current
name-anchor curriculum exposes an unresolved vocabulary/position limitation.
