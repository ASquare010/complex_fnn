# Shared-embedding lexical reader: component and integration gates

Prospective 18 October 2026. This is a new branch, not modification or revival of
any original/native/growing model. Completed pilots remain immutable. The full
23M-reader/two-original-epoch/64:1 quality goal remains active and unmet.

The word-bit storage gate verified exact source IDs inside existing FP32 values,
without losing semantic dimensions or adding KV entries. The learned codec's
inverse and position-dependent keys are unnecessary for these added lexical IDs.
Test a generic differentiable lexical reader instead of a frozen nonlinear word
decoder or a finite-name classifier. A storage roundtrip is not QA or language.

Two read units (intended after core layers7 and14, one-based) use the current raw
token states as queries. Source keys use shared LM embeddings of each old word
and its eight CAUSAL predecessors, with shared linear context maps and a nonlinear
QUERY map. Factor the context maps into query-to-vocabulary scores and scalar ID
gathers, avoiding dense per-old-word keys/values and repeated source-context MLPs.
Normalized fixed-vocabulary embedding tables remain temporary. A learned
query-dependent recency bias uses actual word offsets, not codec feature slots.
Never use
future neighbors: future blocks can exist in a packed training batch but must not
affect an earlier visible source key. Mask every candidate by physical memory
visibility and validity. A fixed internal NULL handles empty/blocked memory;
NULL is not a persistent KV entry. Query-conditioned source probabilities return
weighted word embeddings through a gated residual and a soft full-vocabulary
copy distribution. The generator remains trainable. No fixed entity/value list,
gold spans/answers or reconstructed raw-history input is allowed at inference.

There are still64 IDs per EXISTING256-number FP32 memory value. Global attention
sees one position per completed64-token block; lexical word candidates are
temporary local computations, not64 global KV entries. The source keys/values and
scalar scores use additional TEMPORARY memory and compute: roughly linear in old
word count per query, not free and not a demonstrated64x compute reduction. No
expanded source feature bank may persist across decode steps. Complete integration
must decode IDs before arithmetic and reinsert metadata into existing projected
V slots at the two read layers. The component alone proves none of this cache
integration or total speed.

Candidate allocation is559616 reader weights: two source norms(512), causal
key map2304->128(294912), query FFN128->256->128(65536), two query norms(512), two
query maps256->128(65536), two residual output maps256->256(131072), two scalar
gates(512), two recency maps(512), generator/copy logits256->2(512). All maps have
no biases. Remove1093 FFN hidden units from the native total32467 to31374; each
frees512 weights. Actual
Core+component parameter count must be independently checked as23209216. No
frozen decoder or extra word embedding table is counted outside that budget.
Encoder remains separate and frozen. Extra source IDs are deliberately carried
by the bit payload: do not call this recovery by the original encoder, or byte
compression relative to token IDs.

First component gates: actual budget, literal-ID versus packed-ID parity, source
role/start bit independence, query-conditioned outputs, finite normalized mixtures
and gradients with no/blocked memory, future-code/padding invariance, and direct
selected-word copy gradients. This is not a trained component or useful QA result.
Then an actual new LM and incremental cache must establish packed/public/cached
forward and gradient equivalence, genuine64-token rollover, unchanged KV entry
count/bytes, no retained expanded word bank, and complete forward/prefill/decode/
rollover speed against matched native and true1:1 controls BEFORE training.

The separate name-gradient-isolation512 arm is still evaluated independently;
it does not train this component or authorize adopting it. Any later pilot needs
new source formats/caches/guards and broad held names/values/positions/bindings/
missing/multiword/long-history/reasoning/language checks. Final comparison still
requires identical data/labels and fresh two-original-epoch budgets, plus a genuine
matched1:1 model. No quality/adoption claim from component tests.
