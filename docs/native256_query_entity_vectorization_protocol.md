# Metadata vectorization hypothesis, separate from the running model

18 October2026. Corpus trainer uses the immutable V2 implementation. No running
source/model/recipe/optimizer is changed. This new tool-only runtime experiment
tests mathematically equivalent integer word-key and role-header computations.
It is not a new trained architecture or a substitute for factual evaluation.

Replace the eight per-piece word-key loops with an integer prefix/scatter scan,
and per-role/per-token header loops with a single broadcasted comparison. Preserve
every Words field exactly, including invalid/partial/over-eight-piece words and
causal terminating-boundary semantics; preserve scope masks/start/valid flags
exactly. Float query scoring, gates, core, heads, weights and cached arrays are
unchanged. No old-word self-attention, new global position or persistent history
bank. A fixed role-ID table may be reused globally by tokenizer pattern/device;
record its bytes separately. It must contain no user/history/query state.

Use an explicit process-local context manager that restores the two metadata
functions, including on exceptions. File bytes and all original/checkpoint source
guards remain unchanged. This runtime override must be disclosed and bound by
its own source hashes whenever executed; never silently present it as the
checkpoint's original implementation. CPU randomized/edge integer equality,
active full-model forward/all-gradient/cache/causal equality and optimizer-state
mapping must pass first. GPU parity/timing can run only AFTER the sole training
worker is terminal. No inference-speed claim from CPU tests and no model adoption
without full free-answer/language evidence and relevant output parity.
