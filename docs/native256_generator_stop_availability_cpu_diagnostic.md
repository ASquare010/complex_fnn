# Conditional generator stopping: selected CPU diagnostic

18 October2026. Separate process-local read-only ablation on query V2 step1000.
Existing sources, checkpoints, caches and jobs are unchanged; production updates0.
If any old EOS word has strictly positive copy probability, preserve the original
output exactly. Otherwise use the generator-controlled stopping formula from the
preceding diagnostic. This is a structural fallback; no expected answer, gold
length, record ID or test label is supplied to the model. It is not universal
learned stopping: a numerically underflowed EOS is treated as unavailable.

On the SAME source-selected development subset, original access stays0/28;
prefix-assisted access improves0/28 to15/28. All eight notebook outputs are
identical to the original (4/8 exact), avoiding the preceding unconditional
rule's Remy truncation. The subset is small and does not establish broad quality.

Six CPU public/cached checks passed: first two original and first two prefix
access cases, plus two blocked-prefix controls. Cached outputs match the saved
public outputs, blocked outputs match the old checkpoint, cache bytes and
attribute names stay unchanged, and all model tensors stay unchanged. No GPU,
full-dev or latency proof exists. The diagnostic wrapper repeats the generator
matmul; prospective integrated code could reuse its logits. Do not adopt before
wider development/blocked/counterfactual/language checks and proper GPU gates.

Reports and archived CPU math/cache scripts are under
`dump/native256-generator-stop-availability-cpu-20261018`.
