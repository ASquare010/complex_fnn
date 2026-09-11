# H114 audit recovery — numerical replay, unchanged scientific gates

The original 56-case training study completed once. Its post-study audit stopped
at an exact initial-gradient hash assertion inside its first fixture. Input
tokens, targets, weights and loss matched before that assertion. No optimization
was repeated. Preserve `audit.py`, `audit_protocol.json`, `audit_failure.json`,
`audit.log` and exit code 1 unchanged.

A bounded six-backward, zero-update diagnostic on the first saved native-BF16
case found five exact replays and one non-exact replay. The non-exact replay
differs in 64 embedding elements, three attention-norm elements and one QKV
element. Maximum per-tensor relative L2 is 2.061e-6 and maximum absolute element
difference is 1.193e-7. Loss is exact in all six. Subsequent replays in the same
model return to the saved values. The diagnostic changes optimizer presence and
preceding evaluator together; it does not isolate an operator or prove a cause.
Bitwise stability cannot be assumed for this complete BF16 decoder backward.

Before the recovery executes, freeze its code and all original evidence. The
independent check will report every replay error and bitwise-equality flag. It
requires exact loss plus gradient global relative L2 <=1e-5 and every parameter
relative L2 <=1e-4 (the same norm floors as the original plan). This is a
**posthoc audit reproducibility tolerance**, introduced transparently after the
failed exact assertion. It does not replace or relax H114's prospective candidate
gates: <=0.002 global and <=0.02 per tensor versus the native FP32 classifier,
along with the original loss, memory, timing and continuation-NLL gates.

Compute all candidate-versus-reference comparisons from the **original saved
probe gradients**, not whichever replay looks closest. No best-repeat selection.
Each of 56 probes is replayed once in the recovery; all results are kept. Reuse
the original native evaluator, data sampler, Transformer and loss adapter.
Verify 70 native scores and all 2,800 training batches and final Adam states.
Write `audit_recovery.json`; leave the failed original output paths untouched.

The failed audit stopped before its first fixture result was logged, so its
exact backward count is unavailable (between one and four); do not claim an
exact aggregate audit compute count. Training remains exactly 2,800 updates and
2,856 profile backward passes. Diagnostic adds six, recovery adds 56 backward
passes, all without optimizer updates. No scientific training retry occurred.
