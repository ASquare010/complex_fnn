# Frozen compact FFN selection

Selected research recipe: **curve-wide**. This choice is frozen before independent
seed509, 8000-update validation and one-time test evaluation. It is not a claim
that independent confirmation is already complete.

Parent: curve-parent. Strongest compact control: compact-swiglu-kernel-h384. Strongest full control: full-swiglu.
The longer comparison contains 4 distinct recipes on both corpora.
The existing 200-step warmup and other optimizer settings stay unchanged.

Selection uses the registered two-corpus quality ranking among research recipes
that retain at least 70% FFN compression and lower measured memory than both full
baselines. Variations must improve both parent corpus means without greater than
5% memory/time regression; parents remain eligible. Three seeds alone do not
establish statistical significance. Dense controls remain reported even if stronger.

[Paired language analysis](compact_refinement_confirmation_analysis.md).
[Selection and resource eligibility](../records/compact-refinement-final-selection.json).

Resource limits pass on the recorded point estimates: Curve-Wide is 3.8–4.0%
slower than its parent at approximately the same allocated memory. It uses about
28% less allocated memory than full SwiGLU. WikiText full-baseline timings varied
by more than 2x, so these results do not establish a reliable speedup. All rounds
remain in the evidence; none were filtered out.
[Timing ranges and limitations](compact_refinement_resource_analysis.md).
