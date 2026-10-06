# Step 1: compact FFN replacement

**Execution update:** the results below describe the historical optimized
implementation. The active model now uses ordinary PyTorch for readability,
with unchanged parameters and verified checkpoint compatibility. Its short GPU
check used 553 MiB versus dense's 561 MiB and ran slower; it does not retain the
historical 28% memory saving. No new language training was performed for this
revision. [Current execution report](../src/models/channel_curve_transformer/result.md).

The practical milestone is achieved in our measured four-layer Transformers:
substantially fewer FFN parameters and lower allocated training memory, with
language quality close to the full model. This does not establish a universal
LLM improvement, a reliable speedup, or a novel architectural family.

**Selected model: Curve-Wide**, implemented as `channel_curve_transformer` with
`self_curve_wide`. It retains embeddings, full causal attention, normalization,
residual connections and the output head. Two full-width projections surround
three learned self-gated responses per feature.

Compared with full SwiGLU, the selected model has 74.9% fewer FFN parameters,
42.9% fewer total parameters and about 28% less allocated training memory.

The bounded refinement tested nearby variations of Signed Linear, Self-Curve
and Basis Readout. It completed 12 development runs, 60 paired-seed runs,
60 resource measurements, and eight independent longer runs with held-out tests.
Curve-Wide improved over its parent in all six paired-seed comparisons.

In the independent 8000-update, seed509 comparison, Curve-Wide beats its parent
and compact SwiGLU on validation and test in both corpora. Its held-out NLL is
2.30% higher than full SwiGLU on TinyStories and 1.38% lower on WikiText.
The shorter three-seed comparison favored full SwiGLU on both datasets.
Timing variability and an overnight sleep interruption prevent a reliable
speedup claim. Test results did not select or tune the recipe.

This refinement is complete. Keep Curve-Wide and the dense baseline active.
The new [Step 2 encoder draft](step_2_context_encoder.md) starts a separate
representation-compression discussion; it does not reopen this FFN comparison.
The [shortlist archive](../ffn_experiments/README.md) retains the strongest
alternatives for an explicitly requested future study. Do not expand into
attention replacement, memory systems or kernel optimization under this step.

[Final explanation and results](compact_refinement_summary.md) ·
[Independent evaluation](compact_refinement_independent_result.md) ·
[Leaderboards](leaderboard.md) · [Long-term charter](research_goal.md)

Raw records and old reports are local, ignored artifacts. Current documentation
retains the measured conclusions; links to raw evidence require the local files.
