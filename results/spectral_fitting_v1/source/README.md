# Supervised feature discovery with a small FFN

Hypothesis: a training-label energy statistic can initialize relevant input
directions, allowing a trainable rank-32 input projection to support a much
smaller nonlinear feature bank. This follows H103's measured subspace signal.

Architecture: `x -> A(384,32) -> U(32,128) -> GELU or cubic -> D(128,384)`.
All matrices train; hidden/output biases are included. Total: 66,048 parameters.
The raw/bounded moment initializer is fitted to training data only. Random
initialization and parameter-matched spectral dense models are explicit controls.

This is a research candidate using established factorization, spectral estimation
and activations. No novelty is claimed. Cubic derivatives are unbounded; the
rank constraint may discard useful directions; preprocessing costs data passes
and matrix work. Ordinary one-hot language labels provide no label-energy signal.

Shared training/evaluation lives in `src/core/function_fitting.py`. Frozen recipe:
[H104 plan](../../../research/spectral_fitting_plan.md).

Status: **REJECTED AT THIS BUDGET; scoped cubic lead retained.** All 408 fits and
816 independently reconstructed checkpoint scores are complete. Bounded cubic
beats both spectral full controls on aggregate and each task mean, with 94.41%
fewer parameters. It fails the cubic-versus-tiny-control and inference-cost gates.
The GELU version also fails full-model quality and cubic learning. Neither is
registered as a maintained model or allocated automatic language training.

Preprocessing raises the cubic candidate's peak tensor allocation to 30.77 MiB,
giving 25.02% / 22.68% savings versus full GELU / SwiGLU. Small spectral controls
have the same pipeline peak. [Results](../../../research/spectral_fitting_results.md).

`model.py`, `qualification.py`, `study.py`, the shared trainer and the plan remain
frozen to their protocol hashes. `audit.py`, `analyze.py` and `plot.py` are separate
post-training verification/reporting tools. Plotting reads audited JSON without
importing Torch. Runtime failures and recovery are retained in the experiment.
