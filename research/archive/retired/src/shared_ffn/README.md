# FFN reuse across existing layers

## Hypothesis and inspiration
Retain full dense nonlinear width and cross-channel interaction while sharing
weights across four successive existing Transformer blocks. This is a serious
parameter-efficiency control inspired by [ALBERT](https://arxiv.org/abs/1909.11942)
and [One Wide Feedforward](https://arxiv.org/abs/2309.01826); it is not new research
merely because it is implemented here.

## Architecture
Block i keeps its own attention and RMSNorms, then invokes F_floor(i/4).
Each F is the ordinary full-width GELU or SwiGLU FFN. No extra forward steps.
Placement lives here; Transformer, trainer and metrics remain common.

## Parameter count and compute
K=ceil(L/4) distinct FFNs. Unique FFN weights=K*(2dh or 3dh).
At L=4,d=192: 294912 FFN weights total versus 1179648 for untied controls.
Whole model: 1672896 versus 2557632. FFN matrix FLOPs remain unchanged.
This saves unique parameters and optimizer states, not per-invocation matrix size.

## Expected advantages and risks
Wide reusable features and full dense mixing; no extra kernel sequence.
Risks: conflicting gradients across layers and insufficient layer specialization.
Shared parameter gradients are sums across uses, not distinct per-layer gradients.
All accounting deduplicates tensors by identity.

## Results and verdict
REJECTED as a gold-target result in this screen; retained as a strong control.
At 800 steps, seed 17: shared GELU NLL 3.22460, shared SwiGLU 3.14017, full
SwiGLU 3.06505. Sharing SwiGLU misses the 1% quality limit by a material margin.
Widening shared SwiGLU to h=608 reaches NLL 3.13706 at 70.3125% FFN reduction,
with higher computation. Per-layer affine gates do not materially close the gap.
See [completed report](../../../../second_screen_report.md). Longer training,
equal tuning and broader data are untested; these are setting-specific decisions.
