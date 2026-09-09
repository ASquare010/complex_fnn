# Bezier activation FFNs

The folder now contains both the earlier cubic residual and the requested
single quadratic equation. Shared experiment code stays in `src/core`.

## Quadratic equation and configurations

`quadratic.py` implements `B(t)=2*(1-t)*t*c1+t**2*c2`, with `t=sigmoid(z)`.
`QuadraticBezierActivation` learns two bounded controls per channel group.

- **Direct quadratic:** replace the activation with B. Initial controls (-1,2)
  match SiLU's value, slope and curvature at zero. Two projection matrices.
- **Three-branch products:** split the hidden features into three small projected
  networks; each branch adds a learned multiple of the other two branches'
  product. The final projection mixes their outputs. Zero initial mixing
  coefficients give exactly the unmixed direct quadratic.
- **SwiGLU correction:** add a zero-initialized B to SiLU before the ordinary
  value gate. One shared curve per layer or eight curve groups. This starts
  exactly at calibrated SwiGLU, including BF16 outputs/common-weight gradients.

Parameters per layer: direct `2*d*h+2*G`; mixed `2*d*h+2*G+3`, G=3;
SwiGLU correction `3*d*h+2*G`. The tested configurations retain over 70% FFN
weight reduction. They add pointwise operations and saved tensors.

The direct curve has |B|<=2.5 and |B'|<=2; mixed features have magnitude<=5.625
and feature-Jacobian norm<=7. The SwiGLU correction has magnitude<=.5 and
slope magnitude<=.5. These are scoped mathematical bounds, not guarantees
of trained network stability: sigmoid tails can vanish and matrix norms can
grow. The [plan](../../../../quadratic_bezier_plan.md) gives the derivations,
initialization, nonlinear-interaction limits and primary-source context.

## Quadratic evidence and verdict

**REJECTED for longer-budget promotion in the tested settings.** Five quadratic
screens, matched controls, 15 function trials, exact derivative/equivalence
tests and an actual tiny overfit are retained. The shared residual improves
200-step NLL by only .095% over calibrated SwiGLU. The calibrated direct curve
scores 4.145671 versus equally calibrated GELU 4.100158. Branch products help
the selected function fits but worsen LM NLL. No quadratic meets its frozen
promotion criteria; no architectural novelty is established.

See [results and figures](../../../../quadratic_bezier_results.md), including
measured memory, saturation and the stronger conventional-control follow-up.

```powershell
uv run python -m src.core.cli train --config configs/quadratic_direct_screen.json
uv run python -m src.core.cli train --config configs/quadratic_mixed_screen.json
uv run python -m src.core.cli train --config configs/quadratic_residual_shared_screen.json
uv run python -m src.core.cli train --config configs/quadratic_residual_grouped_screen.json
uv run python -m src.core.quadratic_report
```

# Earlier cubic residual experiment

## Hypothesis and inspiration
A narrow FFN can learn more useful channel responses with two bounded curve
coefficients per group. Inspired by Bézier-Y geometry, learned activation
functions and Group KAN; this is not a claim of a new architecture family.

## Architecture
x → Linear(d,h) → grouped GELU + cubic residual → Linear(h,d).
For each group: t=sigmoid(z), delta_j=.5*tanh(theta_j), and
f(z)=GELU(z)+3t(1-t)[(1-t)delta_1+t*delta_2]. Theta starts at zero.
Fixed X controls 1/3,2/3 give X(t)=t; there is no numerical inversion.
Shared uses G=1; grouped uses G=8. No gating or extra projection matrices.

## Parameters and proofs
P=2dh+2G. At d=h=192,G=8: 73744 weights/block, 74.9946% below the
294912-weight full reference. Whole Transformer: 1672960 vs 2557632 (34.59% less).
Only 16 extra weights/block relative to narrow GELU, a .022% mismatch.
The residual magnitude is <=.375 and its input derivative <=.75.
See [proofs](../../../../theory.md); no full-network gradient guarantee.

## Expected advantages
Exact GELU initialization with immediately live control gradients; preserved
tails; cheap sharing; bounded deformation independent of control drift.

## Expected risks
FP32 elementwise operations and casts add GPU kernels. BF16 output rounding may
hide tiny early deformations. Cubics have limited shape complexity and tanh
controls can saturate. A richer activation cannot recover discarded directions.

## Experiment results and verdict
REJECTED for promotion under the three-seed 200-step screen. Shared NLL 4.2164,
grouped 4.2221, narrow GELU 4.2160, full GELU 4.0846. Shared curves have no
credible benefit; grouped curves lose to narrow GELU in all three seeds.
See the [completed first report](../../../../first_screen_report.md).
The dense controls passed correctness/overfit/CUDA checks before this candidate
was implemented. Keep all learned curves and checkpoints. Broader learned
activation families remain untested, but this branch has not earned a sweep.
