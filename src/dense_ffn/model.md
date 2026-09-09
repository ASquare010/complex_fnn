# Dense equations and parameter controls

All projections are bias-free. With input/output width d and hidden width h:

- GELU: `f(x) = D GELU(Ux)`, using `2*d*h` weights.
- SwiGLU: `f(x) = D [SiLU(Ux) * Vx]`, using `3*d*h` weights.

Full hidden widths are `4*d` for GELU and `8*d/3` for SwiGLU, each giving `8*d^2`
weights per FFN. Narrow widths are explicitly configurable; the main narrow
SwiGLU control uses h=304 at d=384.

Name-local seeded Gaussian initialization preserves common non-FFN tensors across
variants. Projection standard deviation is 0.02; down projections additionally
use `1/sqrt(2*layers)`. The calibrated narrow recipe multiplies initial down
weights by `sqrt(reference_hidden/actual_hidden)` and their learning rate by
`reference_hidden/actual_hidden`. Its configured decay adjustment preserves the
stated shrinkage. These are explicit optimizer controls, not a convergence proof.

See [width calibration](../../research/width_calibration_results.md) and
[the longer comparison](../../research/long_duration_replication_results.md).
