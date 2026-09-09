# Dense FFN controls

Four variants are registered: `gelu`, `swiglu`, `gelu_narrow` and `swiglu_narrow`.
Full models measure retained quality; narrow models test whether the proposed
structure improves on simply using fewer ordinary hidden units.

For the main width-384, eight-layer comparison, full GELU uses hidden width 1536,
full SwiGLU uses 1024, and calibrated narrow SwiGLU uses 304. The full FFNs contain
9,437,184 weights; narrow SwiGLU contains 2,801,664, exactly matching plain
BlockShuffle.

Use the full and calibrated-narrow recipes in [configs](../../configs/README.md).
Keep corpus, data order, steps, precision, tuning allocation and non-FFN model
settings matched. The narrower baseline's explicit initialization and learning-rate
calibration are part of the comparison.

The ReLU/SiLU-only registry entries are retired. The pre-cleanup implementation
and all original results remain in the [archive](../../research/archive/README.md).

## Equations and calibration

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
