# Dense FFN controls

Four variants are registered: `gelu`, `swiglu`, `gelu_narrow` and `swiglu_narrow`.
Full models measure retained quality; narrow models test whether the proposed
structure improves on simply using fewer ordinary hidden units.

For the main width-384, eight-layer comparison, full GELU uses hidden width 1536,
full SwiGLU uses 1024, and calibrated narrow SwiGLU uses 304. The full FFNs contain
9,437,184 weights; narrow SwiGLU contains 2,801,664, exactly matching plain
BlockShuffle. [Equations and calibration](model.md).

Use the full and calibrated-narrow recipes in [configs](../../configs/README.md).
Keep corpus, data order, steps, precision, tuning allocation and non-FFN model
settings matched. The narrower baseline's explicit initialization and learning-rate
calibration are part of the comparison.

The ReLU/SiLU-only registry entries are retired. The pre-cleanup implementation
and all original results remain in the [archive](../../research/archive/README.md).
