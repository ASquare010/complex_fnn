# Additive block/low-rank SwiGLU

**REJECTED at the H061 language-screen budget; archived.**
[H061 results](../../../../additive_block_lowrank_screen_results.md) fail every
quality gate despite passing memory. [H060 qualification](../../../../additive_block_lowrank_qualification_results.md) remains a scoped numerical result. This is an adaptation of a known matrix
family, not a verified novel primitive.

Each gate, value and down projection computes `Sx + L(Rx)`: a block-diagonal
local map plus a global low-rank map. There is no dense effective-weight tensor
in the native forward. The nonlinearity remains ordinary SwiGLU.

At model width384, hidden width1024, eight groups and rank48, this uses
350,208 weights per layer. Eight layers have2,801,664 FFN weights and9,099,648 total
model weights: **70.3125% fewer FFN weights**, exactly matching plain BlockShuffle
and calibrated narrow. Hidden width is1024 rather than BlockShuffle's2048.
These counts alone do not prove lower memory, faster training or better quality.

The [model notes](model.md) specify initialization, update calibration and their
limits. [witness.py](witness.py) constructs the stated three-output quadratic in
a disposable model; it is not training initialization. The17 independent numerical/GPU tests and110-test integrated suite pass.
Three-seed synthetic training allocation is591.438MiB,14.01% below BlockShuffle.
It remains slower than dense controls. The subsequent language screen failed.
No longer training or automatic recipe repair is earned.

Variant: `additive_block_lowrank_swiglu`. The declared optimizer mode is
`--ffn-lr-mode perturbation --ffn-decay-mode parameter`. Native gate recomputation
is supported; other initialization/optimizer recipes have not been qualified.
