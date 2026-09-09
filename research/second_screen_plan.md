# Second screen — frozen before its runs, 2026-09-06

The first cubic settings failed promotion. Test whether retaining nonlinear
expansion with structured weights is more effective than deforming a narrow
activation. This is inspired by existing structured FFNs, not a novelty claim.

Keep micro_v1 data, common decoder, optimizer, precision, seeds and 200-step
budget. First run seed 17 only. Compare:
- narrow dense SwiGLU: h=128, P=3dh=73728 per block;
- grouped wide GELU: h=768, G=4, P=2dh/G=73728;
- grouped wide SwiGLU: h=512, G=4, P=3dh/G=73728.
All are exactly 75% fewer FFN weights than full controls. Existing narrow GELU
is also exactly matched. Attention and embeddings remain identically initialized.

Architecture: each projection is block diagonal; shuffle hidden channels between
activation/gating and the output projection. A group shares input coordinates,
not weights with other groups. Use native batched matrix multiplication.
The permutation is fixed and parameter-free. No extra layer norms, trainable
gates outside SwiGLU, attention changes or custom kernels.

Grouped weights use .02*sqrt(G) initialization so reduced fan-in does not shrink
their preactivation variance by G; output weights also receive the common
1/sqrt(2L) residual scale. This is an approximate variance-matching choice, not
a theorem of identical activation distributions. If the structured model wins,
ablate initialization scale with an equally scaled narrow control before
attributing the gain to structure.

Forward matrix FLOPs/token are 2*P per block, identical to the narrow parameter
controls. Structured wide variants have more nonlinear elementwise work and
potentially worse batched-GEMM utilization; measure real speed and allocated memory.
Cross-group nonlinear interaction is restricted within one FFN, although the
common Transformer mixes channels across successive blocks.

Promotion: NLL within 1% of full controls, or a >=1% relative NLL improvement
over both narrow GELU and narrow SwiGLU without a >20% inference slowdown
against full references. A weaker trend can motivate a targeted ablation,
but not automatic expensive training. No hundreds-of-config search.

Prior art: [Wei et al. 2024](https://arxiv.org/abs/2406.16450),
[official implementation](https://github.com/CLAIRE-Labo/StructuredFFN),
[Monarch](https://arxiv.org/abs/2204.00595). Our direct grouped-FFN sandwich
is not an exact reproduction of their two-factor-per-linear BlockShuffle
parameterization, self-guided training, or modified-attention models.


## Completed seed-17 screen
Narrow SwiGLU NLL 4.41907; structured GELU 4.22408; structured SwiGLU 4.64820.
Full controls were 4.09232/4.07645 and narrow GELU 4.24112.
The .40% structured GELU gain over narrow GELU does not meet the prespecified
1% promotion threshold; grouped SwiGLU is worse. Neither is promoted. These
short-budget settings are rejected, while broader structured methods remain
open. The next tested direction is documented in sharing_screen_plan.md.
