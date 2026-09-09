# Model notes

See [architecture and evidence](README.md).

## Factorization, budget and initialization

For input width n and output width m, a projection uses
`W = inverse(P_out) B2 P_mid B1`, with intermediate width `k=min(n,m)`.
B1/B2 are block-diagonal with G groups and P denotes a fixed permutation.
The weight count is `k*(n+m)/G`. For all three SwiGLU projections at h>=d,
the layer uses `3*d*(d+h)/G` weights and twice that many logical forward matrix
FLOPs per token. At d=384,h=2048,G=8 this is 350,208 weights/layer.
It exactly matches the h=304 narrow SwiGLU control's parameter count and matrix
work. Wider intermediate activations and kernel overhead remain real costs.

Semi-orthogonal factors produce flat nonzero singular values at initialization:
product gain is `0.02*sqrt(max(n,m))`, with `1/sqrt(2*layers)` for down projections.
One factor is square orthogonal, so its product with the rectangular semi-orthogonal
factor and permutations retains this spectrum. Training is unconstrained; this
initial statement is not a trained conditioning or whole-network stability bound.

The retained recipe assigns each factor LR multiplier
`projection_input_width/(2*factor_input_width)` and parameter-level AdamW decay.
Native gate recomputation repeats SiLU/product backward arithmetic without
repeating projection GEMMs. These are explicit parts of the measured recipe,
not an assertion that parameter count alone causes faster learning.

This factorization adapts established structured-matrix ideas. Read the
[derivation and prior art](../../research/blockshuffle_plan.md) and the
[scoped inference result](../../research/fused_execution_results.md).

## Longer-training limit

The [3,200-step replication](../../research/long_duration_replication_results.md)
fails the full target: mean NLL 4.147443 is 1.256% above full SwiGLU, and seed 43
also loses to calibrated narrow. All twelve compared runs are still improving
at the end. The shorter-budget success below does not establish convergence
or robust longer-training superiority.

## Stronger WikiText control after the activation study

At width 384 / layers 8,800 steps and peak LR 0.0012, the plain recipe reaches
4.759029 +/- .010381 mean validation NLL across seeds 17/29/43. It uses
2,801,664 FFN weights (70.3125% fewer than full), passes both full quality/memory
limits and beats calibrated narrow in every seed. The
[corrected report](../../research/affine_rate_replication_results.md) retains
all comparisons. [Configuration](../../configs/wikitext2_blockshuffle_stronger_800.json).

The affine activation's same-rate gain is only 0.101% with 2/3 wins, below its
material-promotion threshold. The earlier selected-rate advantage must not
be assigned to activation learning alone. This stronger optimizer setting does
not establish convergence, a novel architecture or WikiText serving speed;
the existing fused speed result uses an earlier TinyStories checkpoint.

## Isolated factor-decay outcome

The [longer product-decay control](../../research/duration_decay_results.md)
changes only existing factor decay coefficients. Seed-17 final NLL 4.152418 is
0.161% worse than parameter decay, and 1.145% above full SwiGLU. It fails the
local quality and material-gain gates; no additional tuning or seeds are earned.
Clipping rises from 30.50% to 42.41%. Fewer accumulated decay-only contractions
do not establish better optimization; all weights and recorded gradients remain
finite. Initial weights, LR multipliers, data and the 3,200-step budget match.
