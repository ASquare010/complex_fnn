# H060 additive block/low-rank qualification

**PASS for local numerical and synthetic resource qualification at H060.**
The subsequent [H061 language screen](additive_block_lowrank_screen_results.md)
failed quality, and the tested recipe is now archived. At this qualification stage,
language quality was still untested. The candidate earned a separately frozen
balanced language screen. It is not a breakthrough or verified new primitive.

The actual model uses2,801,664 FFN weights and9,099,648 total weights at d384/L8,
h1024,G8,r48:70.3125% fewer FFN weights. Each projection adds a block-diagonal
path to a global low-rank path. The [frozen plan](additive_block_lowrank_qualification_plan.md)
specifies one initialization and an independently derived update calibration.

## Results

Fifteen isolated GPU workers complete20 synthetic updates each: five recipes,
seeds17/29/43, batch16/context128/native BF16. All steps, weights and optimizer
moments are finite. Memory is identical across the three seeds for each recipe.
Step time is the median of per-worker medians after10 warmup updates.

| Recipe | FFN weights | Peak MiB | Step ms |
|---|---:|---:|---:|
| full_swiglu | 9,437,184 | 677.157 | 57.294 |
| full_gelu | 9,437,184 | 636.157 | 52.274 |
| calibrated_narrow | 2,801,664 | 490.125 | 54.367 |
| blockshuffle | 2,801,664 | 687.813 | 77.597 |
| additive | 2,801,664 | 591.438 | 78.814 |

Candidate memory ratios are0.8734/full SwiGLU,0.9297/full GELU,
0.8599/BlockShuffle and1.2067/narrow. Both110% full-control allowances pass.
This saves14.01% allocation versus BlockShuffle, while using20.67% more than narrow.
The candidate is slower than the dense controls in these short sequential timing
windows and approximately as slow as BlockShuffle. Timing is descriptive, with
only10 measured updates and no rotating-order throughput experiment. No language,
autoregressive, compiled or general training-speed advantage follows.

The synthetic task has614,400 total target exposures and300 optimizer updates.
It uses no corpus data or validation/test scoring. Its losses and clipping are
retained in the [worker histories](../results/additive_qualification_v1/workers/),
not treated as a quality leaderboard. Each worker saves its checkpoint and moments.

## Numerical evidence

The17 isolated tests pass independent dense FP64 forward and all factor/input
gradients, gradcheck and gradgradcheck; actual counts at widths24/96/192/384;
constructive quadratic forward at24/96 and analytic Jacobian/Hessian checks at24;
three-seed component energies; the256-draw independent-perturbation calibration;
native CUDA BF16 output/gradient gates; and eager/native gate recomputation for
initial and changed weights. These are the frozen tolerances, not a proof that
all floating-point inputs or whole-network gradients are well conditioned.

The [complete integrated suite](../results/additive_qualification_v1/process_integration_tests.json)
passes **110 tests in25.19s**. Actual optimizer membership verifies the declared
multipliers: up/value local2,left1.25,right2.04124; down local2,left4.08248,right2.5.
The underlying update calculation assumes independent isotropic perturbations,
not actual correlated Adam directions. It establishes neither Adam invariance
nor improved convergence.

All twelve retained CPU FP32/CUDA BF16 signatures remain exactly unchanged:
initial weights, outputs, losses, gradients, diagnostics, groups and two updates
for each of the six previous variants. Common non-FFN initial weights and synthetic
token hashes match across all five full-model recipes within each seed.

The first three witness Hessians are I,diag(1,...,d),ones. The construction escapes
the specified old square-mixer headwise obstruction. H056 already demonstrates
why representability is insufficient evidence of competitive language learning.

## Reproduce and next decision

[Protocol](../results/additive_qualification_v1/protocol.json),
[result](../results/additive_qualification_v1/result.json),
[source snapshot](../results/additive_qualification_v1/source.zip), and
[pre-integration snapshot](../results/additive_qualification_v1/pre_integration_source.zip)
retain the executed state. All19 recorded processes, including local tests,
finish on their first attempts; no cells are retried. The source remains unchanged
through every recorded process. Historical results and checkpoints are preserved.

Next: a balanced three-rate200-step WikiText screen with the previously qualified
full GELU/SwiGLU, calibrated narrow and BlockShuffle controls. Freeze selection,
quality/memory gates and all source/data checks before scoring. A local pass must
still earn longer training, independent seeds, scaling and broader evidence.
Sparse-plus-low-rank weights are established prior art; see the
[primary implementation audit](additive_block_lowrank_qualification_plan.md#primary-implementation-audit).
