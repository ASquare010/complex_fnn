# BlockShuffle: ablations, memory and execution

All seven numbered architecture folders were renamed and active imports, tests and links updated. All 15 original variants preserved their deterministic parameter and output signatures. See the [migration record](naming_migration.md). The new structured family lives in src/blockshuffle_ffn.

## Quality improvement and ablations

The [locked three-seed comparison](blockshuffle_results.md) is the main quality evidence. This table retains the seed-17 branches that selected the recipe; each uses 800 steps. These branches have one seed and are not independently validated architecture wins.

| Method | Unique FFN weights | FFN reduction | Validation NLL |
|---|---:|---:|---:|
| Full SwiGLU, hidden 512 | 1,179,648 | 0.0000% | 3.065045 |
| Narrow SwiGLU, hidden 128 | 294,912 | 75.0000% | 3.166492 |
| Shared SwiGLU, hidden 608 | 350,208 | 70.3125% | 3.137058 |
| BlockShuffle 832, parameter decay | 294,912 | 75.0000% | 3.115322 |
| BlockShuffle 832, product decay | 294,912 | 75.0000% | 3.118254 |
| Narrow SwiGLU, hidden 152 | 350,208 | 70.3125% | 3.149475 |
| BlockShuffle 1024, ordinary gate | 350,208 | 70.3125% | 3.098982 |
| BlockShuffle 1024, recomputed gate | 350,208 | 70.3125% | 3.098956 |

At 200 steps, the same hidden-832 structured architecture and orthogonal initialization reach NLL 5.017039 with uniform learning rates and 4.177205 with factor-aware rates. Appropriate optimization is essential in this setting. This is an explicit optimizer treatment, not an identical-optimizer architecture comparison.

The represented-product weight-decay correction gives NLL 3.118254 versus 3.115322 for parameter decay. It does not earn promotion on this seed; this small difference does not establish statistical superiority. Spending the remaining parameter allowance on hidden width 1,024 improves NLL to 3.098982, earning the predeclared replication.

## Training memory and numerical behavior

The ordinary hidden-1,024 model peaks at 300,809,216 allocated bytes, 12.36% above full SwiGLU. Recomputing only the SiLU gate expression reduces the measured 800-step peak to 285,604,864 bytes: 15,204,352 bytes saved, leaving 6.68% overhead against the seed-17 full reference. No projection GEMMs are recomputed. This passes the 10% memory-increase gate, but does not make training faster than dense SwiGLU.

The non-reentrant checkpoint profile had a small NLL divergence at 20 steps. Native backward matched the ordinary 20-step profile exactly, including all saved weights. At 800 steps it is not bitwise identical: maximum logged NLL difference is 0.0001613, final difference is -0.00002618, and the largest parameter difference is 0.01549. All tensors are finite. The source of long-run finite-precision divergence is not isolated; interpret the native recipe as its own measured training recipe. Independent first/second derivative and CPU/BF16 backward tests pass. See [memory protocol and amendment](gate_recompute_plan.md) and [raw long-run check](../results/verification/gate_native_800.json).

## Eager serving

All modes use the same final seed-17 checkpoint, except the matching full reference. Each mode is evaluated on all 32,768 targets. The table reports six rotating timing rounds as throughput ratios to the dense reference. Single-model memory is measured separately during forward evaluation, including NLL computation, with no optimizer states.

| Execution mode | NLL | Extra cache bytes | Peak allocated MiB | Median relative throughput | Range |
|---|---:|---:|---:|---:|---|
| factorized | 3.098956 | 0 | 102.35 | 0.657 | 0.525-0.789 |
| dense_cache_bf16 | 3.098916 | 4,718,592 | 106.85 | 1.001 | 0.768-1.194 |
| packed_factors_bf16 | 3.098956 | 466,944 | 102.79 | 0.780 | 0.678-0.888 |
| dense_reference | 3.065045 | 0 | 107.76 | 1.000 | 1.000-1.000 |

Packed gate factors batch the up/value operations, reducing six grouped matrix launches to four per FFN while retaining 700,416 FFN matrix FLOPs per token. They add 466,944 cache bytes and tensor movement. The eager median improves from 0.657 to 0.780 times dense, still below the 0.8 acceptance floor. The raw audit also includes FP32 dense caching; no mode is hidden.

Dense caching retains the learned factors and stores extra dense matrices. Its 4,718,592 FFN matrix FLOPs per token are twice the full reference, versus 700,416 for factorized/packed execution. It therefore trades away the arithmetic reduction. BF16 cached NLL changes by only -0.0000395 here. Separate-process absolute throughput varied sharply; use the raw rounds and do not select the fastest isolated run.

## CUDA graph execution check

The [frozen graph experiment](cuda_graph_plan.md) applies the same execution treatment to every mode. Fresh-input outputs match eager exactly for three independent batches, and all held-out validation NLLs match their eager mode. Eight rotating timing rounds include the device input copy. The measurements are fixed-shape full-sequence forwards.

| Execution mode | Peak allocated MiB | Median throughput versus dense graph | Range |
|---|---:|---:|---|
| factorized | 131.99 | 0.699 | 0.687-0.776 |
| packed_factors_bf16 | 132.45 | 0.679 | 0.620-0.725 |
| dense_cache_bf16 | 136.51 | 0.832 | 0.783-0.863 |
| dense_reference | 135.17 | 1.000 | 1.000-1.000 |

These corrected measurements replace the original graph memory comparison, which accumulated cuBLAS workspaces from fresh warmup streams. See the [correction and probe](graph_memory_correction.md). Original raw artifacts remain intact. Training peaks and eager measurements are unaffected.

With equally applied CUDA graph replay, factorized relative throughput is 0.699 times full SwiGLU and packed factors reach 0.679 times. They still fail the 0.8 speed floor. Dense caching reaches median 0.832 times while costing twice the full reference's FFN matrix FLOPs. No training or autoregressive-generation speed claim follows.

Raw [corrected graph audit](../results/serving_blockshuffle_graph_v2.json) and [eager audit](../results/serving_blockshuffle_final.json) include every timing, source hash, cache cost and checkpoint hash.

## Projection conditioning and gradient flow

The CPU FP32 audit compares initialization and final checkpoint on one fixed held-out batch. All audited projections retain their maximum possible numerical rank. Full-rank projections do not prove well-conditioned FFN or Transformer Jacobians.

| Final seed-17 model | Projection condition-number range | Per-layer FFN backward gain range | All captured values finite |
|---|---:|---:|---|
| Full SwiGLU | 4.58-10.82 | 0.0534-0.1017 | True |
| Narrow SwiGLU 152 | 19.81-46.22 | 0.0448-0.0792 | True |
| BlockShuffle 1024 | 2.97-646.30 | 0.0730-0.1361 | True |

Backward gain is the ratio of input/output activation-gradient RMS for this batch loss, recorded separately at each invocation. It is not an extremal Jacobian singular value. BlockShuffle begins with flat projection spectra but reaches condition number 646 after training. This directly limits any claim that the initialization solves deep-network conditioning. Three-seed clipping fractions and pre-clip gradient histories remain in the run records.

## Decision

Keep BlockShuffle as a stronger compressed reference and retain the failed ablations. Mean paired NLL improves by 1.861% against narrow SwiGLU, with the same weights and factorized matrix FLOPs. It remains 1.333% worse than full SwiGLU, and factorized serving fails the speed criterion. No gold result, verified novelty or general superiority is established. Equal tuning, a broader corpus and trained scale tests remain necessary before promotion beyond this setting.

The [source-backed structured protocol](blockshuffle_plan.md) cites Wei et al. for BlockShuffle and Qiu et al. for factor learning-rate scaling. The work here establishes a controlled small-model implementation and measured limitations, not a priority claim.


## Stronger conventional reference added later
The [width-calibration study](width_calibration_results.md) improves narrow SwiGLU to three-seed mean NLL 3.134833. BlockShuffle remains better by 0.855% mean paired NLL against that control, versus 1.861% against the original uniform-rate narrow recipe. Preserve both comparisons and do not present the earlier gain as a gain over a fully tuned narrow model.
