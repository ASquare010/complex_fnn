# H081 - BLAST learning screen and factor calibration

**Complete short screen; the broader research goal remains unmet.** Twenty-four
fresh WikiText-2 trials finish, and independent BF16 rescoring exactly reproduces
all 24 final checkpoint losses. One seed and 200 updates do not establish
significance, convergence, broader-data performance or state-of-the-art.

- **BlockShuffle calibrated: FAILS FIXED SCREEN**. Failed gates: nll within one percent full swiglu, nll within one percent full gelu, nll beats narrow swiglu, nll beats narrow gelu.
- **BLAST SwiGLU: FAILS FIXED SCREEN**. Failed gates: nll within one percent full swiglu, nll within one percent full gelu, nll beats narrow swiglu, nll beats narrow gelu.
- **BLAST GELU: FAILS FIXED SCREEN**. Failed gates: nll within one percent full gelu, nll beats narrow swiglu, nll beats narrow gelu.

## Selected final results

Each form receives the same three peak learning rates and training budget. Select
the lowest final validation NLL; lower is better. These short-budget scores are
not directly comparable to H078's 3,200-update scores.

| Model | Selected LR | Final NLL | Peak MiB | Mean update ms | Clip % |
|---|---:|---:|---:|---:|---:|
| Full SwiGLU | 0.0012 | 5.907694 | 384.56 | 102.25 | 10.5 |
| Full GELU | 0.0012 | 5.879278 | 393.25 | 87.15 | 12.0 |
| Narrow SwiGLU | 0.0012 | 5.912505 | 279.45 | 93.05 | 8.5 |
| Narrow GELU | 0.0006 | 5.902164 | 279.02 | 91.39 | 17.5 |
| BlockShuffle legacy | 0.0006 | 5.970604 | 279.77 | 126.89 | 16.0 |
| BlockShuffle calibrated | 0.0012 | 6.073010 | 279.77 | 139.54 | 7.5 |
| BLAST SwiGLU | 0.0006 | 6.165411 | 281.20 | 153.90 | 38.5 |
| BLAST GELU | 0.0006 | 5.951733 | 280.95 | 125.34 | 11.0 |

All compressed forms have 2,801,664 FFN / 9,099,648 total weights, versus
9,437,184 / 15,735,168 for both full controls: 70.3125% FFN and 42.1700% total
reduction. BLAST hidden widths are 1,984 (SwiGLU) and 3,200 (GELU), rank 48 and eight groups.
Lower weight counts are not a measured latency claim. Times are sequential
laptop-GPU measurements with no repeat-based confidence interval.

## Quality comparisons and frozen gates

Relative NLL changes; negative favors the candidate. Both full controls permit
at most +1%; both narrows must be beaten strictly. Actual peak must be within 10%
of both selected full peaks. The selected NLL run also supplies its resource gate.

| Candidate | vs full SwiGLU | vs full GELU | vs narrow SwiGLU | vs narrow GELU | vs legacy BlockShuffle |
|---|---:|---:|---:|---:|---:|
| BlockShuffle calibrated | +2.798% | +3.295% | +2.715% | +2.895% | +1.715% |
| BLAST SwiGLU | +4.362% | +4.867% | +4.277% | +4.460% | +3.263% |
| BLAST GELU | +0.745% | +1.232% | +0.663% | +0.840% | -0.316% |

Upper-rate boundary selections: Full SwiGLU, Full GELU, Narrow SwiGLU, BlockShuffle calibrated. No rate extension or repair is made
after these outcomes. A passing screen earns a separate longer, then multi-seed
comparison; it does not qualify a new active model. Failed fixed recipes close.

## Optimizer ablation and interpretation

| LR | Calibrated BlockShuffle vs legacy | BLAST SwiGLU vs calibrated BlockShuffle | BLAST GELU vs calibrated BlockShuffle |
|---|---:|---:|---:|
| 0.0003 | +2.687% | +0.641% | -2.638% |
| 0.0006 | +1.995% | +1.243% | -2.266% |
| 0.0012 | +1.454% | +1.639% | -1.536% |

The calibrated BlockShuffle starts byte-identical to legacy, with equal initial
validation NLL and first training loss at every rate. Their difference is a
complete optimizer recipe: initial-RMS multipliers plus product decay replace
fan-in multipliers plus parameter decay. These data do not isolate the two parts.

BLAST is [published prior art](https://arxiv.org/html/2410.21262v1). Its qualified
operator and layer-local orthogonal initialization are reused without changing
the contraction. The shared dense constructor runs before replacing temporary
FFNs, preventing its generic initializer from overwriting BLAST factors.
Common non-FFN tensors match all controls exactly. BLAST comparisons also change
factor structure and hidden width; they are not pure activation ablations.

For K factors, initial RMS rho_j and dense reference scale sigma, fixed
s_j=rho_j/(K*sigma) makes a hypothetical unit-RMS normalized Adam direction have
relative factor update eta/(K*sigma). Actual directions, clipping and Jacobians
need not match. With decay lambda/(K*s_j), zero-gradient/zero-moment shrinkage
of the represented map is (1-eta*lambda/K)^K: it matches dense only to first order.
The [frozen derivation and limits](blast_learning_screen_plan.md) are local
calculations, not nonvanishing-gradient or convergence guarantees.

## Data, execution and verification

WikiText-2 raw-v1, the unchanged train-only 4,096-token BPE cache: 3,083,650 train tokens,
322,802 validation tokens. Batch 16 x 128 gives 2,048 targets/update; 200 updates give
409,600 targets/run. Totals: 4,800 language updates / 9,830,400 training targets.
Shared seven-pass validation presents 54,211,584 development targets; independent
rescoring adds 7,744,512 targets, zero updates. The official test split is unscored.

All nine preflight checks pass in 25.03 seconds, including full-model CPU FP32 and
CUDA BF16 eager/checkpoint fidelity: 24 temporary updates / 24,960 target presentations,
1080 exact gradient tensor pairs across the paired steps.
They never initialize a language run. The coordinator took 23.59 minutes,
including preflight, validation, startup, artifact checks and checkpoint writing.

Native eager BF16/FP32 parameters, TF32 off, four CPU threads, one GPU worker at
a time. Whole-block recomputation except the qualified narrow-GELU block+inner
mode. The unchanged shared trainer handles AdamW, sampling, schedule, loss,
evaluation and clipping. Both narrows retain their established width/decay
calibration. Per-step loss/preclip norm/rate, per-layer gradients/activations and
sampled activation slopes are saved, together with full weights, moments and RNG.

Independent checks verify every checkpoint, actual counts, optimizer membership,
moments/step counters, sampling RNG, source/config/data hashes, histories, gates
and scores. Completed artifacts are fsynced and read back; this does not diagnose
or repair the older H079 corruption. Prior source and evidence remain unchanged.
No scientific retries occurred. No new activation or active folder is added.

A [historical control comparison](../results/blast_learning_screen_v1/control_reproduction.json)
finds exact prior final weights/moments/NLL for 14 of 15 controls. Full GELU at
LR 0.0006 has identical initial NLL but a final NLL difference of -0.0000138162;
its final weights and moments also differ. The cause is not diagnosed. This
cross-run diagnostic is separate from the exact checkpoint rescores. All gates
use fresh H081 controls; no older score is substituted and no run is repeated.

## Every allocated rate

| Model | LR | Final NLL | Peak MiB | Mean update ms |
|---|---:|---:|---:|---:|
| Full SwiGLU | 0.0003 | 6.120134 | 384.56 | 93.91 |
| Full SwiGLU | 0.0006 | 6.005087 | 384.56 | 89.90 |
| Full SwiGLU | 0.0012 | 5.907694 | 384.56 | 102.25 |
| Full GELU | 0.0003 | 6.040670 | 393.25 | 110.44 |
| Full GELU | 0.0006 | 5.901672 | 393.25 | 145.37 |
| Full GELU | 0.0012 | 5.879278 | 393.25 | 87.15 |
| Narrow SwiGLU | 0.0003 | 6.135830 | 279.45 | 91.63 |
| Narrow SwiGLU | 0.0006 | 6.030559 | 279.45 | 91.62 |
| Narrow SwiGLU | 0.0012 | 5.912505 | 279.45 | 93.05 |
| Narrow GELU | 0.0003 | 6.061124 | 279.02 | 89.35 |
| Narrow GELU | 0.0006 | 5.902164 | 279.02 | 91.39 |
| Narrow GELU | 0.0012 | 5.909740 | 279.02 | 86.67 |
| BlockShuffle legacy | 0.0003 | 6.076798 | 279.77 | 125.48 |
| BlockShuffle legacy | 0.0006 | 5.970604 | 279.77 | 126.89 |
| BlockShuffle legacy | 0.0012 | 5.985974 | 279.77 | 126.57 |
| BlockShuffle calibrated | 0.0003 | 6.240110 | 279.77 | 138.31 |
| BlockShuffle calibrated | 0.0006 | 6.089715 | 279.77 | 139.64 |
| BlockShuffle calibrated | 0.0012 | 6.073010 | 279.77 | 139.54 |
| BLAST SwiGLU | 0.0003 | 6.280115 | 281.20 | 154.95 |
| BLAST SwiGLU | 0.0006 | 6.165411 | 281.20 | 153.90 |
| BLAST SwiGLU | 0.0012 | 6.172544 | 281.20 | 153.15 |
| BLAST GELU | 0.0003 | 6.075480 | 280.95 | 124.85 |
| BLAST GELU | 0.0006 | 5.951733 | 280.95 | 125.34 |
| BLAST GELU | 0.0012 | 5.979738 | 280.95 | 125.37 |

[Measurements](../results/blast_learning_screen_v1/result.json),
[independent audit](../results/verification/blast_learning_screen_analysis_v1.json),
[frozen protocol](../results/blast_learning_screen_v1/protocol.json).
