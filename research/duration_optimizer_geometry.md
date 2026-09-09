# Duration changes accumulated factor decay, but does not predict trained norms

**Read-only seed-17 diagnosis.** The longer comparison changes duration and the relative schedule. Even at the same peak rate, this changes accumulated weight decay. The [earlier factor-decay derivation](optimizer_rate_geometry.md) applies directly; it is not a new optimizer or an identified cause of the validation ranking change.

The [CPU audit](../results/duration_optimizer_geometry_v1/result.json) verifies both durations' metric/checkpoint hashes, reconstructs initialization and actual optimizer groups, and materializes the represented maps only offline on CPU. It runs zero optimizer updates and scores no new validation targets. Layers 0 and 7 were fixed as the first/last layer probes. All initial map norms agree exactly between durations; BlockShuffle norms also agree with their analytical initializer norm within 1e-6 relative error.

## Decay-only algebra

For a represented map M=B2*P*B1, zero gradients and zero Adam moments give, when decoupled decay is applied,

    M_T = M_0 * product_t product_j(1 - eta_t*s_j*lambda_j).

The factors commute with scalar shrinkage; this is an exact real-arithmetic identity under those assumptions. It is not the actual trained-map norm. The original schedule sums to 0.5238 over 800 steps and 2.0934 over 3,200; the latter is not exactly four times the former because of discrete warmup/cosine endpoints.

| Duration | Full/narrow dense-map multiplier | BlockShuffle up/gate multiplier | BlockShuffle down multiplier |
|---|---:|---:|---:|
| 800 | 0.948966018 | 0.657629695 | 0.264996139 |
| 3,200 | 0.811111990 | 0.187305801 | 0.004953832 |

Dense and width-calibrated narrow down maps have the same decay-only multiplier because the narrow recipe corrects decay for its width-specific rate. BlockShuffle retains parameter decay with factor-rate pairs (4, 4) for up/gate and (4, 64/3) for down; each factor has decay 0.1. These are the frozen treatments, not an after-the-fact change.

## Actual trained maps

| Recipe | Layer-0 down norm / initialization, 800 steps | Same ratio, 3,200 steps |
|---|---:|---:|
| full swiglu | 4.604954 | 7.663904 |
| calibrated narrow | 8.282736 | 13.627787 |
| blockshuffle | 14.128370 | 12.633983 |
| full gelu | 4.278585 | 7.218308 |

Despite its decay-only multiplier falling below 0.005, the trained BlockShuffle layer-0 down map remains 12.634 times its initialization Frobenius norm. The last-layer down ratio increases from 10.483 to 15.096. Thus these two represented maps do not collapse toward zero in Frobenius norm, and the decay-only calculation cannot be reported as observed weight collapse. It also cannot exclude ill-conditioning: Frobenius norm measures total squared matrix magnitude, not the smallest singular value or a gradient lower bound.

BlockShuffle first/last up and gate map norms grow with duration. Its stored FFN output RMS is about 3.120 to 3.102 at layer 0 and 1.272 to 2.324 at layer 7. These are checkpoint diagnostics on the fixed validation batch, not learning-cause estimates. The modules, surrounding learned representations and optimizer trajectories differ.

## What to investigate next

The result makes duration-specific optimization a plausible control to test, not a proven repair. A future matched rate/decay experiment must be separately frozen and retain full/narrow controls. H013 already tested product decay in a smaller setting without improvement; that negative result remains in the [ledger](product_decay_plan.md). Changing effective-map regularization or adding a learnable activation cannot be credited for an improvement until the corresponding controlled experiment succeeds.

H051 continues unchanged. No new activation, trained architecture, measured speedup or novelty claim follows from this CPU diagnosis.
