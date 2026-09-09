# H052: Longer-duration factor-decay control

**Product decay FAILS the frozen promotion rule; no further product-decay tuning is earned by this cell.** This is one seed, not a robust improvement or a
new optimizer. The earlier three-seed parameter-decay failure remains.

## Controlled endpoint

The [frozen plan](duration_decay_plan.md) changes only the existing
TrainConfig.ffn_decay_mode from parameter to product. The model remains plain
SwiGLU BlockShuffle: d=384, eight layers, hidden 2048, G=8, context 128, batch 16,
peak LR 0.0012, 320-step warmup/cosine, native BF16 and gate recomputation.
All initial weights, LR multipliers and non-FFN treatments are unchanged.

| Recipe | Final validation NLL | FFN weights | Peak MiB | Clipped steps |
|---|---:|---:|---:|---:|
| Full SwiGLU | 4.105417 | 9,437,184 | 702.19 | 9.41% |
| Full GELU | 4.127881 | 9,437,184 | 656.62 | 27.53% |
| Calibrated narrow | 4.153048 | 2,801,664 | 516.28 | 11.41% |
| BlockShuffle: parameter decay | 4.145726 | 2,801,664 | 714.12 | 30.47% |
| BlockShuffle: product decay | 4.152418 | 2,801,664 | 714.12 | 42.41% |

Each row uses seed 17 and 3,200 steps / 6,553,600 sampled tokens. The four H050
references are retained; only the product-decay row adds training tokens.
Every endpoint scores all 322,688 validation targets; the official test remains
unscored. FFN reduction is 70.3125%; both BlockShuffle models have 9,099,648 total
parameters. Inference work and activation are unchanged. Cross-session training
throughput is recorded in raw metrics but does not support a paired speed claim.

| Product-decay comparison | Relative NLL cost |
|---|---:|
| vs Full SwiGLU | +1.1448% |
| vs Calibrated narrow | -0.0152% |
| vs BlockShuffle: parameter decay | +0.1614% |
| vs Full GELU | +0.5944% |

Positive NLL cost is worse. Local quality/memory gates: **FAIL**.
The separately required >=0.2% NLL improvement over parameter-decay BlockShuffle:
**FAIL**. Both conditions are required before more seeds.
The raw [decision](../results/duration_decay_v1/result.json) retains each gate.

## Trajectory and scope

| Step | Parameter-decay NLL | Product-decay NLL |
|---|---:|---:|
| 1 | 8.390344 | 8.390341 |
| 800 | 4.887391 | 4.872540 |
| 1600 | 4.531492 | 4.502681 |
| 2400 | 4.266545 | 4.256237 |
| 3200 | 4.145726 | 4.152418 |

Product decay's final 800-step NLL change is
-2.4392%; its late-plateau diagnostic
**FAILS**. This operational check cannot prove convergence.
Initialization NLL and the first pre-update training loss match the old run within
1e-7. Step-1 validation follows one update, so it need not match after decay changes.
All completed diagnostics and checkpoint weights are finite; no numerical failure
occurred. A one-seed trajectory cannot establish a general mechanism.

The zero-gradient/zero-Adam-moment decay identity explains the intervention,
not actual trained norms or complete updates. Up/gate factors change decay .1
to .0125; the down factors change to .0125/.00234375. LR multipliers stay (4,4)
and (4,64/3). Non-FFN parameters and full/narrow reference treatments are equal.
The [CPU diagnosis](duration_optimizer_geometry.md) did not observe Frobenius
collapse in the old model. H013's smaller TinyStories product-decay negative
result is retained in the [earlier plan](product_decay_plan.md).

## Reproduction and retained evidence

Plan SHA256: `a0732d1e25718b6427174f1c1cc2ee61bbef2c93d1de246dcdec6df080b55e5d`.
Result SHA256: `e13fee7cd233fb86315c00391aacfb49f4483f89ab964c3f23e0c4f22ed2e0bc`.
The [preflight](../results/duration_decay_v1/preflight.json),
[protocol](../results/duration_decay_v1/protocol.json) and
[worker qualification](../results/duration_decay_v1/qualification.json) preserve
all configurations, actual optimizer groups, source and data hashes. Run archives,
checkpoint, full history, diagnostics and phase logs are retained. The verifier
checks every reference artifact, initial loss, schedule, parameter/FLOP count,
final CUDA sampler state and the complete source archive. The original trainer,
optimizer implementation and all 31 models are unchanged.

The durable launcher uses UV with compile/data extras, runs one GPU worker and
stops on any unclassified process failure. Exact commands are in
[the launcher](../results/verification/duration_decay_launcher_v1.py).
No failed or partial trial is assigned an endpoint NLL.
