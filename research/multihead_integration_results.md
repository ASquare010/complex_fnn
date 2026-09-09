# Multi-head comparator: integration and GPU qualification

**Both initialization controls pass the frozen pipeline checks.** The architecture is now registered and ready for a separately frozen training screen. No language-model validation result, convergence or reproduced flash-kernel performance is claimed.

[Frozen H045 plan](multihead_integration_plan.md), [raw decision](../results/multihead_integration_v1/result.json), [source-recorded launcher](../results/verification/multihead_integration_launcher_v1.json), and [model equations](archive/retired/src/multihead_ffn/model.md).

## Initial output scale

The training-independent Gaussian diagnostic uses 4096 rows at width 384, seed 73, first FFN at layer 0 / seed 17. Orthogonal mixers and explicit private-weight scales give the calibrated variant a full-SwiGLU RMS ratio of 1.119251, within the frozen [0.5,2.0] range. The reference initialization has a much smaller initial output scale at these tiny head dimensions.

| Architecture | First-FFN output RMS |
|---|---:|
| swiglu | 0.0129212365 |
| multihead_swiglu | 0.0000018297 |
| multihead_swiglu_calibrated | 0.0144621069 |

The variance argument assumes isotropic inputs and approximately independent centered subnetwork outputs; it retains the epsilon mass factor. Orthogonal mixing supplies an exact norm identity, not an isometric nonlinear FFN or a whole-network gradient guarantee. The observed ratio is approximate calibration, not trained superiority.

## Two fresh GPU workers

Each model uses width 384/layers 8, batch 16 / context 128 BF16 and ten constant-rate AdamW updates at 0.0006 on the SAME frozen training batch. There are 2048 batch tokens and 20,480 repeated token exposures per worker. No validation or test targets were scored.

| Variant | Initial FP32/BF16 logit relative L2 | Peak allocated MiB | Fixed-batch loss, step1 | Fixed-batch loss, step10 | Clip fraction |
|---|---:|---:|---:|---:|---:|
| multihead_swiglu | 0.00625969 | 832.10 | 8.382187 | 6.391807 | 100% |
| multihead_swiglu_calibrated | 0.00807750 | 832.10 | 8.406331 | 6.343028 | 100% |

Losses are measured before each listed update, on the repeatedly trained batch. Their decrease is a pipeline check, not a generalization comparison or evidence that one initialization trains better. All recorded losses, gradients and routing diagnostics are finite; every layer's router has nonzero finite gradients at every update. All ten steps clip in both cases. No global nonvanishing-gradient claim follows.

At this configuration both variants have 2,807,808 FFN weights and 9,105,792 total, a 70.2474% FFN reduction. Native execution materializes intermediate tensors. The 832.10 MiB peak is a pipeline measurement without a paired full-model memory control; it does not reproduce the paper's SRAM kernel or qualify a memory/speed advantage.

## Verification and next step

All 28 earlier registered variants retain exact tiny CPU parameters, logits, loss, gradients and matrix-work counts. Twelve focused comparator tests pass. The full suite passes 213 tests in the [recorded unchanged continuation](../results/verification/multihead_tests_v2.json); its first attempt suffered a native integer divide-by-zero during Matplotlib import in collection. Both records are retained; root cause is unresolved. The two GPU workers finish successfully without retries.

Both source archives and checkpoint/config counts, fixed-batch hashes, ten-step histories and all finite/routing checks are verified. The registered reference and calibrated variants share the same forward implementation. Only initialization differs. The calibrated initialization is our local small-dimension control, not a claimed reproduction of the paper's exact training recipe.

The pass earns a separately frozen equal-budget learning-rate screen against the full, calibrated narrow and stronger plain BlockShuffle controls. It does not earn automatic 800-step promotion. Keep the corrected learnable-activation result, all failed branches and the broad convergence/novelty/stability requirements intact.

```powershell
uv run --extra compile --extra data python -m src.core.multihead_integration_report
```
