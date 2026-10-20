# Installed FP32 attention backend check

18 October2026. Read-only component probe on RTX5070, installed
PyTorch2.14.0+cu132/CUDA13.2. FP32, TF32 disabled, no optimizer or model changes.
For lengths128,551 and3008, bool causal masks and padded-stride float masks both
select aten::_scaled_dot_product_efficient_attention and its backward by default.
Forcing EFFICIENT_ATTENTION also succeeds. All tested outputs/gradients are finite.
Thus this probe does not support a slow math-backend fallback hypothesis for the
tested shapes. Do not change the current attention backend based on that guess.

PyTorch documents automatic backend selection and explicit backend controls:
[official SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).
Installed runtime profiling, rather than the documentation alone, establishes
which backend these actual local tensors use.

Three-sample single-unit forward/backward medians for default bool/aligned-float
are1.046/.695ms at128, .960/.817ms at551, and5.472/5.430ms at3008. These tiny
samples omit model/encoder/optimizer/I/O and are not end-to-end speed or full-model
gradient equivalence evidence. Float-mask preconversion was not adopted.

Artifact dump/native256-fp32-attention-backend-probe-20261018/report.json;
executed script under the corresponding -source directory. All earlier jobs,
checkpoints and source files remain unchanged.
