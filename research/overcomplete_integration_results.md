# H054: Registered Transformer integration; eager memory gate fails

The overcomplete headwise module is now registered as `overcomplete_headwise_swiglu`.
All 31 prior model signatures remain exactly equal: initialized weights, logits,
loss, all gradients and matrix FLOPs. Every new layer exactly matches its
independently initialized standalone module, and all non-FFN tensors match full
SwiGLU. Parent calibration runs after generic nested-factor initialization.

**The eager candidate does not earn a quality screen.** It passes numerical and
parameter checks, but exceeds the frozen memory allowance against full GELU.

| Complete Transformer | FFN weights | Total weights | Allocated training peak MiB | BF16/FP32 logit relative L2 |
|---|---:|---:|---:|---:|
| Full SwiGLU | 9,437,184 | 15,735,168 | 706.733 | .007529 |
| Overcomplete headwise | 2,801,664 | 9,099,648 | 730.389 | .008384 |
| Full GELU | 9,437,184 | 15,735,168 | 660.108 | .007674 |

The candidate is 3.347% above SwiGLU and **10.647% above GELU**, missing the
latter allowance by **4.270 MiB**. The earlier 20.37% isolated-FFN memory cost
was a useful concern but not a complete-model estimate. Fewer parameters do
not imply fewer saved activations. No inference speed was measured here.

The [frozen plan](overcomplete_integration_plan.md) fixes d384/L8, context128,
batch16, native BF16, uniform LR multipliers, ordinary parameter decay, AdamW
with constant LR .0012 and clip1. Each fresh cell repeats the same first CUDA
WikiText training batch for 20 updates. The three cells total **122,880 training
token exposures**, but only 2,048 unique targets per model. This is pipeline and
memory qualification, with **zero validation/test targets scored**, not a
language-quality, convergence or relative training-speed result.

Every weight, gradient, Adam moment and recorded layer diagnostic stays finite;
every parameter has a nonzero initial gradient. All checkpoint round trips
preserve the final fixed-batch logits exactly without recalibration. The sampler
and data hashes match. Clipping occurs in 55%/65%/50% of SwiGLU/candidate/GELU
updates; these short repeated-batch frequencies are diagnostic only.

All **241 tests pass**, with the full [test record](../results/verification/overcomplete_integration_tests_v1.json)
and source archive retained. The [raw outcome](../results/overcomplete_integration_v1/result.json)
and three per-cell histories/checkpoints record the failed gate. Existing 165
LM/profile runs remain; these qualifications are kept in their own directory.
There are now 32 registered variants in the same 11 architecture folders.

The candidate uses m=4*d/3, H=`groups`, G=2*H and k=`hidden`. The frozen case
uses groups=8/hidden=200. With hidden=0, configuration chooses the largest multiple
of 8 within the 30% FFN-weight ceiling, with a minimum 8 for tiny diagnostic cases;
that tiny-case minimum need not meet 70% compression. Explicit hidden values
remain authoritative. No serialized model fields were added.

The next justified execution ablation is the repository's existing native
SwiGLU gate recomputation. It can discard a saved activation while preserving
the same real-arithmetic function and parameter count. It must earn its own
forward/gradient/memory checks and frozen qualification; no such change was
included in this result. A richer activation or router is not justified by a
memory-only failure. The full research goal remains unmet.
