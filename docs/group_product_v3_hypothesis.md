# Group-product v3: save only the projected features

Registered after v2's terminal hardware failure, before this implementation.
V1 hidden 512 missed memory; v2 hidden 480 saved enough memory but exceeded the
5% slowdown limit. Both fixed implementations are retired without language loss.
The acceptance contract is unchanged. This tests execution, not a new feature map.

Return to hidden 512, 32 groups of 16, the same size and controls as v1. Implement
the exact [v1 forward equation](group_product_hypothesis.md) with a custom autograd
operation that saves only z, which the direct SiLU path already retains. Do not
retain the full neighbor signal for backward. Learned coefficients remain outside
the operation; their ordinary autograd path retains q. The derivative for a group
and upstream t is:

```text
q_i = z_i * (S-z_i) / sqrt(15), S = sum_j z_j
dL/dz_i = (t_i*S - 2*t_i*z_i + sum_j(t_j*z_j)) / sqrt(15)
```

Compute group sums in FP32 during backward (FP64 in mathematical checks), then
cast the input gradient to the feature dtype. The forward uses the same BF16
operations/order as v1. FP64 forward/gradient agreement must be exact within
numerical tolerance; mixed-precision gradients may differ because of reduced
rounding. Report that numerical change. It is an implementation tradeoff, not
evidence that polynomial interactions are new or useful.

Hypothesis: removing one large saved neighbor tensor per layer restores the
memory margin while the direct adjoint avoids enough autograd overhead to keep
training within 5% of both full dense controls. FP32 temporaries and recomputation
could instead erase the memory gain or slow training. Measure, do not assume.

FFN weights remain 2,099,200, 75.17% fewer than full SwiGLU; projection FLOPs
remain 4,194,304/token, excluding reductions/products. Same width-512 four-layer
backbone, context 256, vocabulary 4096, batch eight and BF16. Before GPU profiling,
check direct dense-adjacency gradients and finite differences, full Transformer
causality/locality, nonzero coefficients, counts, initial controls, save/load and
exact CPU checkpoint recovery. Then run the same three alternating-order rounds
on both corpora, 20 warmup plus 100 measured steps, maximum allocated peak and
median throughput. Apply unchanged memory/speed/other-resource gates against both
full controls and <6 GiB reserved. Retire before language if it fails.

If it passes, integrate an equivalent production model, repeat Trainer resume and
CUDA forward/gradient checks, then use the seven v1 variants and budgets on both
corpora. Both dense language margins, plain/square controls, inference removals,
equal tuning, fresh seeds and independent confirmation remain required. Test loss
stays unscored. This implementation inherits v1's prior-art and originality limits.
