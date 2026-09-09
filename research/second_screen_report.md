# Longer sharing screen and cross-group test - 2026-09-06

The gold target remains unmet. All comparisons below use the same decoder,
TinyStories data/tokenizer, seed 17, BF16 and 800 optimizer steps (1,638,400
training tokens). They are short-training results, not convergence or novelty
claims. The full SwiGLU control is materially stronger than GELU at this budget.

| Run | Unique FFN weights | FFN reduction | NLL | NLL change vs full SwiGLU | Forward tokens/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| swiglu_h608_s17_800 | 1,400,832 | -18.75% | 3.051180 | -0.45% | 78,439 | 266.3 |
| swiglu_s17_800 | 1,179,648 | 0.00% | 3.065045 | +0.00% | 81,813 | 255.3 |
| shared_swiglu_h608_s17_800 | 350,208 | 70.31% | 3.137058 | +2.35% | 83,084 | 251.0 |
| shared_swiglu_affine_s17_800 | 301,056 | 74.48% | 3.139184 | +2.42% | 68,597 | 262.0 |
| shared_swiglu_s17_800 | 294,912 | 75.00% | 3.140172 | +2.45% | 85,060 | 243.4 |
| swiglu_narrow_h152_s17_800 | 350,208 | 70.31% | 3.149475 | +2.75% | 79,562 | 222.5 |
| swiglu_narrow_s17_800 | 294,912 | 75.00% | 3.166492 | +3.31% | 78,919 | 218.9 |
| gelu_s17_800 | 1,179,648 | 0.00% | 3.193122 | +4.18% | 90,569 | 245.6 |
| gelu_narrow_s17_800 | 294,912 | 75.00% | 3.204491 | +4.55% | 82,337 | 215.0 |
| shared_gelu_s17_800 | 294,912 | 75.00% | 3.224598 | +5.21% | 81,427 | 233.5 |

## Decisions
- Sharing GELU looked better than narrow GELU at 200 steps but loses that
  advantage at 800. The early ranking was not reliable enough for acceptance.
- Strict shared SwiGLU h=512 is 2.45% worse than full SwiGLU, with 75% fewer
  unique FFN weights. It remains a serious control, without meeting the target.
- Per-layer affine specialization changes NLL by less than .001 and adds
  substantial eager overhead. Reject promotion in this setting.
- Shared SwiGLU h=608 spends the available 70% reduction allowance but barely
  improves NLL. Its parameter-matched narrow h=152 control is only .0124 worse;
  the compute-matched full h=608 control is .0859 better. Do not widen past the
  frozen cap. The candidate uses 18.75% more FFN matrix FLOPs than full h=512.

## Cross-group coupling: useful proof, failed LM screen
A fixed neighboring-group mixture inside the structured value gate restores
cross-group mixed derivatives without adding trainable weights. A constructive
polynomial separation and mixer singular-value bounds are checked independently
in tests/test_coupled.py. See [proof and frozen protocol](cross_group_hypothesis.md).

At 200 steps, coupled SwiGLU NLL=4.635548 versus uncoupled grouped SwiGLU=4.648195.
Both are well behind narrow GELU=4.241118 and narrow SwiGLU=4.419071. Coupled
forward throughput is 68551 tokens/s and peak allocated memory is 248.76 MiB.
This does not earn a longer LM run. The completed [three-seed function report](interaction_report.md) is a mechanism
diagnostic, not a way to overturn this negative LM result. The later adaptive
variant reaches NLL 4.630757 and also fails promotion.

## Limits and artifacts
These are single-seed LM screens on one narrow dataset and one laptop GPU;
no confidence interval or publication-level generalization claim follows.
Almost all steps of the strongest compressed runs clip at norm 1; finite
post-clip gradients do not establish robust gradients at larger scale. Sharing
aggregates parameter gradients across uses, and the logs say so explicitly.
Timing measures eager full-sequence forward passes, without a KV cache;
laptop timing variation makes small throughput differences inconclusive.
Source snapshots, checkpoints, histories, data hashes and raw metrics remain
in results/runs. [First three-seed report](first_screen_report.md) preserves the
initial learned-activation results. Learned affine curves and all coefficients
are saved with the affine checkpoint. No failed branch has been removed.
