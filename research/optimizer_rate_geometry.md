# H048 optimizer accounting: global LR also changes represented-map decay

This calculation explains what changes in the frozen global-rate grid; it does
not modify any running recipe or identify the cause of a validation loss gap.
The [earlier H013 control](product_decay_plan.md) already derived this elementary
factor-decay effect and tested a small-model ablation. It is not a new optimizer.

## Exact decay-only statement

For a bias-free factored projection M = B2 P B1, suppose both factor gradients
and Adam moments are zero and the optimizer applies its decoupled decay step.
Let eta_t be global LR, s_j the LR multiplier, and lambda_j the decay coefficient.
Then in real arithmetic

    B_j(t+1) = (1 - eta_t*s_j*lambda_j) B_j(t)
    M(T) = M(0) product_t product_j (1 - eta_t*s_j*lambda_j).

Proof: each factor is multiplied by a scalar, so the scalars commute with the
fixed permutation and factor matrices. Repeating the step gives the product.
Floating-point rounding means an actual tensor simulation need not be bit-exact
to this scalar expression. Nonzero gradient updates break pure scalar shrinkage;
the following values are not predicted trained norms or observed spectral changes.

At d=384/h=2048/G=8, the actual BlockShuffle LR multipliers are 4 and 4 for
each up/value map, and 4 and 64/3 for the down map. Parameter decay is 0.1 for
every factor. Full dense maps have multiplier 1 and decay 0.1. Narrow uses
product decay with its width-scaled down LR; that single down matrix retains
the dense scalar shrinkage exactly. The four recipes intentionally preserve
their previously selected optimizer treatments.

| Global peak LR | Sum of global rates over 800 steps | Full dense map multiplier | BlockShuffle up/value multiplier | BlockShuffle down multiplier |
|---|---:|---:|---:|---:|
| 0.0012 | 0.523800 | 0.948966018 | 0.657629695 | 0.264996139 |
| 0.0024 | 1.047600 | 0.900532379 | 0.432413404 | 0.070071019 |
| 0.0048 | 2.095200 | 0.810943706 | 0.186871625 | 0.004867419 |

The 80-step warmup and cosine-to-0.1 schedule is taken from the unchanged
trainer. [Raw scalar calculations](../results/optimizer_rate_geometry.json)
record the numbers. The existing product-decay alternative uses
lambda_j = 0.1/(K*s_j), giving (1-eta_t*0.1/K)^K per represented map per step.
For K=2 its difference from dense decay is exactly (eta_t*0.1)^2/4 per step.

## What this does and does not justify

A higher-rate loss change is a complete optimizer-recipe result. It is not an
isolated causal test of activation shape, matrix expressivity or gradient step
size. Conversely, different decay-only contractions do not prove the recipes
are unfair or that a matched-decay version would win. H013 previously found
no benefit in its smaller setting (NLL 3.118254 versus 3.115322); that outcome
does not decide this larger WikiText setting. No new decay trial is added to
H048 after observing the grid. Any future control needs a separate hypothesis,
budget and fixed decision rule.
