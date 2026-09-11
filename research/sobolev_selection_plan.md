# H108: select FFN neurons by outputs and directional derivatives

Previous goal turn: progress. H106/H107 added a capacity proof and audited
negative learning evidence. The broader VRAM/quality/parameter goal is unchanged
and unmet. The failed affine residual recipes remain closed.

## Hypothesis and prior art

Selecting teacher neurons by their contribution to both function values and
input sensitivities may preserve a more useful small FFN than random subsets or
value-only selection. Compression physically removes rows/columns; deployment
remains an ordinary dense GELU/SwiGLU FFN with a fitted output bias. It adds no
new activation, gate, full affine bypass, lookup, index gather or derivative
calculation at inference. This is post-training compression, not learning from
scratch or a claimed new scalar nonlinearity.

Feature selection by orthogonal least squares is established
([Chen et al.](https://doi.org/10.1109/72.80341));
[Sobolev training](https://arxiv.org/abs/1706.04859) explicitly distills derivatives.
[GRAIL](https://proceedings.mlr.press/v328/tang26a.html) reconstructs compressed
blocks through calibrated Gram/ridge fits, and
[FLAP](https://ojs.aaai.org/index.php/AAAI/article/view/28960) studies fluctuation
scores, structure allocation and bias compensation. We test the specific
derivative-aware selection/readout combination against these kinds of controls;
the components and Schur-complement identity below are not claimed novel.
The baseline implementations are scoped comparators, not full reproductions
of those papers' multi-layer methods or reported results.

## Objective and a checkable greedy rule

Let H contain centered teacher hidden features, standardized by their training
SD (floor 1e-6), and Y centered outputs normalized by one global training RMS
(floor 1e-6). Let K and T be matching directional derivatives of these features
and normalized outputs. Derivatives are not centered: an output bias has zero
derivative. For a selected neuron set S, fit B by minimizing

    L_S = ||H_S B - Y||_F^2 / N
        + beta ||K_S B - T||_F^2 / N + epsilon ||B||_F^2.

epsilon=1e-4; beta=0 for value fitting or 0.1 / mean(T^2), with denominator floor
1e-12, for the Sobolev objective. The value target's training RMS is one, so this
sets the derivative term's total target energy to 10% of the value target's.
All normalizers depend only on calibration data. The intercept is unpenalized.

Write G=(H' H + beta K' K)/N + epsilon I, C=(H' Y + beta K' T)/N.
Given S, the residual variance and residual cross-target vector for feature j are

    g_j = G_jj - G_jS G_SS^-1 G_Sj
    c_j = C_j - G_jS G_SS^-1 C_S.

Adding j reduces the optimal regularized objective by exactly ||c_j||^2/g_j.
This follows by completing the square in the Schur complement of G_SS. Positive
ridge makes G positive definite. Maximizing this exact one-step gain is greedy;
it does not solve the globally optimal subset problem or guarantee held-out
performance. Rank-one Schur updates implement the same calculation.

For v with independent Rademacher coordinates, E[v v']=I. Thus
E||J diag(sx) v||^2 = ||J diag(sx)||_F^2, where sx is the training input SD.
This gives an unbiased random-direction sensitivity measure in standardized
input coordinates, not a bound on worst-case gradients. One direction per
calibration row and two independent directions per reporting row are used.
Neither Jacobian matching nor a small local MSE proves language NLL stability.

## Data, budgets and controls

Reuse the 18 independently audited H107 input datasets: seed-17 GELU/SwiGLU
teachers at depths 0/3/7, sampling seeds 71/83/97. Use the FIRST 8,192 rows of
each training partition for calibration and the same LAST 4,096 reporting rows.
The H107 selection partition is unused. These are reused development inputs,
not a fresh independent confirmation. No new corpus/teacher training occurs.

Recompute smooth FP32 teacher function values and analytic directional
derivatives; do not pretend that BF16 rounded arithmetic has the same derivative.
The old BF16 output labels are not the H108 targets. Disable TF32. Use CPU
FP64 sufficient statistics/selection/readout solves, four CPU threads, streamed
CUDA feature and derivative evaluation in batches of 512. All input data stays
on CPU between batches. Preserve full preprocessing wall time and peak CUDA
allocation, including the H107 teacher-capture cost in pipeline accounting.

Calibration direction seed=27000+sampling_seed; reporting directions use
29000+sampling_seed and 30000+sampling_seed. Rademacher vectors are multiplied
by the calibration input SD (floor 1e-3), not normalized to unit length. Random
neuron ordering seed=28000+sampling_seed. One deterministic order per selector
is reused across widths. Ties favor the smallest original neuron index.

Eight methods:

1. Random subset, value ridge readout.
2. Output-column weight norm ranking, value ridge readout.
3. Hidden-feature SD times output-column norm (fluctuation), value ridge readout.
4. Pivoted conditional standardized feature variance, value ridge readout.
5. Value-greedy gain selection, value ridge readout.
6. Sobolev-greedy gain selection, Sobolev ridge readout (candidate).
7. Value-greedy selection, Sobolev readout (objective ablation).
8. Sobolev-greedy selection, value readout (selection ablation).

Widths: primary GELU448/SwiGLU298 (over 70% fewer FFN parameters), plus nominal
half-width GELU768/SwiGLU512 and three-quarter GELU1152/SwiGLU768 to measure the
frontier. Actual counts are 2dh+d or 3dh+d including output bias; report exact
savings, not nominal ones. Eight methods x three widths x 18 datasets = 432
compressed endpoints, all produced without SGD or tuning rates/steps.

Before real-data access, qualify analytic GELU/SwiGLU JVPs against autograd and
central finite differences, exported values/JVPs, parameter counts, derivative
isotropy, and every greedy-step gain against independent augmented least squares
on small full-rank and rank-deficient fixtures. Selection sets must be unique.

## Measurements and decisions

Record FP32 value MSE / calibration variance; directional-derivative error /
reporting teacher derivative energy; maximum sampled relative errors; counts;
calibration/selection/readout time; selected indices; stationarity; source/data/
state hashes; and actual no-gradient forward allocation/reservation and median
latency at batch256. Inference: ten warmup calls, five timing blocks of twenty
calls, CUDA synchronization. Profile the original full FFN identically. No Adam
state or training-memory saving is claimed by this zero-SGD study.

The derivative-aware component qualifies only if, for BOTH teachers at a width:
paired geometric-mean derivative error is at least 5% below value-greedy, value
error rises by at most 1%, and each sampling seed satisfies both comparisons.
It must not be Pareto-dominated in both errors by a cheaper static selector.
The two cross-objective ablations determine what caused any gain.

A width earns a separately frozen language-insertion check only if BOTH teacher
families have mean value error <=0.05 and mean derivative error <=0.10, every
sampling seed meets those two absolute thresholds, mean raw inference allocation
is <=90% of full and median-block mean inference latency <=110% of full.
Require finite outputs/JVPs and at least 20% actual parameter reduction; identify
whether the original 70% architectural target is met separately. Value-greedy
is eligible as an established control lead under the same absolute gate. A lower
compression point never counts as achieving the original architectural target.

If the derivative recipe fails the comparative component gate, close that fixed
recipe without changing beta/epsilon, adding probes, or training it. Frontier
and positive control outcomes remain recorded. Any qualified local lead still
needs fresh downstream NLL, full-model memory, longer/broader validation and
independent teacher-training seeds. This screen cannot declare a breakthrough.

Preserve runtime/numerical failures without silent retries. Use the recorded
UV-managed Python3.12.9 environment; PYTHONMALLOC=malloc and PYTHONHASHSEED=107
carry forward the qualified postprocessing settings. Independent audit must
rebuild statistics/derivatives, compare actual selected gains to direct solves
and rescore every deployed checkpoint before a follow-up allocation.
