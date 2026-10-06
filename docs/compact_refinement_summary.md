# Compact FFN refinement: selected recipe and evidence

**Historical execution:** all language and resource results in this study used
the optimized implementation. The active code has since been simplified to
ordinary PyTorch. Parameter counts are unchanged, but the old memory and timing
claims do not transfer to it. See the [current execution checks](../src/models/channel_curve_transformer/result.md)
for numerical verification and the measured resource tradeoff.

**Selected recipe: Curve-Wide (`self_curve_wide`).** The three-family screen,
paired-seed confirmation and registered resource measurements are complete.
The independent seed509/8000-update evaluation and all eight held-out tests are
complete and audited. The choice was frozen before that evaluation; held-out
test scores did not select the recipe. This bounded refinement phase is complete.

This phase refines an existing compact FFN through bounded variations. It does
not establish a newly discovered class of neural networks or a universal LLM
improvement. The practical result is a useful quality/size/memory tradeoff in
the measured four-layer language models.

## What was tested

| Starting recipe | Nearby variation | Outcome |
| --- | --- | --- |
| Signed Linear | Smaller detector bank with finer gate groups | Initial gains did not hold: fresh-seed mean loss is slightly worse than its parent on both datasets. |
| Self-Curve | Broader initial slopes and offsets, unchanged equations and parameter count | Improved over its parent in all six fresh-seed comparisons; selected by the combined quality ranking and recorded resource limits. |
| Fixed Basis | Smaller feature groups in the readout | Improved TinyStories but worsened WikiText in the initial screen; did not advance. |

The initial screen contains all 12 runs and frozen feature-removal checks.
Confirmation contains 60 runs: ten recipes, three fresh seeds and two corpora,
with parents, matched dense controls and full dense baselines. No extra tuning
was introduced after partial results. [Screen](compact_refinement_result.md) /
[confirmation](compact_refinement_confirmation_analysis.md).

## How Curve-Wide works

The full Transformer retains its embeddings, causal attention, normalization,
residual paths and output head. Only its FFN recipe changes. A dense input
projection mixes the features. Each resulting coordinate receives three
learnable nonlinear responses; a dense output projection mixes them again.
The selected self-gated mode uses the same coordinate in each response's gate.
It does not use the neighboring-channel shifts of the older coupled-curve mode.

For normalized FFN input x, the equations are:

$$
z=W_{up}x,\qquad y=W_{down}\phi(z),
$$
$$
\phi_j(z)=s_0\left[\frac{1}{\sqrt{3}}\sum_{k=1}^{3}
\operatorname{SiLU}(a_{kj}z_j+b_{kj})(c_{kj}z_j+e_{kj})-\mu_0\right].
$$

The branch parameters a, b, c and e learn independently for each coordinate.
The initial centering and scale are fixed calibration constants. Both projections
are full d-by-d matrices. Cross-feature mixing occurs through these projections;
the nonlinear branches themselves act coordinate-wise.

The refinement changes initial slopes from [0.75, 1, 1.25] to [0.5, 1, 1.5],
and offsets from [-0.5, 0, 0.5] to [-1, 0, 1], with recalibrated centering/scale.
It keeps c=1 and e=0 initially. The hypothesis is that a broader initial response
range improves learning. This is an initialization hypothesis, not a larger
function class. Restoring initial shapes after training does not by itself
identify the cause of the gain.

The FFN uses 2d²+12d parameters per layer. At d=512 with four layers, this is
2,121,728 FFN parameters and 8,417,792 total model parameters, versus full SwiGLU's
8,454,144 and 14,750,208: **74.9% fewer FFN parameters and 42.9% fewer total
parameters**. Reduced projection work does not imply the same percentage of
end-to-end speedup; nonlinear work and the unchanged backbone still cost time.
[Implementation](../src/models/channel_curve_transformer/transformer.py).

## Confirmed development comparison

Full last-checkpoint validation means for seeds211/307/401 at 2000 updates.
Lower NLL is better; percentages of NLL are not accuracy percentages.

| Recipe | TinyStories mean NLL | WikiText mean NLL |
| --- | ---: | ---: |
| Full SwiGLU | 2.542837 | 4.220415 |
| Full GELU | 2.589529 | 4.254606 |
| **Curve-Wide** | **2.592300** | **4.272862** |
| Signed parent | 2.600803 | 4.271632 |
| Curve parent | 2.605145 | 4.282949 |
| Strongest compact dense control, SwiGLU h384 | 2.607348 | 4.282197 |

Curve-Wide ranks first among the compact recipes on the registered combined
score. The signed parent is slightly better on WikiText; Curve-Wide is better on
TinyStories and smaller. Full SwiGLU retains better quality on both datasets.
The h384 compact control has 2,359,296 FFN weights, more than Curve-Wide;
the closer h346 and exactly matched GELU h518 controls are also retained in the
complete confirmation table. Three seeds do not establish statistical significance.

## Resource result and limitation

Three alternating measurement rounds per recipe and corpus give Curve-Wide
403.2/405.2 MiB peak allocated training memory, about 28% below full SwiGLU and
25–26% below full GELU. These are allocated-memory measurements, not total process
GPU usage. Curve-Wide's recorded median time is 3.8–4.0% higher than its parent's,
within the registered 5% limit, at approximately the same allocated memory.

WikiText full-baseline timings varied by more than 2x across rounds. The cause is
unresolved; no reliable speedup is claimed, and no rounds were discarded.
[All timing ranges](compact_refinement_resource_analysis.md).

All four kernel modules match the pre-refinement snapshot byte-for-byte, and no
standalone CUDA/C++ sources were added. Existing kernels were reused unchanged.
[Preservation evidence](../records/compact-refinement-kernel-preservation.json).

## Independent longer evaluation

The WikiText Curve-Wide run's approximately 10-hour elapsed training time includes
Windows standby and hibernation, confirmed by the system event log. Its raw time
is retained but is unsuitable for speed comparisons; no corrected compute time
is inferred. This is separate from the earlier resource-profile variability.
[Timing evidence](../records/compact-final-v1-timing-review.json).

All eight fixed runs at seed509 and 8000 updates are complete, with 16,384,000
training targets each and full last-checkpoint validation and held-out testing.
Lower NLL is better; compare only within a corpus and split.

| Recipe | TinyStories validation | TinyStories test | WikiText validation | WikiText test |
| --- | ---: | ---: | ---: | ---: |
| **Curve-Wide** | **2.122985** | **2.020291** | **3.942322** | **3.966762** |
| Curve parent | 2.125866 | 2.020990 | 3.962206 | 3.986061 |
| Compact SwiGLU h384 | 2.127756 | 2.024020 | 3.946195 | 3.971140 |
| Full SwiGLU | 2.078244 | 1.974934 | 3.991952 | 4.022206 |

Curve-Wide improves over its parent and compact SwiGLU on both validation and
test splits. Against full SwiGLU, its held-out NLL is 2.30% higher on TinyStories
and 1.38% lower on WikiText. This longer single-seed outcome differs from the
shorter three-seed comparison, where full SwiGLU leads on both corpora. It does
not establish universal superiority, equal accuracy or scaling to larger LLMs.
No tuning or reselection followed the held-out results.

The practical goal is achieved in these small Transformers: 74.9% fewer FFN
weights, 42.9% fewer total weights and about 28% less allocated training memory,
with close language quality. Curve-Wide is the selected refinement; reliable
speed improvement remains unproven. The gain over compact SwiGLU is small
(0.18%/0.11% lower held-out NLL), so retain that ordinary control in future work.

[All run results](compact_refinement_independent_result.md) /
[Final integrity audit](../records/compact-refinement-final-audit.json).

[Frozen selection](compact_refinement_final_selection.md) /
[all dataset leaderboards](leaderboard.md) /
[original Step 1 protocol](compact_refinement_hypothesis.md).
