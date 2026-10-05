# Hypothesis: a small independent context bank guides a wider FFN

Registered on 2026-10-04 after the completed group-product v4 language screen,
before this implementation or any context-bank measurements. The size, quality,
resource and confirmation requirements are unchanged. This is a constrained
gating experiment, not a claim of an undiscovered primitive.

## What the previous experiment changes

Group products passed the short memory screen but lost to compact SwiGLU and a
learned-square control on both corpora. Their fixed recipe is retired. Zeroing
the trained coefficients worsens loss: the model uses its interactions, but that
does not prove the interactions are a better allocation of parameters.

Its context signal is constrained to a sum of the same features it modulates.
Our next hypothesis is that detectors and context predicates should be learned
independently. A small context bank may preserve gating's usefulness while
spending more of the weight budget on distinct detectors and output directions.
This explanation is a hypothesis; the prior loss results do not establish it.

## Computation

For one normalized token x of width 512:

```text
u = Ux                    # 512 detector features
v = Cx                    # 64 independent context features
p_i = SiLU(u_i)
b_i = tanh(a_i)           # learned per detector, initialized to zero
t_g = tanh(v_g)
r_i = p_i * (1 + b_i*t_group(i))
FFN(x) = D r
```

Eight contiguous detectors share one context feature. All U/C/D/a are independent
across layers. U and C execute as one 576-row input projection; D has 512 input
features. Learned U can arrange detectors into useful groups, but membership is
fixed and no semantic meaning is assigned to a group. C can read every input
coordinate independently of U. No token pooling, softmax, expert dispatch, h-by-h
matrix or modification of attention is introduced.

The learned multiplicative factor lies between zero and two. It can amplify or
suppress existing detector activity without reversing its sign. This prevents
the unbounded neighbor products of the previous recipe, but may also limit useful
interactions. SiLU is the existing unary activation; the context and coefficient
change its response to input. This is not an information-creation claim.

The connection to attention is conditional feature selection with a small shared
context bank. It is not an approximation to dot-product attention. The connection
to a control system is bounded gain modulation. The advantage over ordinary
SwiGLU, if any, must come from the allocation of detectors versus predicates.

## Costs and controls

Four layers: 2,230,272 FFN weights, including 2,048 coefficients, versus full
SwiGLU's 8,454,144 (73.62% fewer). Projection FLOPs are 4,456,448 per token versus
16,908,288 (73.64% fewer); tanh, SiLU, products and gradient reductions are extra.
These arithmetic counts do not imply measured speed or memory improvements.

Language controls, all with the same backbone and budget:

* Full native-activation SwiGLU hidden 1376, fused-activation SwiGLU hidden 1376,
  and GELU hidden 2064. Select the lowest full validation NLL independently per corpus.
* Compact fused-activation SwiGLU hidden 368: 2,260,992 FFN weights, 1.38% more
  than the candidate. Compact GELU hidden 544: 2,228,224, 0.09% fewer. Include
  plain SiLU hidden 512 as an ordinary dense control (6% fewer weights).
* An additive-context control with exactly the same U/C/D/a and
  `r_i = p_i + b_i*t_group(i)`. This spends all context weights on additional
  unary features rather than detector/context conjunctions. Same FP32 arithmetic
  and fused elementwise implementation. Compare it to the multiplicative candidate.

At a=0, product/additive/plain outputs match apart from explicitly checked
mixed-precision rounding. Test nonzero coefficients; identical initial outputs
do not establish correct learned behavior. Require candidate NLL lower than both
the additive and plain controls, as well as both registered dense language margins.

## Execution, verification and failure predictions

Use the bundled runtime CUDA compiler already verified in v4. Fuse the unary
activation, tanh operations, mixing and elementwise adjoint. Save the 576-wide
projected bank and the coefficient vector; backward recomputes unary values.
FP32 activation and gradient arithmetic for BF16 input; final feature/input
gradients cast to BF16. FP64 CPU reference uses ordinary equations. For upstream t:

```text
product: du_i = t_i * SiLU'(u_i) * (1 + b_i*t_g)
         dv_g = sum_i t_i*p_i*b_i*(1-t_g^2) over its eight detectors
         da_i = sum_tokens t_i*p_i*t_g*(1-b_i^2)
additive:du_i = t_i * SiLU'(u_i)
         dv_g = sum_i t_i*b_i*(1-t_g^2)
         da_i = sum_tokens t_i*t_g*(1-b_i^2)
```

Check independent broadcast/dense group-selection equations, CPU FP64 forward
and gradients, finite differences, CUDA FP32/BF16 reference tolerances, parameter
counts, causality, token locality, save/load and CPU exact resume before profiling.
Record numeric differences. No higher-order gradient support is assumed.

Profile product, additive, plain SiLU and all three full controls on both corpora:
three alternating-order rounds, 20 warmup plus 100 measured updates; record cold
warmup/compilation, maximum allocated peak and median throughput. Same width 512,
four layers, 16 heads, vocabulary 4096, context 256, batch eight, BF16, four CPU
threads and RTX 4070 Laptop GPU. Apply >=20% memory reduction OR >=1.2x throughput,
with <=5% deterioration in the other resource against every full control and
<6 GiB reserved. Retire this fixed implementation before language if it fails.

If it passes, integrate one readable complete production Transformer and verify
equivalence and Trainer resume. Train the eight listed variants on both prepared
corpora at seed 101, 2,000 updates, rate 0.0006, warmup 200, AdamW decay 0.1.
Use identical sampled windows and 4,096,000 supervised targets; report full
eligible validation NLL, not test loss. Freeze coefficients to zero with the same
backend after training; also zero half the detector features and replace FFN
output with its mean from 32 fixed training batches, seed 2027. Compare removal
with separately trained plain and additive controls; neither alone proves richer
inference computation. Survivors require equal rate/seed tuning, fresh seeds,
sustained resources and independent confirmation before qualification.

Predicted failures: shared predicates may be too coarse; bounded factors may be
too weak; zero-start coefficients initially block learning C; additive context
may be enough; independent predicates may add no benefit over compact SwiGLU;
gradient reductions or the wider saved bank may erase the resource margin.
Retire if any required gate fails. Do not choose a new group count after looking
at language loss without registering a separate experiment.

## Prior work and originality

[GLU variants](https://arxiv.org/abs/2002.05202) already study products of learned
projections inside Transformer FFNs. [Gated Channel Transformation](https://arxiv.org/abs/1909.11519)
already studies inexpensive channel interaction and modulation in vision.
These precede this study and motivate its controls; their results do not establish
this recipe's usefulness in language. Group-shared predicates are a structural
restriction of learned gating, not a new category of neural computation. An
initial search did not verify priority for this exact allocation. Originality
remains unverified; do not claim discovery from an unsuccessful search or a gain.

The scientific question is whether this particular constrained allocation beats
equally budgeted ordinary FFNs, with a mechanism that survives inference removal
and independent confirmation. The main Atlas goal remains a separate document.
