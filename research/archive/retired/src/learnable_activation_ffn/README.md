# Learnable activation FFNs

**Corrected result:** At the same peak LR 0.0012, the
[three-seed comparison](../../../../affine_rate_replication_results.md) finds only
0.101% lower mean NLL with affine and wins in 2/3 seeds. This fails the frozen
material-benefit gate. The earlier 1.755% advantage compared different rates.
Keep these implemented families and their proofs as research alternatives;
they have not earned promotion over the stronger plain control.

Six variants add small group-shared activation functions to the two compressed
front-runners. The original projection parameters and initialization are preserved.

| Variant | Starting architecture | Added weights per layer, 8 groups |
|---|---|---:|
| swiglu_narrow_shifted | Calibrated narrow SwiGLU | 72 |
| swiglu_narrow_rational | Calibrated narrow SwiGLU | 40 |
| blockshuffle_swiglu_shifted | BlockShuffle SwiGLU | 72 |
| blockshuffle_swiglu_rational | BlockShuffle SwiGLU | 40 |
| swiglu_narrow_affine_activation | Calibrated narrow SwiGLU | 16 |
| blockshuffle_swiglu_affine_activation | BlockShuffle SwiGLU | 16 |

Each FFN is `down(phi(up(x)) * gate(x))`. The learnable shape lives inside the
existing multiplicative gate; this adds no projection or attention layer.
All corrections start at zero, so `phi(z) = SiLU(z)` initially, including the
BF16 output rounding. Training learns shapes through the usual loss gradients.

The [equations and proofs](model.md) explain the bounded Bezier correction and
positive-denominator rational correction. The [domain overview](../../../../learnable_activation_domain.md)
explains related approaches and limits. The [frozen screen](../../../../learnable_activation_plan.md)
defines which evidence can earn more training. These are prior-art-inspired
hypotheses, not established improvements or a verified new primitive.

## Add a curve to an existing trained model

```python
from src.core.activation_retrofit import add_learnable_activations

adapted = add_learnable_activations(trained_base, family="rational")
# family="shifted" chooses the four-coordinate Bezier bank.
# family="affine" learns just a slope and offset around SiLU.
```

This accepts the registered `swiglu_narrow` or `blockshuffle_swiglu` Transformer,
copies its trained weights and adds zero corrections. The source is untouched.
It preserves the current function; it does not improve it until further training.
Create a fresh optimizer for `adapted`, and do not repeat dense width
initialization on the copied weights. Optimizer state is not transferred.
The controlled LM screen trains matched recipes from initialization; it does
not report a fine-tuning gain from this convenience function.

## Reproduce a controlled training trial

From the repository root, this creates a fresh timestamped run directory:

```powershell
uv run --extra compile --extra data python -m src.core.cli train --config configs/wikitext2_blockshuffle_rational_screen.json --cache data/wikitext2_v1 --learning-rate 0.0012
```

The other five configuration files use the same `wikitext2_..._screen.json`
convention. The frozen audit command uses fixed archival names and intentionally
refuses to overwrite completed runs. Use the ordinary CLI for independent repeats.

## Results and optional compiler

The [native screen](../../../../learnable_activation_results.md) finds a
1.216% NLL improvement for rational BlockShuffle over its selected base, with
only 320 added weights across eight layers. This is a single-seed short-budget
finding; eager training memory fails the target.

`--activation-backend inductor_rational` enables experimental pointwise fusion
for rational BlockShuffle on CUDA BF16. It passes isolated memory and gradient
checks but **fails the full training repeat**, so it is not the recommended
training recipe. The [execution](../../../../activation_execution_results.md)
and [trajectory failure](../../../../activation_training_repeat_results.md)
remain separate evidence. Installing this backend preserves parameters and
state-dict keys. Layer diagnostics use the native pointwise reference and then
restore compiled execution.

The affine variants isolate a simpler mechanism: a linear and constant residual
inside the gate. At eight layers they add only 128 weights. The scoped
[origin-Jacobian proof](model.md#h037-affine-correction-and-an-explicit-linear-path)
shows strict expressivity beyond a bias-free SwiGLU layer, without claiming
better optimization or whole-network gradient stability. Their
[frozen retraining ablation](../../../../affine_activation_plan.md) uses the
same three learning rates on both bases.

The completed [affine result](../../../../affine_activation_results.md) selects
BlockShuffle NLL 5.894765 with 718.08 MiB training peak, passing the frozen short
screen. It slightly improves on rational NLL while spending 128 rather than 320
new weights. Narrow affine loses. The simplified architecture now earns longer
training, with the one-seed and tuning limitations retained.

The additional `inductor_rational_correction` backend compiles only the rational
residual, retaining native SiLU/product derivatives. It passes the stricter
[local audit](../../../../activation_correction_execution_results.md), but
its [full repeat also fails](../../../../activation_correction_repeat_results.md):
NLL 5.949104 versus native 5.898003. Keep eager execution for the promoted affine
architecture. Both experimental compiler results are preserved.

## Longer-budget evidence and its limits

The [three-seed WikiText study](../../../../affine_activation_replication_results.md)
finds mean NLL 4.754207 for affine BlockShuffle, 1.755% below selected plain
BlockShuffle, with only 128 extra weights. All frozen quality/memory gates pass.
The recipes use different selected learning rates; the
[completed same-rate control](../../../../affine_rate_replication_results.md)
reduces the measured mean gain to 0.101%. This does not establish an activation-specific training cause.

[Removing the correction after training](../../../../affine_activation_removal_results.md)
costs less than .020% NLL in every seed and preserves quality gates. That result
motivates studying the learned common weights and optimization path. It does
not show that the final scalar shape is necessary for the observed performance.
No all-data optimum, convergence, training-speed improvement or full-network
nonvanishing-gradient guarantee follows. The two richer compiler variants
remain failed training backends despite passing local derivative checks.
